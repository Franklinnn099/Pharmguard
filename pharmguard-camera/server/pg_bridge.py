"""
Bridge between the phone-camera server and the PharmaGuard pipeline (../../pharmaguard).

Each phone (camera_id) gets its own ``PharmaGuardPipeline``: tracker, behaviour engine, alert manager, alarm
and evidence recorder are all per camera, so one phone's people never mix with another's. The heavy models
(pose detector, bag detector) are loaded once and shared between cameras behind a lock, so adding a phone
does not load another copy of YOLO into memory.

Detector modes (env ``PG_DETECTOR``):
    pharmaguard  full pipeline: pose detection -> ByteTrack -> hand/shelf/bag behaviour -> alerts (default)
    none         no inference; frames are accepted, counted and shown, detections are always []
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("pharmguard-camera.bridge")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

Detection = Dict[str, Any]   # {x1, y1, x2, y2, label, conf, alert, ...}; coordinates normalized to [0, 1]


@dataclass
class FrameResult:
    detections: List[Detection]
    annotated_rgb: Optional[np.ndarray]     # what /view/{id} shows (None in "none" mode -> raw frame)
    width: int
    height: int
    processing_ms: float
    alarm: str = "IDLE"


@dataclass
class CameraStats:
    camera_id: str
    connected: bool = False
    frames: int = 0
    skipped: int = 0
    last_frame_at: float = 0.0
    processing_ms: float = 0.0
    width: int = 0
    height: int = 0
    persons: int = 0
    flagged: int = 0
    alarm: str = "IDLE"
    connected_at: float = field(default_factory=time.time)


def _resolve_model(name: str) -> str:
    """Prefer weights already in the repo root, so the server never re-downloads them into server/."""
    p = Path(name)
    if not p.is_absolute() and not p.exists() and (REPO_ROOT / name).exists():
        return str(REPO_ROOT / name)
    return name


class PharmaGuardBridge:
    """Per-camera PharmaGuard pipelines with shared, lock-protected models."""

    def __init__(self, mode: str = "pharmaguard", config_path: Optional[str] = None):
        self.mode = mode
        self._pipelines: Dict[str, Any] = {}
        self._create_lock = threading.Lock()
        self._config_path = config_path
        self._shared_detector = None
        self._shared_objects = None
        self.model_name = "none"
        if mode == "pharmaguard":
            self._load_models()

    # ------------------------------------------------------------------ setup
    def _make_config(self, camera_id: str):
        from pharmaguard.config import PharmaGuardConfig
        if self._config_path:
            cfg = PharmaGuardConfig.load_from_yaml(self._config_path)
        else:
            cfg = PharmaGuardConfig()
        cfg.detector.model_name = _resolve_model(cfg.detector.model_name)
        cfg.objects.model_name = _resolve_model(cfg.objects.model_name)
        cfg.recording.camera_id = camera_id      # every stored event says which phone saw it
        if os.environ.get("PG_NO_RECORDING") == "1":
            cfg.recording.enabled = False
        return cfg

    def _load_models(self) -> None:
        from pharmaguard.detector import YOLODetector
        from pharmaguard.config import DetectorConfig
        cfg = self._make_config("_shared")
        logger.info("Loading PharmaGuard detector %s ...", cfg.detector.model_name)
        self._shared_detector = _LockedDetector(YOLODetector(cfg.detector))
        if cfg.objects.enabled:
            oc = cfg.objects
            self._shared_objects = _LockedDetector(YOLODetector(DetectorConfig(
                model_name=oc.model_name, confidence_threshold=oc.confidence_threshold,
                target_classes=oc.target_classes, device=cfg.detector.device,
                imgsz=cfg.detector.imgsz, half_precision=cfg.detector.half_precision)))
        info = self._shared_detector.get_model_info()
        self.model_name = f"{Path(str(info.get('model_name'))).name} on {info.get('device')}"
        if not info.get("has_keypoints"):
            logger.warning("Detector has no pose keypoints: hand/shelf/bag behaviour analysis is disabled.")

    def pipeline_for(self, camera_id: str):
        """Get (or lazily create) the camera's own pipeline."""
        p = self._pipelines.get(camera_id)
        if p is not None:
            return p
        with self._create_lock:
            p = self._pipelines.get(camera_id)
            if p is None:
                from pharmaguard.pipeline import PharmaGuardPipeline
                cfg = self._make_config(camera_id)
                cfg.objects.enabled = False          # don't load a second bag model; shared one injected below
                p = PharmaGuardPipeline(config=cfg, detector=self._shared_detector)
                if self._shared_objects is not None:
                    p.config.objects.enabled = True
                    p.object_detector = self._shared_objects
                self._pipelines[camera_id] = p
                logger.info("Created PharmaGuard pipeline for camera '%s'", camera_id)
        return p

    # ------------------------------------------------------------------ per frame
    def process(self, camera_id: str, frame_rgb: np.ndarray) -> FrameResult:
        t0 = time.perf_counter()
        if self.mode != "pharmaguard":
            h, w = frame_rgb.shape[:2]
            return FrameResult([], None, w, h, (time.perf_counter() - t0) * 1000.0)

        from pharmaguard.behavior import LEVEL_NAMES, LEVEL_REVIEW
        p = self.pipeline_for(camera_id)
        out = p.process_frame(frame_rgb)
        h, w = out.annotated_frame_rgb.shape[:2]
        dets: List[Detection] = []
        overlay = out.metadata.get("overlay") or {}
        for entry in overlay.get("persons", []):
            tid = int(entry["id"])
            x1, y1, x2, y2 = entry["box"]
            level = int(entry.get("level", 0))
            dets.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "label": f"person #{int(tid)}",
                "conf": float(entry.get("conf", 0.0)),
                "alert": level >= LEVEL_REVIEW,          # REVIEW or ALERT = suspicious, for staff review
                "level": LEVEL_NAMES.get(level, "NORMAL"),
                "track_id": int(tid),
                "score": float(entry.get("score", 0.0)),
                "occluded": bool(entry.get("occluded", False)),
            })
        for b in overlay.get("bags", []):
            dets.append({"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "label": "bag", "conf": 1.0,
                         "alert": False, "level": "NORMAL"})
        return FrameResult(dets, out.annotated_frame_rgb, w, h, (time.perf_counter() - t0) * 1000.0,
                           alarm=str(overlay.get("alarm", "IDLE")))

    def alarm_snapshot(self, camera_id: str) -> Optional[Dict[str, Any]]:
        p = self._pipelines.get(camera_id)
        if p is None:
            return None
        s = p.alarm.snapshot()
        return {"state": s.state, "sounding": bool(s.sounding), "sound_seq": int(s.sound_seq), "kind": s.kind,
                "acknowledged_by": s.acknowledged_by,
                "active": [{"title": a.title, "message": a.message, "track_id": a.tracker_id,
                            "time": a.timestamp_iso} for a in s.active_alerts]}

    def acknowledge(self, camera_id: str, operator: str) -> bool:
        p = self._pipelines.get(camera_id)
        return bool(p and p.alarm.acknowledge(operator=operator))

    def reset_alarm(self, camera_id: str, operator: str) -> bool:
        p = self._pipelines.get(camera_id)
        return bool(p and p.alarm.reset(operator=operator))

    def close(self) -> None:
        for p in list(self._pipelines.values()):
            try:
                p.close()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                logger.exception("Closing pipeline failed")


class _LockedDetector:
    """Shares one model between camera pipelines; ultralytics predict() is not thread-safe."""

    def __init__(self, inner: Any):
        self._inner = inner
        self._lock = threading.Lock()

    def detect(self, frame: np.ndarray):
        with self._lock:
            return self._inner.detect(frame)

    def get_model_info(self) -> Dict[str, Any]:
        return self._inner.get_model_info()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def make_bridge(mode: Optional[str] = None, config_path: Optional[str] = None) -> PharmaGuardBridge:
    """PG_DETECTOR=pharmaguard|none. Falls back to 'none' (with a loud warning) if torch/ultralytics are missing."""
    mode = (mode or os.environ.get("PG_DETECTOR", "pharmaguard")).strip().lower()
    if mode not in ("pharmaguard", "none"):
        logger.warning("Unknown PG_DETECTOR=%r; using 'pharmaguard'", mode)
        mode = "pharmaguard"
    if mode == "pharmaguard":
        try:
            return PharmaGuardBridge("pharmaguard", config_path)
        except ImportError as exc:
            logger.error("PharmaGuard pipeline unavailable (%s). Running with PG_DETECTOR=none: "
                         "NO detection is performed.", exc)
            return PharmaGuardBridge("none")
    return PharmaGuardBridge("none")
