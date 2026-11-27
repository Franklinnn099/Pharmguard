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

import json
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


# One JSON file per camera: zones/<camera_id>.json. PG_ZONES_DIR moves them (e.g. to Google Drive on Colab).
ZONES_DIR = Path(os.environ.get("PG_ZONES_DIR") or Path(__file__).resolve().parent / "zones")
MAX_ZONES = 8
MAX_POINTS = 24
MIN_AREA = 0.002            # fraction of the image; same floor as the dashboard's zone editor


def validate_zones(raw: Any) -> "tuple[List[Dict[str, Any]], str]":
    """Check zones sent by the editor. Returns (clean zone dicts, error message or "")."""
    if not isinstance(raw, list):
        return [], "zones must be a list"
    if len(raw) > MAX_ZONES:
        return [], f"at most {MAX_ZONES} zones"
    out: List[Dict[str, Any]] = []
    for i, z in enumerate(raw):
        if not isinstance(z, dict):
            return [], f"zone {i + 1} is not an object"
        name = "".join(c for c in str(z.get("name", "")).strip() if c.isalnum() or c in " _-")[:40].strip()
        if not name:
            return [], f"zone {i + 1} needs a name"
        if any(o["name"] == name for o in out):
            return [], f"two zones are named '{name}'"
        pts = z.get("polygon_normalized")
        if not isinstance(pts, list) or not 3 <= len(pts) <= MAX_POINTS:
            return [], f"zone '{name}' needs 3 to {MAX_POINTS} corners"
        try:
            poly = [[round(min(1.0, max(0.0, float(p[0]))), 4), round(min(1.0, max(0.0, float(p[1]))), 4)]
                    for p in pts]
        except (TypeError, ValueError, IndexError):
            return [], f"zone '{name}' has an invalid corner"
        area = 0.5 * abs(sum(poly[k][0] * poly[k - 1][1] - poly[k - 1][0] * poly[k][1] for k in range(len(poly))))
        if area < MIN_AREA:
            return [], f"zone '{name}' is too small or its corners are in a line"
        entry: Dict[str, Any] = {"name": name, "polygon_normalized": poly}
        if "prolonged_presence_sec" in z:
            try:
                entry["prolonged_presence_sec"] = max(1.0, float(z["prolonged_presence_sec"]))
            except (TypeError, ValueError):
                pass
        out.append(entry)
    return out, ""


def _zones_file(camera_id: str) -> Path:
    return ZONES_DIR / f"{camera_id}.json"   # camera_id is already sanitized to [A-Za-z0-9_-]


def load_saved_zones(camera_id: str) -> Optional[List[Dict[str, Any]]]:
    f = _zones_file(camera_id)
    if not f.exists():
        return None
    try:
        zones, err = validate_zones(json.loads(f.read_text(encoding="utf-8")).get("zones"))
    except (OSError, ValueError, AttributeError) as exc:
        logger.warning("Ignoring unreadable zone file %s: %s", f, exc)
        return None
    if err:
        logger.warning("Ignoring invalid zone file %s: %s", f, err)
        return None
    return zones


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
        self._shared_items = None
        self.items_model = "off"
        self.items_show = "off"
        self.send_width = 640
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
        saved = load_saved_zones(camera_id)      # zones drawn for this camera override the config file's
        if saved is not None:
            from pharmaguard.config import ShelfZone
            cfg.zones = [ShelfZone(**z) for z in saved]
        if os.environ.get("PG_NO_RECORDING") == "1":
            cfg.recording.enabled = False
        # "Every object" scan (YOLOE prompt-free), display only. On by default for phone cameras; PG_ITEMS=0 = off.
        cfg.items.enabled = os.environ.get("PG_ITEMS", "1") != "0"
        if os.environ.get("PG_ITEMS_IMGSZ"):
            cfg.items.imgsz = int(os.environ["PG_ITEMS_IMGSZ"])     # e.g. 960 for smaller objects (slower)
        if os.environ.get("PG_ITEMS_INTERVAL"):
            cfg.items.min_interval_sec = float(os.environ["PG_ITEMS_INTERVAL"])  # seconds between scans; 0 = flat out
        if os.environ.get("PG_ITEMS_SHOW"):
            cfg.items.show = os.environ["PG_ITEMS_SHOW"]            # key | all | off (for /view and the default)
        if os.environ.get("PG_ITEMS_MODEL"):
            cfg.items.model_name = os.environ["PG_ITEMS_MODEL"]
        cfg.items.model_name = _resolve_model(cfg.items.model_name)
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
        if cfg.items.enabled:
            from pharmaguard.detector import make_item_detector
            logger.info("Loading object scan model %s (imgsz %d) ...", cfg.items.model_name, cfg.items.imgsz)
            self._shared_items = _LockedDetector(make_item_detector(
                cfg.items, device=cfg.detector.device, half_precision=cfg.detector.half_precision))
            self.items_model = f"{Path(cfg.items.model_name).name} @{cfg.items.imgsz}"
            # The phone uploads frames at least as wide as the scan resolution, or small objects lose their pixels.
            self.send_width = max(640, cfg.items.imgsz)
            self.items_show = cfg.items.show
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
                cfg.objects.enabled = False          # don't load extra models per camera; shared ones injected below
                items_on, cfg.items.enabled = cfg.items.enabled, False
                p = PharmaGuardPipeline(config=cfg, detector=self._shared_detector)
                if items_on and self._shared_items is not None:
                    p.enable_object_scan(self._shared_items)
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
        bag_conf = overlay.get("bag_conf", [])
        for i, b in enumerate(overlay.get("bags", [])):
            dets.append({"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "label": "bag",
                         "conf": float(bag_conf[i]) if i < len(bag_conf) else 0.0,   # real detector score
                         "alert": False, "level": "NORMAL"})
        for it in overlay.get("items", []):          # display only: never flagged, never used for alerts
            b = it["box"]
            dets.append({"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "label": str(it["label"]),
                         "name": str(it.get("name", it["label"])), "conf": float(it["conf"]),
                         "alert": False, "level": "NORMAL", "kind": "item", "key": bool(it.get("key", False))})
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

    # ------------------------------------------------------------------ zones (per camera)
    def get_zones(self, camera_id: str) -> Dict[str, Any]:
        """Zones in effect for a camera: its saved zones, else the config file's (shared default)."""
        p = self._pipelines.get(camera_id)
        if p is not None:
            zones = [z.model_dump() for z in p.config.zones]
        else:
            saved = load_saved_zones(camera_id)
            zones = saved if saved is not None else self._default_zones()
        return {"camera_id": camera_id, "zones": zones,
                "source": "camera" if _zones_file(camera_id).exists() else "default"}

    def _default_zones(self) -> List[Dict[str, Any]]:
        try:
            from pharmaguard.config import PharmaGuardConfig
            cfg = PharmaGuardConfig.load_from_yaml(self._config_path) if self._config_path else PharmaGuardConfig()
            return [z.model_dump() for z in cfg.zones]
        except ImportError:      # "none" mode without pydantic/pharmaguard deps
            return []

    def set_zones(self, camera_id: str, zones: List[Dict[str, Any]]) -> None:
        """Save a camera's zones (validated) and apply them to its running pipeline on the next frame."""
        ZONES_DIR.mkdir(parents=True, exist_ok=True)
        f = _zones_file(camera_id)
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"camera_id": camera_id, "zones": zones}, indent=2), encoding="utf-8")
        os.replace(tmp, f)
        p = self._pipelines.get(camera_id)
        if p is not None:
            from pharmaguard.config import ShelfZone
            p.set_zones([ShelfZone(**z) for z in zones])
        logger.info("Zones for camera '%s' set to %s", camera_id, [z["name"] for z in zones])

    def reset_zones(self, camera_id: str) -> None:
        """Forget a camera's own zones and go back to the config file's."""
        f = _zones_file(camera_id)
        if f.exists():
            f.unlink()
        p = self._pipelines.get(camera_id)
        if p is not None:
            from pharmaguard.config import ShelfZone
            p.set_zones([ShelfZone(**z) for z in self._default_zones()])

    def raw_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Last un-annotated RGB frame (what the zone editor draws on), or None."""
        p = self._pipelines.get(camera_id)
        return None if p is None else p.last_raw_frame

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
