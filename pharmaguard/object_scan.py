"""
Background object scan for the operator display ("every object").

The open-vocabulary detector is several times slower than the pose model (measured 350-570 ms per 640 px frame on
the dev laptop CPU), so it must never sit on the per-frame path. ``AsyncObjectScanner`` keeps only the newest frame,
scans it on a worker thread, and publishes the result; the pipeline reads the latest result every frame. Person
tracking speed is unaffected; object boxes refresh as fast as the model allows.

DISPLAY ONLY: results are drawn and sent to clients but are never given to the behaviour engine or alerts.
Boxes are stored normalized to [0, 1], so they can be drawn on any later frame from the same camera.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScannedObject:
    box: List[float]        # [x1, y1, x2, y2] normalized to [0, 1]
    label: str              # display label: a known name, or "object" when the model is unsure
    raw_name: str           # what the model actually said (kept for review/diagnostics)
    conf: float
    key: bool = False       # label is one of the configured key objects (shown in the uncluttered 'key' view)


def visible(objects: List[ScannedObject], show: str) -> List[ScannedObject]:
    """Objects the operator should see for a display mode: 'key', 'all' or 'off'."""
    if show == "off":
        return []
    return list(objects) if show == "all" else [o for o in objects if o.key]


def postprocess(result: Any, frame_w: int, frame_h: int, cfg: Any) -> List[ScannedObject]:
    """Turn a DetectionResult into display objects: drop people/body parts and scene-sized boxes, merge synonyms,
    and label uncertain detections plainly as 'object'."""
    exclude = {n.lower() for n in cfg.exclude_names}
    aliases = {k.lower(): v for k, v in cfg.aliases.items()}
    key_names = {n.lower() for n in getattr(cfg, "key_names", [])}
    names = result.class_name or ["object"] * result.count
    out: List[ScannedObject] = []
    for i in range(result.count):
        raw = str(names[i]).strip().lower()
        if raw in exclude:
            continue
        x1, y1, x2, y2 = [float(v) for v in result.xyxy[i]]
        nx1, ny1 = max(0.0, x1 / frame_w), max(0.0, y1 / frame_h)
        nx2, ny2 = min(1.0, x2 / frame_w), min(1.0, y2 / frame_h)
        if nx2 <= nx1 or ny2 <= ny1 or (nx2 - nx1) * (ny2 - ny1) > cfg.max_box_area:
            continue
        conf = float(result.confidence[i])
        label = aliases.get(raw, raw) if conf >= cfg.name_confidence else "object"
        out.append(ScannedObject([round(nx1, 4), round(ny1, 4), round(nx2, 4), round(ny2, 4)], label, raw,
                                 round(conf, 3), key=label.lower() in key_names))
    out.sort(key=lambda o: (not o.key, -o.conf))    # key objects first, so max_items never drops them
    return out[: cfg.max_items]


class AsyncObjectScanner:
    """Scan the newest submitted frame on a background thread; read the latest result at any time."""

    def __init__(self, detector: Any, cfg: Any, name: str = "object-scan"):
        self._detector = detector
        self._cfg = cfg
        self._cond = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._objects: List[ScannedObject] = []
        self._scanned_at = 0.0
        self.last_ms = 0.0
        self.scans = 0
        self._stop = False
        self._last_start = 0.0
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, frame_bgr: np.ndarray) -> None:
        """Offer a frame; replaces any frame not yet scanned (only the newest matters)."""
        with self._cond:
            self._frame = frame_bgr
            self._cond.notify()

    def latest(self) -> List[ScannedObject]:
        with self._cond:
            return list(self._objects)

    def age_sec(self) -> Optional[float]:
        with self._cond:
            return None if not self._scanned_at else time.monotonic() - self._scanned_at

    def clear(self) -> None:
        with self._cond:
            self._frame = None
            self._objects = []
            self._scanned_at = 0.0

    def close(self, timeout: float = 5.0) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()
        self._thread.join(timeout)

    def _run(self) -> None:
        while True:
            with self._cond:
                # Rate limit (shares the CPU with person tracking), then wait for a frame.
                wait_until = self._last_start + float(getattr(self._cfg, "min_interval_sec", 0.0))
                while not self._stop and (self._frame is None or time.monotonic() < wait_until):
                    self._cond.wait(timeout=max(0.01, wait_until - time.monotonic()) if self._frame is not None else None)
                if self._stop:
                    return
                frame, self._frame = self._frame, None
                self._last_start = time.monotonic()
            try:
                t0 = time.perf_counter()
                result = self._detector.detect(frame)
                h, w = frame.shape[:2]
                objects = postprocess(result, w, h, self._cfg)
                ms = (time.perf_counter() - t0) * 1000.0
            except Exception:  # noqa: BLE001 - a failed scan must not kill the display thread
                logger.exception("Object scan failed")
                continue
            with self._cond:
                self._objects = objects
                self._scanned_at = time.monotonic()
                self.last_ms = ms
                self.scans += 1
