"""
Detector module for PharmaGuard.
Provides an abstract base interface and concrete YOLO detector implementation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import torch

from pharmaguard.config import DetectorConfig


@dataclass
class DetectionResult:
    """Standardized detection result output."""
    # Bounding boxes in format [x_min, y_min, x_max, y_max] shape (N, 4)
    xyxy: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=np.float32))
    confidence: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    class_id: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int32))
    class_name: List[str] = field(default_factory=list)
    # Optional COCO-17 keypoints from pose models, shape (N, 17, 3) = (x, y, confidence) in pixels.
    keypoints: Optional[np.ndarray] = None

    @property
    def count(self) -> int:
        return len(self.xyxy)

    def is_empty(self) -> bool:
        return self.count == 0

    def subset(self, mask: np.ndarray) -> "DetectionResult":
        """Rows selected by a boolean mask or index array, keypoints kept aligned."""
        idx = np.arange(self.count)[mask]
        return DetectionResult(
            xyxy=self.xyxy[idx], confidence=self.confidence[idx], class_id=self.class_id[idx],
            class_name=[self.class_name[i] for i in idx] if self.class_name else [],
            keypoints=None if self.keypoints is None else self.keypoints[idx],
        )


def suppress_duplicate_poses(result: "DetectionResult", kp_conf: float = 0.3, dist_frac: float = 0.08,
                             min_shared: int = 3) -> "DetectionResult":
    """Remove detections that are the same body as another detection.

    Two detections are the same person when at least ``min_shared`` keypoints are confidently visible in both
    and their median distance is below ``dist_frac`` x the taller box's height. Of each duplicate pair the one
    with more visible keypoints is kept (ties: higher confidence), so the fuller skeleton (hips for the pocket
    check) survives. Boxes alone cannot tell this apart from two people overlapping; skeletons can.
    """
    if result.keypoints is None or result.count < 2:
        return result
    kps = result.keypoints
    visible = kps[:, :, 2] >= kp_conf
    n_vis = visible.sum(axis=1)
    order = sorted(range(result.count), key=lambda i: (-int(n_vis[i]), -float(result.confidence[i])))
    keep: List[int] = []
    for i in order:
        dup = False
        for j in keep:
            shared = visible[i] & visible[j]
            if int(shared.sum()) < min_shared:
                continue
            d = np.linalg.norm(kps[i, shared, :2] - kps[j, shared, :2], axis=1)
            h = max(result.xyxy[i, 3] - result.xyxy[i, 1], result.xyxy[j, 3] - result.xyxy[j, 1], 1.0)
            if float(np.median(d)) < dist_frac * h:
                dup = True
                break
        if not dup:
            keep.append(i)
    if len(keep) == result.count:
        return result
    return result.subset(np.array(sorted(keep), dtype=np.int64))


# COCO-17 keypoint indices used by the behaviour engine.
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW, KP_R_ELBOW = 7, 8
KP_L_WRIST, KP_R_WRIST = 9, 10
KP_L_HIP, KP_R_HIP = 11, 12


class BaseDetector(ABC):
    """Abstract base class for object detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run detection on a single RGB/BGR image frame.

        Args:
            frame: Numpy array representing image (H, W, C).

        Returns:
            DetectionResult containing bounding boxes, confidences, and class labels.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded model."""
        pass


class YOLODetector(BaseDetector):
    """YOLO detector implementation leveraging Ultralytics.

    Supports YOLOv8, YOLOv9, YOLOv10, YOLO11, and RT-DETR weights.
    Filters specifically for configured target classes (e.g., person = 0).
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self.device = self._resolve_device(self.config.device)
        # Auto inference size: CPU is the bottleneck locally (measured 90 ms @640 vs 61 ms @480, same detections).
        self.imgsz = self.config.imgsz or (640 if self.device.startswith("cuda") else 480)
        self.model = None
        self._load_model()

    def _resolve_device(self, requested_device: str) -> str:
        """Resolve target compute device (cuda / cpu)."""
        if requested_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return requested_device

    def _load_model(self) -> None:
        """Load pretrained model weights."""
        from ultralytics import YOLO
        self.model = YOLO(self.config.model_name)
        self.model.to(self.device)
        self.half = bool(self.config.half_precision and self.device.startswith("cuda"))
        self._precision_kwargs = self._resolve_precision_kwargs(self.half)
        # Warm-up: the first CUDA inference pays kernel/cuDNN initialisation (often >1 s).
        # Pay it at startup, not on the first live webcam frame.
        self.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    @staticmethod
    def _resolve_precision_kwargs(half: bool) -> Dict[str, Any]:
        """FP16 flag across Ultralytics versions: 8.4+ uses ``quantize=16``; older uses ``half=True``."""
        if not half:
            return {}
        try:
            from ultralytics.cfg import get_cfg
            if "quantize" in vars(get_cfg()):
                return {"quantize": 16}
        except Exception:  # pragma: no cover - defensive against Ultralytics internals moving
            pass
        return {"half": True}

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run inference on the frame and filter for target classes."""
        if frame is None or frame.size == 0:
            return DetectionResult()

        # Run inference with verbose=False for speed
        results = self.model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            classes=self.config.target_classes,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
            **self._precision_kwargs
        )

        if not results or len(results) == 0:
            return DetectionResult()

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return DetectionResult()

        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        confidence = boxes.conf.cpu().numpy().astype(np.float32)
        class_id = boxes.cls.cpu().numpy().astype(np.int32)
        names = result.names or {0: "person"}
        class_name = [names.get(int(cid), f"class_{cid}") for cid in class_id]
        keypoints = None
        kp = getattr(result, "keypoints", None)
        if kp is not None and getattr(kp, "data", None) is not None and len(kp.data) == len(xyxy):
            keypoints = kp.data.cpu().numpy().astype(np.float32)
            if keypoints.shape[-1] == 2:  # some exports omit confidence
                keypoints = np.concatenate([keypoints, np.ones(keypoints.shape[:-1] + (1,), np.float32)], -1)

        out = DetectionResult(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            class_name=class_name,
            keypoints=keypoints,
        )
        if self.config.dedupe_poses:
            out = suppress_duplicate_poses(out, dist_frac=self.config.dedupe_kp_dist,
                                           min_shared=self.config.dedupe_min_shared)
        return out

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "model_name": self.config.model_name,
            "device": self.device,
            "imgsz": self.imgsz,
            "half_precision": self.half,
            "has_keypoints": bool(getattr(self.model, "task", "") == "pose"),
            "confidence_threshold": self.config.confidence_threshold,
            "iou_threshold": self.config.iou_threshold,
            "target_classes": self.config.target_classes,
        }
