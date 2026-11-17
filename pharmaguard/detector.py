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

    @property
    def count(self) -> int:
        return len(self.xyxy)

    def is_empty(self) -> bool:
        return self.count == 0


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
        # Warmup or push to target device
        self.model.to(self.device)

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
            verbose=False
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

        return DetectionResult(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            class_name=class_name
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "model_name": self.config.model_name,
            "device": self.device,
            "confidence_threshold": self.config.confidence_threshold,
            "iou_threshold": self.config.iou_threshold,
            "target_classes": self.config.target_classes,
        }
