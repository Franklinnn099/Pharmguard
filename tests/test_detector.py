"""
Unit tests for PharmaGuard detector module.
"""

import numpy as np
import pytest
from pharmaguard.config import DetectorConfig
from pharmaguard.detector import YOLODetector, DetectionResult


def test_detector_initialization():
    config = DetectorConfig(model_name="yolov8n.pt", device="cpu")
    detector = YOLODetector(config)
    info = detector.get_model_info()

    assert info["model_name"] == "yolov8n.pt"
    assert info["device"] == "cpu"
    assert info["target_classes"] == [0]


def test_detector_inference_on_empty_frame():
    config = DetectorConfig(model_name="yolov8n.pt", device="cpu")
    detector = YOLODetector(config)

    empty_frame = np.empty((0, 0, 3), dtype=np.uint8)
    res = detector.detect(empty_frame)
    assert res.is_empty()
    assert res.count == 0


def test_detector_inference_on_synthetic_frame():
    config = DetectorConfig(model_name="yolov8n.pt", device="cpu")
    detector = YOLODetector(config)

    # Frame with arbitrary noise / blank background
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.detect(frame)

    assert isinstance(res, DetectionResult)
    assert isinstance(res.xyxy, np.ndarray)
    assert isinstance(res.confidence, np.ndarray)
    assert isinstance(res.class_id, np.ndarray)
    assert isinstance(res.class_name, list)
