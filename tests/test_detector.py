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


def test_pose_model_returns_aligned_wrist_keypoints_on_real_people():
    """Default yolov8n-pose: keypoints aligned with person boxes; confident wrists on a real photo."""
    import os, cv2, ultralytics
    from pharmaguard.detector import KP_L_WRIST, KP_R_WRIST
    img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
    detector = YOLODetector(DetectorConfig(device="cpu"))
    assert detector.get_model_info()["has_keypoints"] is True
    res = detector.detect(img)
    assert res.keypoints is not None and res.keypoints.shape == (res.count, 17, 3)
    confident = res.confidence >= 0.45
    assert confident.sum() >= 3
    wrists = res.keypoints[confident][:, [KP_L_WRIST, KP_R_WRIST], 2]
    assert (wrists.max(axis=1) >= 0.5).sum() >= 2          # most confident people have a visible wrist
    # Each wrist lies inside (or right at the edge of) its own person box.
    for box, kp in zip(res.xyxy[confident], res.keypoints[confident]):
        for x, y, c in kp[[KP_L_WRIST, KP_R_WRIST]]:
            if c >= 0.5:
                assert box[0] - 20 <= x <= box[2] + 20 and box[1] - 20 <= y <= box[3] + 20
    sub = res.subset(confident)
    assert sub.count == confident.sum() and sub.keypoints.shape[0] == sub.count
