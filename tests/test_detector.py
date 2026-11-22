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


# --------------------------------------------------------------------------- duplicate pose suppression

def _skel(cx, top=80.0, h=380.0, conf=0.9, upper_only=False):
    """COCO-17 skeleton in a standing pose; upper_only hides hips and below (as in a partial-body box)."""
    import numpy as np
    rel = [(0, .05), (-.02, .04), (.02, .04), (-.04, .05), (.04, .05), (-.09, .18), (.09, .18), (-.12, .33),
           (.12, .33), (-.13, .46), (.13, .46), (-.06, .52), (.06, .52), (-.06, .74), (.06, .74), (-.06, .95), (.06, .95)]
    kp = np.array([[cx + dx * h, top + dy * h, conf] for dx, dy in rel], np.float32)
    if upper_only:
        kp[11:, 2] = 0.05
    return kp


def _result(entries):
    import numpy as np
    from pharmaguard.detector import DetectionResult
    boxes = np.array([e[0] for e in entries], np.float32)
    return DetectionResult(xyxy=boxes, confidence=np.array([e[1] for e in entries], np.float32),
                           class_id=np.zeros(len(entries), np.int32), class_name=["person"] * len(entries),
                           keypoints=np.stack([e[2] for e in entries]))


def test_same_body_reported_twice_is_merged_keeping_fuller_skeleton():
    from pharmaguard.detector import suppress_duplicate_poses
    full = ([250, 80, 390, 460], 0.70, _skel(320))
    upper = ([255, 80, 385, 280], 0.80, _skel(321, upper_only=True))   # higher conf, but no hips
    out = suppress_duplicate_poses(_result([upper, full]))
    assert out.count == 1
    assert out.xyxy[0][3] == 460                    # full-body box kept (hips needed for the pocket check)


def test_two_real_people_overlapping_are_both_kept():
    from pharmaguard.detector import suppress_duplicate_poses
    a = ([250, 80, 390, 460], 0.9, _skel(320))
    b = ([300, 90, 440, 470], 0.6, _skel(370, top=90))   # one partly behind the other: boxes overlap ~60%
    out = suppress_duplicate_poses(_result([a, b]))
    assert out.count == 2


def test_dedupe_keeps_all_people_on_real_photo():
    import os, cv2, ultralytics
    from pharmaguard.config import DetectorConfig
    from pharmaguard.detector import YOLODetector
    img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
    on = YOLODetector(DetectorConfig(model_name="yolov8n-pose.pt", device="cpu")).detect(img)
    off = YOLODetector(DetectorConfig(model_name="yolov8n-pose.pt", device="cpu", dedupe_poses=False)).detect(img)
    strong = lambda r: int((r.confidence >= 0.5).sum())
    assert strong(on) == strong(off) >= 3
