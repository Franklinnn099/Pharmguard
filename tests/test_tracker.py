"""
Unit tests for PharmaGuard multi-object ByteTrack tracking.
"""

import numpy as np
import pytest
from pharmaguard.config import TrackerConfig
from pharmaguard.detector import DetectionResult
from pharmaguard.tracker import ByteTrackerWrapper, TrackingResult


def test_tracker_initialization():
    config = TrackerConfig()
    tracker = ByteTrackerWrapper(config)
    assert len(tracker.tracked_registry) == 0


def test_tracker_empty_update():
    config = TrackerConfig()
    tracker = ByteTrackerWrapper(config)

    empty_det = DetectionResult()
    result = tracker.update(empty_det)

    assert isinstance(result, TrackingResult)
    assert len(result.active_ids) == 0
    assert len(result.new_ids) == 0


def test_tracker_tracking_consistency():
    config = TrackerConfig(track_activation_threshold=0.2, lost_track_buffer=5)
    tracker = ByteTrackerWrapper(config)

    # Frame 1: Person detection at box [100, 100, 200, 300]
    det1 = DetectionResult(
        xyxy=np.array([[100.0, 100.0, 200.0, 300.0]], dtype=np.float32),
        confidence=np.array([0.85], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
        class_name=["person"]
    )
    res1 = tracker.update(det1)
    assert len(res1.active_ids) == 1
    tid = res1.active_ids[0]
    assert tid in tracker.tracked_registry
    assert tracker.tracked_registry[tid].frames_active == 1

    # Frame 2: Person moves slightly to [105, 102, 205, 302]
    det2 = DetectionResult(
        xyxy=np.array([[105.0, 102.0, 205.0, 302.0]], dtype=np.float32),
        confidence=np.array([0.88], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
        class_name=["person"]
    )
    res2 = tracker.update(det2)
    # The same ID should persist without identity switching
    assert len(res2.active_ids) == 1
    assert res2.active_ids[0] == tid
    assert tracker.tracked_registry[tid].frames_active == 2
    assert len(tracker.tracked_registry[tid].trajectory) == 2


def test_tracker_reset():
    config = TrackerConfig()
    tracker = ByteTrackerWrapper(config)
    det = DetectionResult(
        xyxy=np.array([[50.0, 50.0, 150.0, 250.0]], dtype=np.float32),
        confidence=np.array([0.90], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
        class_name=["person"]
    )
    tracker.update(det)
    assert len(tracker.tracked_registry) > 0

    tracker.reset()
    assert len(tracker.tracked_registry) == 0
