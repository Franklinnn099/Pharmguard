"""
Tracking tests for PharmaGuard's ByteTrack implementation.

Scenario tests use synthetic detections with KNOWN identities (tests/scenario_utils.py), so ID switches,
swaps and losses are measured rather than eyeballed. Real-detector integration is in
test_tracker_with_real_detector_on_panning_photo.
"""

import numpy as np
import pytest

import scenario_utils as S
from pharmaguard.config import TrackerConfig, PharmaGuardConfig
from pharmaguard.detector import DetectionResult
from pharmaguard.tracker import ByteTrackerWrapper, KalmanBoxFilter, TrackingResult, TrackState, xyxy_to_xyah, xyah_to_xyxy


def run(frames, config=None):
    trk = ByteTrackerWrapper(config or TrackerConfig())
    return S.evaluate(lambda d, t: trk.update(d, timestamp=t), frames), trk


# --------------------------------------------------------------------------- basics

def test_tracker_initialization():
    tracker = ByteTrackerWrapper(TrackerConfig())
    assert len(tracker.tracked_registry) == 0


def test_tracker_empty_update():
    result = ByteTrackerWrapper(TrackerConfig()).update(DetectionResult(), timestamp=0.0)
    assert isinstance(result, TrackingResult)
    assert result.active_ids == [] and result.new_ids == []


def test_tracker_reset():
    tracker = ByteTrackerWrapper(TrackerConfig())
    for i in range(3):
        tracker.update(S.det([("A", (50, 50, 150, 250), 0.9)]), timestamp=i * 0.1)
    assert tracker.tracked_registry
    tracker.reset()
    assert not tracker.tracked_registry
    assert tracker.update(S.det([("A", (50, 50, 150, 250), 0.9)]), timestamp=0.0).active_ids == []


def test_box_conversion_roundtrip():
    box = np.array([10, 20, 110, 320], np.float32)
    assert np.allclose(xyah_to_xyxy(xyxy_to_xyah(box)), box, atol=1e-3)


def test_kalman_learns_velocity():
    kf = KalmanBoxFilter()
    mean, cov = kf.initiate(xyxy_to_xyah(np.array([0, 0, 100, 300], np.float32)))
    for i in range(1, 15):
        mean, cov = kf.predict(mean, cov, 1.0)
        mean, cov = kf.update(mean, cov, xyxy_to_xyah(np.array([20 * i, 0, 20 * i + 100, 300], np.float32)))
    assert mean[4] == pytest.approx(20.0, abs=2.0)   # px per nominal frame
    assert abs(mean[5]) < 2.0


# --------------------------------------------------------------------------- confirmation / false positives

def test_id_assigned_only_after_confirmation():
    tracker = ByteTrackerWrapper(TrackerConfig())
    r1 = tracker.update(S.det([("A", (100, 100, 200, 400), 0.9)]), timestamp=0.0)
    assert r1.active_ids == [] and r1.tracked_persons == {}
    r2 = tracker.update(S.det([("A", (104, 100, 204, 400), 0.9)]), timestamp=0.1)
    assert r2.new_ids == [1] and r2.active_ids == [1]


def test_single_frame_false_positive_never_gets_an_id():
    res, _ = run(S.single_frame_false_positive())
    assert res.all_ids == set()


def test_low_confidence_detection_cannot_start_a_track():
    tracker = ByteTrackerWrapper(TrackerConfig())
    for i in range(10):
        r = tracker.update(S.det([("A", (100, 100, 200, 400), 0.3)]), timestamp=i * 0.1)
    assert r.tracked_persons == {}


# --------------------------------------------------------------------------- motion

@pytest.mark.parametrize("fps", [3, 5, 10, 25])
@pytest.mark.parametrize("hps", [0.8, 1.2, 1.8])
def test_walker_keeps_one_id_across_tunnel_frame_rates(hps, fps):
    """Walking (0.8), brisk (1.2) and hurried (1.8 body-heights/s) at 3-25 FPS.
    At 3 FPS a walker shifts ~0.27-0.6 body heights per frame; IoU alone fails there."""
    res, _ = run(S.walk_body_heights(hps, fps))
    assert len(res.ids_per_person["A"]) == 1, res.ids_per_person
    assert res.frames_with_id["A"] >= res.frames_visible["A"] - 1


@pytest.mark.parametrize("fps", [3, 5, 10])
def test_side_by_side_walkers_do_not_swap_at_low_fps(fps):
    res, _ = run(S.parallel_walkers(fps))
    assert len(res.ids_per_person["A"]) == 1 and len(res.ids_per_person["B"]) == 1, res.ids_per_person
    assert not res.shared_ids()


def test_motion_fallback_disabled_reproduces_pure_bytetrack_limit():
    """Documents why the fallback exists: pure IoU gating loses a walker at 3 FPS."""
    res, _ = run(S.walk_body_heights(1.2, 3), TrackerConfig(max_center_speed=0.0))
    assert len(res.ids_per_person["A"]) != 1


def test_irregular_frame_spacing_keeps_one_id():
    res, _ = run(S.irregular_rate())
    assert len(res.ids_per_person["A"]) == 1


def test_stationary_jitter_keeps_one_id():
    res, _ = run(S.jitter())
    assert len(res.ids_per_person["A"]) == 1
    assert len(res.all_ids) == 1


def test_five_people_get_five_distinct_stable_ids():
    res, _ = run(S.crowd())
    assert len(res.all_ids) == 5
    assert res.id_switches() == 0
    assert not res.shared_ids()


# --------------------------------------------------------------------------- crossing / occlusion

@pytest.mark.parametrize("occlusion", ["low_conf", "missing"])
def test_people_crossing_paths_keep_their_ids(occlusion):
    res, _ = run(S.crossing(occlusion=occlusion))
    assert len(res.ids_per_person["A"]) == 1, res.ids_per_person
    assert len(res.ids_per_person["B"]) == 1, res.ids_per_person
    assert not res.shared_ids(), "identity swap between A and B"


def test_low_confidence_partial_occlusion_stays_active():
    """ByteTrack stage 2: confidence 0.2 must keep the existing ID visible, not drop it."""
    res, _ = run(S.partial_occlusion_low_conf())
    assert len(res.ids_per_person["A"]) == 1
    assert res.frames_with_id["A"] >= res.frames_visible["A"] - 1


@pytest.mark.parametrize("moving", [True, False])
def test_short_full_occlusion_recovers_same_id(moving):
    res, _ = run(S.occlusion_gap(1.0, moving=moving))
    assert len(res.ids_per_person["A"]) == 1


def test_long_occlusion_expires_by_time_not_frames():
    """Removal after lost_track_seconds (2 s) of REAL time, independent of frame rate."""
    for fps in (5.0, 30.0):
        res, _ = run(S.occlusion_gap(3.0, fps=fps, moving=False))
        assert len(res.ids_per_person["A"]) == 2, fps
        assert len(res.removed_ids) == 1


def test_occluded_track_is_exposed_as_lost_with_predicted_box():
    tracker = ByteTrackerWrapper(TrackerConfig())
    for i in range(5):
        tracker.update(S.det([("A", (100 + 10 * i, 100, 220 + 10 * i, 400), 0.9)]), timestamp=i * 0.1)
    r = tracker.update(DetectionResult(), timestamp=0.5)
    person = r.tracked_persons[1]
    assert person.state == TrackState.LOST and person.frames_lost == 1
    assert r.occluded_ids == [1] and r.active_ids == []
    assert person.current_box[0] > 140  # predicted forward, not frozen at last detection


# --------------------------------------------------------------------------- real detector integration

def test_tracker_with_real_detector_on_panning_photo():
    """Real YOLO detections on a photo of people panned 12 px/frame: one stable ID per person."""
    import os, cv2, ultralytics
    from pharmaguard.pipeline import PharmaGuardPipeline

    img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
    base = cv2.cvtColor(cv2.resize(img, (640, 853)), cv2.COLOR_BGR2RGB)
    canvas = np.zeros((853, 900, 3), np.uint8)
    pipeline = PharmaGuardPipeline(PharmaGuardConfig())
    pipeline.config.alert.alert_log_file = None
    counts = []
    for i in range(20):
        frame = canvas.copy()
        x = 12 * i
        frame[:, x:x + 640] = base
        out = pipeline.process_frame(frame, timestamp=i * 0.1)
        counts.append(out.active_tracks_count)
    total_ids = pipeline.tracker._next_id - 1
    steady = max(counts[2:])
    assert steady >= 3                       # people in the photo are tracked
    assert total_ids <= steady + 1, (total_ids, counts)  # at most one spurious/extra ID over 20 frames
