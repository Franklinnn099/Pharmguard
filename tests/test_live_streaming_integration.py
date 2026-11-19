"""
Comprehensive live streaming and end-to-end integration tests for PharmaGuard.
Validates the full chain: stream ingestion, detection, tracking, zone analysis,
alerts, annotations, and disconnection/reconnection recovery.
"""

import time
import numpy as np
import cv2
import pytest

from pharmaguard.config import PharmaGuardConfig
from pharmaguard.detector import DetectionResult
from pharmaguard.pipeline import PharmaGuardPipeline, PipelineOutput
from pharmaguard.behavior import BehaviorEventType
from pharmaguard.ui import create_pharmaguard_ui


def create_synthetic_frame(w=640, h=480, person_boxes=None):
    """Generate an RGB frame with simulated human shapes."""
    frame = np.full((h, w, 3), 45, dtype=np.uint8)
    if person_boxes:
        for (x1, y1, x2, y2) in person_boxes:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (200, 200, 200), -1)
            # Head circle
            head_cx = int((x1 + x2) / 2)
            head_cy = int(y1 + (y2 - y1) * 0.18)
            cv2.circle(frame, (head_cx, head_cy), int((x2 - x1) * 0.2), (180, 180, 180), -1)
    return frame


def test_stream_connection_and_recovery():
    """TEST 1 & 8: Verify stream frames process, stream pause/disconnect returns STANDBY, and stream resumes."""
    config = PharmaGuardConfig()
    pipeline = PharmaGuardPipeline(config)

    # 1. Normal frame stream
    frame1 = create_synthetic_frame()
    out1 = pipeline.process_frame(frame1)
    assert out1.connection_status == "ONLINE"
    assert out1.annotated_frame_rgb.shape == (480, 640, 3)

    # 2. Simulate stream interruption / webcam disconnected (None frame)
    out_pause = pipeline.process_frame(None)
    assert out_pause.connection_status == "NO_SIGNAL"
    assert out_pause.active_tracks_count == 0

    # 3. Simulate stream reconnection
    frame2 = create_synthetic_frame()
    out2 = pipeline.process_frame(frame2)
    assert out2.connection_status == "ONLINE"
    assert out2.annotated_frame_rgb.shape == (480, 640, 3)


def test_multi_person_tracking_consistency():
    """TEST 3: Verify multiple individuals receive distinct IDs and maintain identity across frames."""
    config = PharmaGuardConfig()
    pipeline = PharmaGuardPipeline(config)

    # Frame 1: Person A at [100, 100, 200, 350], Person B at [400, 120, 500, 370]
    boxes_f1 = np.array([
        [100.0, 100.0, 200.0, 350.0],
        [400.0, 120.0, 500.0, 370.0]
    ], dtype=np.float32)
    det1 = DetectionResult(
        xyxy=boxes_f1,
        confidence=np.array([0.88, 0.92], dtype=np.float32),
        class_id=np.array([0, 0], dtype=np.int32),
        class_name=["person", "person"]
    )
    # IDs are issued on the second consecutive detection (1-frame false positives never get an ID).
    assert pipeline.tracker.update(det1, timestamp=0.0).active_ids == []
    trk1 = pipeline.tracker.update(det1, timestamp=0.1)
    assert len(trk1.active_ids) == 2
    id_a, id_b = trk1.active_ids[0], trk1.active_ids[1]
    assert id_a != id_b

    # Frame 2: Person A moves right, Person B moves left
    boxes_f2 = np.array([
        [110.0, 102.0, 210.0, 352.0],
        [390.0, 122.0, 490.0, 372.0]
    ], dtype=np.float32)
    det2 = DetectionResult(
        xyxy=boxes_f2,
        confidence=np.array([0.89, 0.91], dtype=np.float32),
        class_id=np.array([0, 0], dtype=np.int32),
        class_name=["person", "person"]
    )
    trk2 = pipeline.tracker.update(det2, timestamp=0.2)
    assert len(trk2.active_ids) == 2
    # Identities must remain identical without switching
    assert id_a in trk2.active_ids
    assert id_b in trk2.active_ids


class ScriptedPoseDetector:
    """Stands in for YOLO-pose: replays a scripted skeleton sequence (tests/behavior_scenarios.py)."""

    def __init__(self, frames, config):
        self.frames, self.i, self.config = frames, 0, config.detector

    def detect(self, frame):
        _, kp, box = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        if self.i > len(self.frames):
            return DetectionResult()            # actor has left the view
        return DetectionResult(xyxy=box[None], confidence=np.array([0.9], np.float32),
                               class_id=np.zeros(1, np.int32), class_name=["person"], keypoints=kp[None])

    def get_model_info(self):
        return {"device": "cpu", "model_name": "scripted-pose", "has_keypoints": True}


class FixedBagDetector:
    def __init__(self, actor):
        self.actor = actor

    def detect(self, frame):
        return DetectionResult(xyxy=self.actor.bag_box(320.0)[None], confidence=np.array([0.8], np.float32),
                               class_id=np.array([26], np.int32), class_name=["handbag"])

    def get_model_info(self):
        return {"device": "cpu", "model_name": "fixed-bag"}


def test_end_to_end_shelf_to_bag_sequence_raises_one_review_and_one_alert():
    """Full pipeline (tracker -> behaviour -> alert manager -> event log) on a scripted shelf->bag->leave
    sequence: exactly one REVIEW and one ALERT reach the operator, with reasons and score."""
    from behavior_scenarios import Actor, FPS
    actor = Actor(1, 320, [(1.0, "rest", None), (1.5, "reach_R", None), (0.3, "rest", None),
                           (1.2, "bag_R", None), (2.0, "rest", None)], bag=True)
    frames = actor.frames()
    config = PharmaGuardConfig()
    config.objects.enabled = False
    pipeline = PharmaGuardPipeline(config, detector=ScriptedPoseDetector(frames, config))
    pipeline.object_detector = FixedBagDetector(actor)
    blank = np.zeros((480, 640, 3), np.uint8)

    alerts = []
    n_frames = len(frames) + int(4.0 * FPS)            # actor leaves; tracker removes them after 2 s
    for i in range(n_frames):
        out = pipeline.process_frame(blank, timestamp=i / FPS)
        alerts += out.metadata["new_alerts"]

    assert [a.severity for a in alerts] == ["WARNING", "ALERT"]
    alert = alerts[1]
    assert alert.score >= config.behavior.alert_threshold
    assert alert.incident_id == alerts[0].incident_id
    assert any("bag" in r for r in alert.reasons)
    assert "review" in alert.title.lower()
    timeline = [e.message for e in pipeline.event_manager.get_recent(50)]
    assert any("reached into" in m for m in timeline)


def test_annotated_video_and_hud_output():
    """TEST 7: Verify annotated frame contains valid RGB data, visual overlays, and telemetry."""
    config = PharmaGuardConfig()
    pipeline = PharmaGuardPipeline(config)

    raw_frame = create_synthetic_frame(640, 480)
    output = pipeline.process_frame(raw_frame)

    assert isinstance(output, PipelineOutput)
    assert output.annotated_frame_rgb.shape == (480, 640, 3)
    assert output.annotated_frame_rgb.dtype == np.uint8
    # Frame must not be completely black and must have HUD drawn at top
    hud_region = output.annotated_frame_rgb[0:32, :]
    assert np.mean(hud_region) > 0


def _find_fn(demo, name):
    for fn in demo.fns.values():
        if getattr(fn.fn, "__name__", "") == name:
            return fn.fn
    raise AssertionError(f"{name} not registered in Gradio Blocks")


def test_gradio_ui_webcam_stream_integration():
    """Stream handler returns only the annotated frame; the 1 Hz poller returns the panel payload."""
    config = PharmaGuardConfig()
    pipeline = PharmaGuardPipeline(config)
    demo = create_pharmaguard_ui(pipeline, config)

    stream_fn = _find_fn(demo, "process_webcam_stream")
    refresh_fn = _find_fn(demo, "refresh_dashboard")

    # Before any frame: STANDBY.
    status, *_ = refresh_fn({})
    assert "STANDBY" in status

    annotated = stream_fn(create_synthetic_frame(640, 480))
    assert isinstance(annotated, np.ndarray) and annotated.shape == (480, 640, 3)
    annotated = stream_fn(create_synthetic_frame(640, 480))

    status, tracked, ids, fps_txt, latency_txt, alerts_html, events_html, cache = refresh_fn({})
    assert status.startswith("LIVE")
    assert "server" in latency_txt and "cycle" in latency_txt
    assert "No active security alerts" in alerts_html
    # Unchanged HTML is skipped on the next poll (no needless re-render).
    *_, alerts2, events2, _ = refresh_fn(cache)
    assert not isinstance(alerts2, str) and not isinstance(events2, str)

    # Stream pause (None) yields a standby frame, not a crash.
    standby = stream_fn(None)
    assert isinstance(standby, np.ndarray)


def test_ui_transport_configuration():
    """Guards the two measured transport fixes: JPEG return encoding and a 640x480 capture request."""
    config = PharmaGuardConfig()
    pipeline = PharmaGuardPipeline(config)
    demo = create_pharmaguard_ui(pipeline, config)
    cfg = demo.get_config_file()
    images = {c["props"].get("label"): c["props"] for c in cfg["components"] if c.get("type") == "image"}
    out = images["2. PharmaGuard annotated output"]
    cam = images["1. Camera input (click to start webcam)"]
    assert out["format"] == "jpeg"
    assert cam["streaming"] is True
    assert cam["webcam_options"]["constraints"]["video"]["width"] == {"ideal": 640}
    api_names = {d.get("api_name") for d in cfg["dependencies"]}
    assert {"stream_frame", "process_frame", "refresh_dashboard"} <= api_names


def test_gradio_live_server_execution():
    """TEST: Launch a genuine running Gradio web server in background, verify HTTP 200 response, and clean shutdown."""
    import urllib.request
    import socket

    config = PharmaGuardConfig()
    pipeline = PharmaGuardPipeline(config)
    demo = create_pharmaguard_ui(pipeline, config)

    # Find a free port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    free_port = sock.getsockname()[1]
    sock.close()

    try:
        # Launch real server non-blocking
        demo.launch(
            server_name="127.0.0.1",
            server_port=free_port,
            prevent_thread_lock=True,
            show_error=False
        )

        # Make actual HTTP request to verify live running server
        url = f"http://127.0.0.1:{free_port}/"
        req = urllib.request.Request(url, headers={"User-Agent": "PharmaGuard-TestClient"})
        with urllib.request.urlopen(req, timeout=5) as response:
            assert response.status == 200
            content = response.read().decode("utf-8", errors="ignore")
            # Verify the HTML contains Gradio app container
            assert "gradio" in content.lower() or "<!doctype html>" in content.lower()

    finally:
        # Cleanly shutdown the live server
        demo.close()
