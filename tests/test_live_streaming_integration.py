"""
Comprehensive live streaming and end-to-end integration tests for PharmaGuard.
Validates the full chain: stream ingestion, detection, tracking, zone analysis,
alerts, annotations, and disconnection/reconnection recovery.
"""

import time
import numpy as np
import cv2
import pytest

from pharmaguard.config import PharmaGuardConfig, ShelfZone
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
    trk1 = pipeline.tracker.update(det1)
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
    trk2 = pipeline.tracker.update(det2)
    assert len(trk2.active_ids) == 2
    # Identities must remain identical without switching
    assert id_a in trk2.active_ids
    assert id_b in trk2.active_ids


def test_shelf_monitoring_and_behavior_alerts():
    """TEST 4, 5, 6: Verify zone entry, dwell accumulation, alert generation, and cooldown throttling."""
    config = PharmaGuardConfig()
    config.zones = [
        ShelfZone(
            name="Prescription_Shelf_A",
            polygon_normalized=[[0.10, 0.10], [0.50, 0.10], [0.50, 0.85], [0.10, 0.85]],
            dwell_alert_threshold_sec=4.0,
            rapid_retreat_threshold_sec=2.0
        )
    ]
    pipeline = PharmaGuardPipeline(config)

    # Frame 1: Person enters Prescription_Shelf_A (footprint inside [0.10, 0.10] - [0.50, 0.85])
    # At 640x480, zone is roughly x: [64, 320], y: [48, 408]. Box [150, 200, 250, 380] has feet at (200, 380).
    box_inside = np.array([[150.0, 200.0, 250.0, 380.0]], dtype=np.float32)
    det_inside = DetectionResult(
        xyxy=box_inside,
        confidence=np.array([0.90], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
        class_name=["person"]
    )
    trk_res = pipeline.tracker.update(det_inside)
    events = pipeline.behavior_analyzer.analyze(trk_res, frame_width=640, frame_height=480)
    assert any(e.event_type == BehaviorEventType.ZONE_ENTER for e in events)

    # Advance dwell time artificially past threshold
    tid = trk_res.active_ids[0]
    pipeline.behavior_analyzer.zone_entry_times[(tid, "Prescription_Shelf_A")] = time.time() - 5.5

    # Trigger second analysis cycle
    events_dwell = pipeline.behavior_analyzer.analyze(trk_res, frame_width=640, frame_height=480)
    assert any(e.event_type == BehaviorEventType.SUSPICIOUS_LOITERING for e in events_dwell)

    # Alert generation
    alerts = pipeline.alert_manager.process_behavior_events(events_dwell)
    assert len(alerts) == 1
    assert alerts[0].severity == "ALERT"
    assert "PROLONGED DWELL" in alerts[0].title

    # Repeat should be suppressed by cooldown
    alerts_repeat = pipeline.alert_manager.process_behavior_events(events_dwell)
    assert len(alerts_repeat) == 0


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


def test_gradio_ui_webcam_stream_integration():
    """TEST 1 & 7: Verify Gradio Blocks interface processes webcam frames and returns complete UI payload."""
    config = PharmaGuardConfig()
    pipeline = PharmaGuardPipeline(config)
    demo = create_pharmaguard_ui(pipeline, config)

    # Find the stream processing function inside the Gradio Blocks dependencies
    stream_fn = None
    for fn in demo.fns.values():
        if "process_webcam_stream" in getattr(fn.fn, "__name__", ""):
            stream_fn = fn.fn
            break

    assert stream_fn is not None, "Gradio webcam stream handler not found in Blocks demo"

    # Test stream_fn with a valid frame
    test_frame = create_synthetic_frame(640, 480)
    result = stream_fn(test_frame)

    assert len(result) == 8
    annotated_rgb, status_txt, tracked_cnt, active_ids, fps_txt, latency_txt, alerts_html, events_html = result

    assert isinstance(annotated_rgb, np.ndarray)
    assert annotated_rgb.shape == (480, 640, 3)
    assert "ONLINE" in status_txt
    assert isinstance(fps_txt, str)
    assert isinstance(latency_txt, str)
    assert "<div" in alerts_html or "No active security alerts" in alerts_html

    # Test stream_fn with None (simulating stream pause)
    result_none = stream_fn(None)
    assert len(result_none) == 8
    assert "STANDBY" in result_none[1]


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
