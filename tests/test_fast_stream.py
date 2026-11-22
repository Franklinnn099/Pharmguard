"""
Fast stream transport (pharmaguard/fast_stream.py): ordering, newest-wins scheduling, error handling,
login enforcement, and a genuine HTTP round trip through a live server with the real detector.
"""

import os
import socket
import threading
import time

import cv2
import httpx
import numpy as np
import pytest

from pharmaguard.config import PharmaGuardConfig
from pharmaguard.fast_stream import FAST_STREAM_HEAD, FRAME_PATH, FastFrameService, install_fast_stream_route
from pharmaguard.pipeline import PharmaGuardPipeline


def _jpeg(value: int = 0) -> bytes:
    img = np.full((48, 64, 3), value, dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _identity_service(order: list):
    def process(rgb: np.ndarray) -> np.ndarray:
        order.append(int(rgb[0, 0, 0]))
        return rgb
    return FastFrameService(process)


def test_processes_frames_and_returns_jpeg():
    order: list = []
    svc = _identity_service(order)
    res = svc.handle(_jpeg(10), seq=0)
    assert res.status == 200
    decoded = cv2.imdecode(np.frombuffer(res.body, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (48, 64, 3)
    assert svc.processed == 1


def test_out_of_order_frame_is_skipped_never_processed_backwards():
    order: list = []
    svc = _identity_service(order)
    assert svc.handle(_jpeg(0), seq=0).status == 200
    assert svc.handle(_jpeg(50), seq=5).status == 200
    assert svc.handle(_jpeg(40), seq=4).status == 204      # arrived late: tracker must not go back in time
    assert svc.skipped == 1 and len(order) == 2


def test_concurrent_frames_are_processed_in_increasing_order():
    """Frame 1 is being processed while 2 and 3 queue behind it: whatever the lock order, the tracker
    only ever sees increasing frame numbers, and the newest frame is always processed."""
    started, release = threading.Event(), threading.Event()
    processed: list = []

    def slow(rgb):
        value = int(rgb[0, 0, 0])
        if value == 10:
            started.set()
            release.wait(5)
        processed.append(value)
        return rgb

    svc = FastFrameService(slow)
    results: dict = {}

    def send(seq, value):
        results[seq] = svc.handle(_jpeg(value), seq=seq).status

    svc.handle(_jpeg(0), seq=0)
    t1 = threading.Thread(target=send, args=(1, 10)); t1.start()
    assert started.wait(5)
    t2 = threading.Thread(target=send, args=(2, 20)); t2.start()
    t3 = threading.Thread(target=send, args=(3, 30)); t3.start()
    time.sleep(0.1)
    release.set()
    for t in (t1, t2, t3):
        t.join(5)
    values = [v for v in processed if v]
    assert values == sorted(values) and values[0] == 10 and values[-1] == 30
    assert results[1] == 200 and results[3] == 200 and results[2] in (200, 204)


def test_sequence_restart_after_page_reload():
    order: list = []
    svc = _identity_service(order)
    for seq in range(5):
        assert svc.handle(_jpeg(seq), seq=seq, session="a").status == 200
    assert svc.handle(_jpeg(0), seq=0, session="a").status == 204      # same session: stale
    assert svc.handle(_jpeg(0), seq=0, session="b").status == 200      # new camera session restarts at 0


def test_invalid_body_is_rejected():
    svc = _identity_service([])
    assert svc.handle(b"not a jpeg", seq=0).status == 400
    assert svc.handle(b"", seq=1).status == 400


def test_client_round_trip_is_recorded_in_telemetry():
    pipeline = PharmaGuardPipeline(PharmaGuardConfig())
    svc = FastFrameService(lambda rgb: rgb, on_client_rtt=pipeline.telemetry.record_client_rtt)
    for rtt in (100.0, 120.0, 140.0):
        svc.handle(_jpeg(), seq=0, client_rtt_ms=rtt)
    snap = pipeline.telemetry_snapshot()
    assert snap.client_rtt_p50_ms == pytest.approx(120.0)


def test_dashboard_head_contains_fast_client():
    from pharmaguard.ui import DASHBOARD_HEAD
    assert FAST_STREAM_HEAD in DASHBOARD_HEAD
    assert "pg/frame" in FAST_STREAM_HEAD


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_live_server_fast_stream_requires_login_and_returns_annotated_frame():
    """Real server + login + real YOLO: unauthenticated POST is refused, logged-in POST is processed."""
    import ultralytics
    from pharmaguard.ui import create_pharmaguard_ui

    bus = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
    frame = cv2.imencode(".jpg", cv2.resize(bus, (640, 480)))[1].tobytes()
    pipeline = PharmaGuardPipeline(PharmaGuardConfig())
    demo = create_pharmaguard_ui(pipeline)
    demo.queue()
    port = _free_port()
    try:
        demo.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True,
                    auth=("op", "pw"))
        svc = FastFrameService(lambda rgb: pipeline.process_frame(rgb).annotated_frame_rgb,
                               on_client_rtt=pipeline.telemetry.record_client_rtt)
        install_fast_stream_route(demo.server_app, svc, require_login=True)
        base = f"http://127.0.0.1:{port}"
        with httpx.Client(timeout=60) as anon:
            assert anon.post(base + FRAME_PATH, content=frame).status_code == 401
        with httpx.Client(timeout=60) as c:
            assert c.post(base + "/login", data={"username": "op", "password": "pw"}).status_code < 400
            for seq in range(3):
                r = c.post(base + FRAME_PATH, content=frame,
                           headers={"X-PG-Seq": str(seq), "X-PG-RTT-Ms": "150"})
                assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
            img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            assert img is not None and img.shape[1] == 640
            assert c.get(base + "/").status_code == 200      # Gradio's own routes still served
        assert pipeline.frame_counter == 3
        assert pipeline.last_output.active_tracks_count >= 2
        assert pipeline.telemetry_snapshot().client_rtt_p50_ms == pytest.approx(150.0)
    finally:
        demo.close()
