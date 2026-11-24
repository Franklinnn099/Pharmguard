"""Phone-camera server (pharmguard-camera/server): WebSocket protocol, routes, real-pipeline bridge, dashboard tab."""

import importlib.util
import os
import sys

import cv2
import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_DIR = os.path.join(REPO, "pharmguard-camera", "server")


def _load_server(monkeypatch, detector: str):
    """Import server/main.py fresh under a unique module name (the repo root also has a main.py)."""
    monkeypatch.setenv("PG_DETECTOR", detector)
    monkeypatch.setenv("PG_NO_RECORDING", "1")
    monkeypatch.syspath_prepend(SERVER_DIR)
    sys.modules.pop("pg_bridge", None)
    spec = importlib.util.spec_from_file_location("pg_camera_server", os.path.join(SERVER_DIR, "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _jpeg(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img)[1].tobytes()


def _bus() -> np.ndarray:
    import ultralytics
    img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
    assert img is not None
    return img


def test_ws_protocol_one_ordered_reply_per_frame(monkeypatch):
    srv = _load_server(monkeypatch, "none")
    client = TestClient(srv.app)
    frame = _jpeg(np.full((240, 320, 3), 100, np.uint8))
    with client.websocket_connect("/ws/camera?cam=unit_cam") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["camera_id"] == "unit_cam" and hello["detector"] == "none"
        replies = []
        for _ in range(5):                       # one in flight: every frame is processed
            ws.send_bytes(frame)
            replies.append(ws.receive_json())
        ws.send_bytes(b"not a jpeg")
        bad = ws.receive_json()
        ws.send_text('{"type":"ping"}')
        assert ws.receive_json()["type"] == "pong"
    assert [r["frame"] for r in replies] == [1, 2, 3, 4, 5]
    assert all(r["type"] == "detections" and r["detections"] == [] for r in replies)
    assert replies[0]["w"] == 320 and replies[0]["h"] == 240
    assert bad == {"type": "skipped", "frame": 6, "reason": "not a JPEG"}

    cams = client.get("/api/cameras").json()["cameras"]
    assert cams[0]["camera_id"] == "unit_cam" and cams[0]["frames"] == 5 and cams[0]["connected"] is False


def test_camera_names_are_sanitized(monkeypatch):
    srv = _load_server(monkeypatch, "none")
    assert srv.sanitize_cam("../../etc<script>") == "etcscript"
    assert srv.sanitize_cam("") == "phone"
    assert srv.pair_url("shelf-2").endswith("/?cam=shelf-2&start=1")


def test_http_routes_respond(monkeypatch):
    srv = _load_server(monkeypatch, "none")
    client = TestClient(srv.app)
    r = client.get("/")
    assert r.status_code == 200 and "/app.js" in r.text
    assert client.get("/app.js").status_code == 200
    assert "/ws/camera" in client.get("/app.js").text
    r = client.get("/pair?cam=front")
    assert r.status_code == 200 and "/?cam=front&start=1" in r.text
    assert client.get("/monitor").status_code == 200
    assert client.get("/api/cameras").json()["detector"] == "none"
    # CORS lets the dashboard (port 7860) read the status
    r = client.get("/api/cameras", headers={"Origin": "http://127.0.0.1:7860"})
    assert r.headers.get("access-control-allow-origin") == "http://127.0.0.1:7860"
    if importlib.util.find_spec("qrcode"):
        r = client.get("/pair/qr.svg?cam=front")
        assert r.status_code == 200 and r.headers["content-type"].startswith("image/svg")


def test_view_stream_serves_latest_frame_as_mjpeg(monkeypatch):
    srv = _load_server(monkeypatch, "none")
    st = srv.camera("mjpeg_cam")
    st.jpeg, st.version = _jpeg(np.zeros((10, 10, 3), np.uint8)), 1

    async def first_part():
        resp = await srv.view("mjpeg_cam")
        assert resp.media_type.startswith("multipart/x-mixed-replace")
        gen = resp.body_iterator
        part = await gen.__anext__()
        await gen.aclose()
        return part

    import asyncio
    part = asyncio.run(first_part())
    assert part.startswith(b"--frame\r\nContent-Type: image/jpeg") and st.jpeg in part


def test_real_pipeline_per_camera_state_and_return_format(monkeypatch):
    """Phone frames go through the real PharmaGuard pipeline; each camera has its own tracker, one shared model."""
    pytest.importorskip("ultralytics")
    monkeypatch.setenv("PG_NO_RECORDING", "1")
    monkeypatch.syspath_prepend(SERVER_DIR)
    sys.modules.pop("pg_bridge", None)
    import pg_bridge
    bridge = pg_bridge.PharmaGuardBridge("pharmaguard")
    rgb = cv2.cvtColor(_bus(), cv2.COLOR_BGR2RGB)
    for _ in range(4):                          # tracks need a few hits before they are confirmed
        res_a = bridge.process("cam_a", rgb)
    res_b = bridge.process("cam_b", rgb)

    people = [d for d in res_a.detections if d["label"].startswith("person")]
    assert len(people) >= 2, res_a.detections
    for d in res_a.detections:
        assert {"x1", "y1", "x2", "y2", "label", "conf", "alert"} <= set(d)
        assert 0.0 <= d["x1"] < d["x2"] <= 1.0 and 0.0 <= d["y1"] < d["y2"] <= 1.0
        assert isinstance(d["alert"], bool) and d["alert"] is False   # standing at a bus stop: nothing to review
    assert all(0.0 < d["conf"] <= 1.0 for d in people)
    assert res_a.annotated_rgb is not None and res_a.annotated_rgb.shape[1] == res_a.width

    pa, pb = bridge.pipeline_for("cam_a"), bridge.pipeline_for("cam_b")
    assert pa is not pb and pa.tracker is not pb.tracker and pa.behavior_analyzer is not pb.behavior_analyzer
    assert pa.detector is pb.detector                       # one model in memory
    assert pa.frame_counter == 4 and pb.frame_counter == 1  # independent state
    assert pa.config.recording.camera_id == "cam_a"
    assert bridge.alarm_snapshot("cam_a")["state"] == "IDLE"
    assert bridge.alarm_snapshot("unknown") is None
    bridge.close()


def test_dashboard_has_phone_cameras_tab():
    from pharmaguard.ui import DASHBOARD_HEAD
    from pharmaguard.ui_phones import PHONE_CAMERAS_HEAD, phone_cameras_html
    assert PHONE_CAMERAS_HEAD in DASHBOARD_HEAD
    assert "/api/cameras" in PHONE_CAMERAS_HEAD and "/view/" in PHONE_CAMERAS_HEAD
    html = phone_cameras_html("http://192.168.1.5:8000/")
    assert 'data-server="http://192.168.1.5:8000"' in html
    assert "data-server" not in phone_cameras_html(None)
