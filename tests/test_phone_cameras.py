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


def _load_server(monkeypatch, detector: str, access_key: str = ""):
    """Import server/main.py fresh under a unique module name (the repo root also has a main.py)."""
    monkeypatch.setenv("PG_DETECTOR", detector)
    if access_key:
        monkeypatch.setenv("PG_ACCESS_KEY", access_key)
    else:
        monkeypatch.delenv("PG_ACCESS_KEY", raising=False)
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


PROTECTED_GETS = ["/monitor", "/pair", "/api/cameras", "/api/cameras/c1/zones", "/zones/c1",
                  "/api/cameras/c1/snapshot.jpg", "/view/c1"]


def test_access_key_denies_every_route_without_it(monkeypatch):
    srv = _load_server(monkeypatch, "none", access_key="s3cret")
    client = TestClient(srv.app)
    for path in PROTECTED_GETS:
        assert client.get(path).status_code == 401, path
        assert client.get(path + "?key=wrong").status_code == 401, path
    for path in ("/api/cameras/c1/ack", "/api/cameras/c1/reset"):
        assert client.post(path, json={"operator": "x"}).status_code == 401, path
    assert client.put("/api/cameras/c1/zones", json={"zones": []}).status_code == 401
    assert client.delete("/api/cameras/c1/zones").status_code == 401
    # The WebSocket is closed with 4001 before any hello or frame.
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/camera?cam=c1") as ws:
            ws.receive_json()
    assert exc.value.code == 4001
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/camera?cam=c1&key=wrong") as ws:
            ws.receive_json()
    assert "c1" not in srv.cameras   # a rejected phone never creates camera state
    # The static PWA shell has no camera data and stays reachable so the QR link can load.
    assert client.get("/").status_code == 200 and client.get("/app.js").status_code == 200
    assert client.get("/manifest.webmanifest").status_code == 200
    # CORS headers survive a 401, so the dashboard sees a clean "unauthorised" instead of a CORS error.
    r = client.get("/api/cameras", headers={"Origin": "http://127.0.0.1:7860"})
    assert r.status_code == 401 and r.headers.get("access-control-allow-origin") == "http://127.0.0.1:7860"


def test_access_key_allows_query_header_and_cookie(monkeypatch):
    srv = _load_server(monkeypatch, "none", access_key="s3cret")
    client = TestClient(srv.app)
    assert client.get("/api/cameras", headers={"X-PG-Key": "s3cret"}).status_code == 200
    assert client.post("/api/cameras/c1/ack", json={}, headers={"X-PG-Key": "s3cret"}).status_code == 200
    # One visit with ?key= sets the cookie; later pages, <img> streams and fetches need nothing else.
    r = client.get("/?cam=c1&start=1&key=s3cret")
    assert r.status_code == 200 and "pg_key=s3cret" in r.headers.get("set-cookie", "").replace('"', "")
    assert "httponly" in r.headers["set-cookie"].lower()
    for path in PROTECTED_GETS[:-2]:   # snapshot 404s (no frame yet) and /view streams forever; checked below
        assert client.get(path).status_code == 200, path
    assert client.get("/api/cameras/c1/snapshot.jpg").status_code == 404   # past auth, just no frame yet
    with client.websocket_connect("/ws/camera?cam=c1&key=s3cret") as ws:
        assert ws.receive_json()["type"] == "hello"
    # Pairing stays one scan: the QR/pair link carries the key.
    assert srv.pair_url("c1").endswith("/?cam=c1&start=1&key=s3cret")
    assert "key=s3cret" in client.get("/pair?cam=c1").text


def test_no_access_key_keeps_laptop_mode_open(monkeypatch):
    srv = _load_server(monkeypatch, "none")
    client = TestClient(srv.app)
    for path in PROTECTED_GETS[:-2]:
        assert client.get(path).status_code == 200, path
    assert client.post("/api/cameras/c1/ack", json={}).status_code == 200
    assert "set-cookie" not in client.get("/?key=anything").headers
    assert "key=" not in srv.pair_url("c1")
    with client.websocket_connect("/ws/camera?cam=c1") as ws:
        assert ws.receive_json()["type"] == "hello"


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


def test_real_pipeline_per_camera_state_and_return_format(monkeypatch, tmp_path):
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

    # Zone editor: applied live to that camera only, and loaded by a camera's pipeline when it is created.
    monkeypatch.setattr(pg_bridge, "ZONES_DIR", tmp_path)
    box = [[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5]]
    bridge.set_zones("cam_a", [{"name": "Phone_Shelf", "polygon_normalized": box}])
    assert [z.name for z in pa.config.zones] == ["Phone_Shelf"]
    assert [z.name for z in pa.behavior_analyzer.zones] == ["Phone_Shelf"]
    assert [z.name for z in pb.config.zones] != ["Phone_Shelf"]
    bridge.process("cam_a", rgb)                                   # next frame runs with the new zone
    assert bridge.raw_frame("cam_a") is not None and bridge.raw_frame("cam_a").shape[2] == 3
    bridge.set_zones("cam_c", [{"name": "Saved_First", "polygon_normalized": box}])
    assert [z.name for z in bridge.pipeline_for("cam_c").config.zones] == ["Saved_First"]
    bridge.reset_zones("cam_a")
    assert [z.name for z in pa.config.zones] != ["Phone_Shelf"]
    bridge.close()


def test_phone_ws_reports_sounding_alarm_once_and_phone_can_acknowledge(monkeypatch, tmp_path):
    """Scripted shelf->bag sequence sent as phone frames over /ws/camera: the replies carry the alarm sounding
    (one incident -> one sound_seq) and the phone's Acknowledge route silences it."""
    from behavior_scenarios import Actor, FPS
    from test_live_streaming_integration import FixedBagDetector, ScriptedPoseDetector
    from pharmaguard.config import PharmaGuardConfig
    from pharmaguard.pipeline import PharmaGuardPipeline

    srv = _load_server(monkeypatch, "none")
    actor = Actor(1, 320, [(1.0, "rest", None), (1.5, "reach_R", None), (0.3, "rest", None),
                           (1.2, "bag_R", None), (2.0, "rest", None)], bag=True)
    frames = actor.frames()
    cfg = PharmaGuardConfig()
    cfg.objects.enabled = False
    cfg.alert.alert_log_file = str(tmp_path / "alerts.jsonl")
    p = PharmaGuardPipeline(cfg, detector=ScriptedPoseDetector(frames, cfg))
    p.object_detector = FixedBagDetector(actor)
    ticks = iter(range(10_000))
    real = p.process_frame
    p.process_frame = lambda f: real(f, timestamp=next(ticks) / FPS)   # scripted time, not wall time
    srv.bridge.mode, srv.bridge._pipelines["alarm_cam"] = "pharmaguard", p

    client = TestClient(srv.app)
    blank = _jpeg(np.zeros((480, 640, 3), np.uint8))
    replies = []
    with client.websocket_connect("/ws/camera?cam=alarm_cam") as ws:
        ws.receive_json()
        for _ in range(len(frames) + int(3 * FPS)):
            ws.send_bytes(blank)
            replies.append(ws.receive_json())
        assert {r["sound_seq"] for r in replies if r["sounding"]} == {1}   # sounded, exactly one incident
        assert replies[-1]["alarm"] == "ACTIVE"
        assert client.post("/api/cameras/alarm_cam/ack", json={"operator": "phone alarm_cam"}).json()["ok"]
        ws.send_bytes(blank)
        after = ws.receive_json()
    assert after["alarm"] == "ACKNOWLEDGED" and after["sounding"] is False
    assert p.alarm.snapshot().acknowledged_by == "phone alarm_cam"


def test_dashboard_has_phone_cameras_tab():
    from pharmaguard.ui import DASHBOARD_HEAD
    from pharmaguard.ui_phones import PHONE_CAMERAS_HEAD, phone_cameras_html
    assert PHONE_CAMERAS_HEAD in DASHBOARD_HEAD
    assert "/api/cameras" in PHONE_CAMERAS_HEAD and "/view/" in PHONE_CAMERAS_HEAD
    html = phone_cameras_html("http://192.168.1.5:8000/")
    assert 'data-server="http://192.168.1.5:8000"' in html
    assert "data-server" not in phone_cameras_html(None)


def test_dashboard_tab_passes_camera_server_key(monkeypatch):
    from pharmaguard.ui_phones import PHONE_CAMERAS_HEAD, phone_cameras_html
    monkeypatch.delenv("PG_ACCESS_KEY", raising=False)
    assert "data-key" not in phone_cameras_html(None)
    monkeypatch.setenv("PG_ACCESS_KEY", 'k"<1')
    assert 'data-key="k&quot;&lt;1"' in phone_cameras_html(None)   # escaped into the attribute
    assert PHONE_CAMERAS_HEAD.count("keyOf(root)") == 3          # status poll, MJPEG <img>, ack/reset


# --------------------------------------------------------------------------- per-camera zone editor
def test_zone_validation_rules(monkeypatch):
    monkeypatch.syspath_prepend(SERVER_DIR)
    sys.modules.pop("pg_bridge", None)
    import pg_bridge
    box = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.6], [0.1, 0.6]]
    ok, err = pg_bridge.validate_zones([{"name": "Shelf A", "polygon_normalized": box}])
    assert err == "" and ok[0]["name"] == "Shelf A" and ok[0]["polygon_normalized"] == box
    clamped, err = pg_bridge.validate_zones([{"name": "x", "polygon_normalized": [[-1, 0], [2, 0], [2, 3]]}])
    assert err == "" and clamped[0]["polygon_normalized"] == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    assert pg_bridge.validate_zones([])[1] == ""                        # no zones is allowed
    bad = [
        ("not a list", "must be a list"),
        ([{"name": "", "polygon_normalized": box}], "needs a name"),
        ([{"name": "a", "polygon_normalized": box[:2]}], "3 to"),
        ([{"name": "a", "polygon_normalized": [[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]]}], "too small"),
        ([{"name": "a", "polygon_normalized": box}, {"name": "a", "polygon_normalized": box}], "two zones"),
        ([{"name": "a", "polygon_normalized": [[0, "x"], [1, 0], [1, 1]]}], "invalid corner"),
        ([{"name": f"z{i}", "polygon_normalized": box} for i in range(9)], "at most"),
    ]
    for raw, msg in bad:
        assert msg in pg_bridge.validate_zones(raw)[1], raw


def test_zone_editor_endpoints_save_per_camera(monkeypatch, tmp_path):
    srv = _load_server(monkeypatch, "none")
    monkeypatch.setattr(sys.modules["pg_bridge"], "ZONES_DIR", tmp_path)
    client = TestClient(srv.app)

    assert client.get("/api/cameras/cam1/snapshot.jpg").status_code == 404      # nothing streamed yet
    with client.websocket_connect("/ws/camera?cam=cam1") as ws:
        ws.receive_json()
        ws.send_bytes(_jpeg(np.full((120, 160, 3), 50, np.uint8)))
        ws.receive_json()
    snap = client.get("/api/cameras/cam1/snapshot.jpg")
    assert snap.status_code == 200 and snap.headers["content-type"] == "image/jpeg"
    page = client.get("/zones/cam1")
    assert page.status_code == 200 and "/api/cameras/" in page.text and ">cam1<" in page.text

    assert client.get("/api/cameras/cam1/zones").json()["source"] == "default"
    box = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.7], [0.2, 0.7]]
    r = client.put("/api/cameras/cam1/zones", json={"zones": [{"name": "Front", "polygon_normalized": box}]})
    assert r.status_code == 200 and r.json()["source"] == "camera"
    assert (tmp_path / "cam1.json").exists()
    got = client.get("/api/cameras/cam1/zones").json()
    assert got["zones"][0]["name"] == "Front" and got["zones"][0]["polygon_normalized"] == box
    assert client.get("/api/cameras/cam2/zones").json()["source"] == "default"   # other cameras unaffected

    r = client.put("/api/cameras/cam1/zones", json={"zones": [{"name": "x", "polygon_normalized": box[:2]}]})
    assert r.status_code == 400 and "corners" in r.json()["error"]
    assert client.put("/api/cameras/cam1/zones", content=b"{oops").status_code == 400
    assert client.get("/api/cameras/cam1/zones").json()["zones"][0]["name"] == "Front"   # bad input changed nothing

    r = client.delete("/api/cameras/cam1/zones")
    assert r.status_code == 200 and r.json()["source"] == "default" and not (tmp_path / "cam1.json").exists()
    # path tricks are sanitized into a plain camera name inside the zones folder
    client.put("/api/cameras/..%2F..%2Fevil/zones", json={"zones": []})
    assert all(p.parent == tmp_path for p in tmp_path.rglob("*.json"))


def _result(boxes, confs, names):
    from pharmaguard.detector import DetectionResult
    return DetectionResult(xyxy=np.array(boxes, np.float32), confidence=np.array(confs, np.float32),
                           class_id=np.arange(len(boxes), dtype=np.int32), class_name=list(names))


def test_object_scan_postprocess_labels_and_filters():
    from pharmaguard.config import ItemDetectorConfig
    from pharmaguard.object_scan import postprocess
    cfg = ItemDetectorConfig()
    res = _result(
        [[10, 10, 50, 90], [100, 100, 120, 140], [0, 0, 200, 190], [60, 60, 90, 100], [130, 20, 190, 80],
         [5, 100, 40, 150]],
        [0.80, 0.30, 0.90, 0.95, 0.55, 0.50],
        ["smartphone", "tinsel", "martial arts gym", "person", "bookshelf", "office chair"])
    got = {o.raw_name: o for o in postprocess(res, 200, 200, cfg)}
    assert got["smartphone"].label == "mobile phone"               # synonym merged
    assert got["tinsel"].label == "object"                         # unsure -> plain 'object', raw name kept
    assert "martial arts gym" not in got                           # scene-sized box dropped
    assert "person" not in got                                     # people are the pose model's job
    assert got["bookshelf"].label == "shelf" and got["office chair"].label == "chair"
    assert got["smartphone"].box == [0.05, 0.05, 0.25, 0.45]       # normalized
    assert list(got) == ["smartphone", "bookshelf", "office chair", "tinsel"]   # most confident first


def test_object_display_modes_hide_clutter_by_default():
    from pharmaguard.config import ItemDetectorConfig
    from pharmaguard.object_scan import postprocess, visible
    cfg = ItemDetectorConfig()
    assert cfg.show == "key"                                         # uncluttered by default
    res = _result([[0, 0, 20, 20], [30, 30, 60, 60], [70, 70, 90, 95], [100, 100, 130, 140]],
                  [0.86, 0.30, 0.60, 0.70], ["tablet computer", "tinsel", "lampshade", "bookshelf"])
    objs = postprocess(res, 200, 200, cfg)
    by = {o.raw_name: o for o in objs}
    assert by["tablet computer"].label == "phone/tablet" and by["tablet computer"].key
    assert by["bookshelf"].key and not by["tinsel"].key and not by["lampshade"].key
    assert [o.raw_name for o in objs][:2] == ["tablet computer", "bookshelf"]     # key objects sorted first
    assert {o.label for o in visible(objs, "key")} == {"phone/tablet", "shelf"}
    assert len(visible(objs, "all")) == 4 and visible(objs, "off") == []
    # key objects survive the max_items cap even when many confident non-key boxes exist
    many = _result([[i, i, i + 15, i + 15] for i in range(0, 100, 2)] + [[150, 150, 190, 190]],
                   [0.9] * 50 + [0.5], ["lampshade"] * 50 + ["chair"])
    capped = postprocess(many, 200, 200, ItemDetectorConfig(max_items=5))
    assert len(capped) == 5 and capped[0].label == "chair"


def test_object_scan_runs_in_background_and_is_display_only(monkeypatch):
    """The scan never blocks frames, its boxes reach overlay + phone, and the behaviour engine never sees them."""
    import threading
    import time as _time
    from pharmaguard.config import PharmaGuardConfig
    from pharmaguard.pipeline import PharmaGuardPipeline

    gate = threading.Event()

    class SlowStub:
        calls = 0

        def detect(self, frame):
            SlowStub.calls += 1
            gate.wait(5)                                  # simulate a slow model until released
            h, w = frame.shape[:2]
            return _result([[0.1 * w, 0.1 * h, 0.3 * w, 0.5 * h]], [0.71], ["bottle"])

        def get_model_info(self):
            return {"model_name": "stub"}

    cfg = PharmaGuardConfig()
    cfg.recording.enabled = False
    assert cfg.items.enabled is False                     # off by default for the main dashboard
    p = PharmaGuardPipeline(config=cfg)
    p.enable_object_scan(SlowStub())
    seen_bags = []
    real_analyze = p.behavior_analyzer.analyze
    monkeypatch.setattr(p.behavior_analyzer, "analyze",
                        lambda *a, **k: (seen_bags.append(k.get("bag_boxes")), real_analyze(*a, **k))[1])
    frame = np.full((240, 320, 3), 90, np.uint8)
    t0 = _time.perf_counter()
    for _ in range(3):
        out = p.process_frame(frame)                      # model is blocked: frames must still flow
    assert _time.perf_counter() - t0 < 4.0 and out.metadata["overlay"]["items"] == []
    gate.set()
    deadline = _time.time() + 5
    while not p.object_scanner.latest() and _time.time() < deadline:
        _time.sleep(0.02)
    out = p.process_frame(frame)
    item = out.metadata["overlay"]["items"][0]
    assert item["label"] == "bottle" and item["conf"] == pytest.approx(0.71, abs=1e-3)
    assert item["box"] == [0.1, 0.1, 0.3, 0.5]
    assert item["key"] is True and out.metadata["overlay"]["items_show"] == "key"
    assert all(not b for b in seen_bags)                  # behaviour never received an item box
    assert out.metadata["stage_timings"]["object_scan_ms"] > 0

    monkeypatch.syspath_prepend(SERVER_DIR)
    sys.modules.pop("pg_bridge", None)
    import pg_bridge
    bridge = pg_bridge.PharmaGuardBridge("none")
    bridge.mode = "pharmaguard"
    bridge._pipelines["stubcam"] = p
    items = [d for d in bridge.process("stubcam", frame).detections if d.get("kind") == "item"]
    assert items and items[0]["label"] == "bottle" and items[0]["alert"] is False and items[0]["name"] == "bottle"
    p.close()
    assert p.object_scanner is None


@pytest.mark.skipif(not os.path.exists(os.path.join(REPO, "yoloe-11s-seg-pf.pt")), reason="YOLOE weights not present")
def test_yoloe_detector_finds_named_objects_on_real_image():
    pytest.importorskip("ultralytics")
    import ultralytics
    from pharmaguard.config import ItemDetectorConfig
    from pharmaguard.detector import YOLOEDetector, make_item_detector
    from pharmaguard.object_scan import postprocess
    cfg = ItemDetectorConfig(model_name=os.path.join(REPO, "yoloe-11s-seg-pf.pt"))
    det = make_item_detector(cfg, device="cpu")
    assert isinstance(det, YOLOEDetector) and det.get_model_info()["vocabulary_size"] > 1000
    img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
    objs = postprocess(det.detect(img), img.shape[1], img.shape[0], cfg)
    assert objs and all(o.raw_name not in ("person", "man", "woman") for o in objs)
    assert all(0.0 <= v <= 1.0 for o in objs for v in o.box)


def test_laptop_demo_defaults_scan_off_one_in_flight(monkeypatch):
    """Measured on the laptop CPU: the object scan halves frames/s and 2 in flight doubles latency."""
    monkeypatch.delenv("PG_ITEMS", raising=False)
    monkeypatch.delenv("PG_MAX_IN_FLIGHT", raising=False)
    srv = _load_server(monkeypatch, "none")
    assert srv.bridge._make_config("c").items.enabled is False
    with TestClient(srv.app).websocket_connect("/ws/camera?cam=c") as ws:
        assert ws.receive_json()["max_in_flight"] == 1
    monkeypatch.setenv("PG_ITEMS", "1")                     # opt in
    monkeypatch.setenv("PG_MAX_IN_FLIGHT", "2")
    srv = _load_server(monkeypatch, "none")
    assert srv.bridge._make_config("c").items.enabled is True
    with TestClient(srv.app).websocket_connect("/ws/camera?cam=c") as ws:
        assert ws.receive_json()["max_in_flight"] == 2


def test_api_flags_cameras_too_slow_for_reliable_detection(monkeypatch):
    import time as _t
    srv = _load_server(monkeypatch, "none")
    client = TestClient(srv.app)
    with client.websocket_connect("/ws/camera?cam=c") as ws:
        assert ws.receive_json()["slow_fps"] == srv.SLOW_FPS == 5.0
    now = _t.time()

    def cam(name, frames, times, connected=True):
        st = srv.camera(name)
        st.stats.connected, st.stats.frames, st.stats.last_frame_at = connected, frames, times[-1]
        st._fps_times = list(times)

    cam("fast", 100, [now - 1 + i * 0.1 for i in range(10)])          # ~10 fps, fresh
    cam("slow", 100, [now - 2, now - 1, now - 0.1])                    # ~1 fps
    cam("warming", 3, [now - 2, now - 1, now - 0.1])                   # too early to judge
    cam("stalled", 100, [now - 6 + i * 0.1 for i in range(10)])       # 10 fps but nothing for 5 s
    cam("gone", 100, [now - 2, now - 1], connected=False)             # offline: no fps warning
    flags = {c["camera_id"]: c["slow"] for c in client.get("/api/cameras").json()["cameras"]}
    assert flags == {"c": False, "fast": False, "slow": True, "warming": False, "stalled": True, "gone": False}


def test_monitor_and_dashboard_render_the_slow_warning():
    from pharmaguard.ui_phones import PHONE_CAMERAS_HEAD
    monitor = open(os.path.join(SERVER_DIR, "templates", "monitor.html"), encoding="utf-8").read()
    for page in (monitor, PHONE_CAMERAS_HEAD):
        assert "c.slow" in page and "too slow for reliable detection" in page
