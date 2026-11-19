"""
Dashboard tests (Priority 6): zone editor logic and safety, live zone updates, event review tab.
"""

import os

import numpy as np
import pytest
import yaml

from pharmaguard.config import PharmaGuardConfig, ShelfZone
from pharmaguard.ui_zones import (add_point, draft_to_zone, new_draft, render_preview, resolve_config_path,
                                  save_zones_yaml, undo_point)
from pharmaguard.ui_review import clip_value, events_table, format_event_details, review_stats
from pharmaguard.storage import EventStore


# --------------------------------------------------------------------------- zone editor: pure logic

def _draft(points, w=640, h=480):
    d = new_draft(np.zeros((h, w, 3), np.uint8))
    for p in points:
        d = add_point(d, *p)
    return d


def test_draft_points_add_undo_and_clamp():
    d = _draft([(10, 10), (700, 500)])               # second point outside the image -> clamped
    assert d["points"] == [[10.0, 10.0], [639.0, 479.0]]
    assert undo_point(d)["points"] == [[10.0, 10.0]]


def test_draft_to_zone_normalises_coordinates():
    zone, err = draft_to_zone(_draft([(64, 48), (576, 48), (576, 264), (64, 264)]), "Shelf_A", 20, [])
    assert err == "" and isinstance(zone, ShelfZone)
    assert zone.polygon_normalized == [[0.1, 0.1], [0.9, 0.1], [0.9, 0.55], [0.1, 0.55]]


@pytest.mark.parametrize("points,name,existing,fragment", [
    ([(0, 0), (100, 0), (100, 100)], "", [], "name"),
    ([(0, 0), (100, 0)], "Z", [], "at least 3"),
    ([(0, 0), (100, 100), (200, 200)], "Z", [], "too small"),                 # collinear
    ([(0, 0), (300, 0), (300, 300)], "Dup", [{"name": "Dup"}], "already exists"),
])
def test_draft_validation_errors(points, name, existing, fragment):
    zone, err = draft_to_zone(_draft(points), name, 20, existing)
    assert zone is None and fragment in err


def test_render_preview_draws_zones_and_draft():
    frame = np.zeros((480, 640, 3), np.uint8)
    zones = [ShelfZone().model_dump()]
    out = render_preview(frame, zones, _draft([(300, 300), (400, 300), (400, 400)]))
    assert out.shape == frame.shape and out.any() and not frame.any()     # input untouched


@pytest.mark.parametrize("bad", ["../evil.yaml", "..\\evil.yaml", "/etc/passwd", "notes.txt", ".yaml", ""])
def test_config_save_path_is_confined_to_configs_dir(tmp_path, bad):
    if bad in ("../evil.yaml", "..\\evil.yaml"):
        # traversal is neutralised to configs/evil.yaml, never outside it
        assert os.path.dirname(resolve_config_path(bad, str(tmp_path))) == os.path.abspath(str(tmp_path))
    else:
        with pytest.raises(ValueError):
            resolve_config_path(bad, str(tmp_path))


def test_save_zones_preserves_other_settings_and_loads(tmp_path):
    cfg_file = tmp_path / "demo.yaml"
    cfg_file.write_text("behavior:\n  alert_threshold: 1.2\nzones: []\n", encoding="utf-8")
    zone, _ = draft_to_zone(_draft([(64, 48), (576, 48), (576, 264)]), "New_Shelf", 15, [])
    save_zones_yaml("demo.yaml", [zone.model_dump()], base_dir=str(tmp_path))
    data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    assert data["behavior"]["alert_threshold"] == 1.2
    loaded = PharmaGuardConfig.load_from_yaml(str(cfg_file))
    assert [z.name for z in loaded.zones] == ["New_Shelf"] and loaded.zones[0].prolonged_presence_sec == 15


# --------------------------------------------------------------------------- live zone update

def test_set_zones_takes_effect_on_next_frame():
    from behavior_scenarios import person_box, skeleton
    from pharmaguard.detector import DetectionResult
    from pharmaguard.pipeline import PharmaGuardPipeline

    class OnePerson:
        def __init__(self, config):
            self.config = config.detector

        def detect(self, frame):
            kp = skeleton(320, "reach_R")                       # right wrist at x=440
            return DetectionResult(person_box(kp)[None], np.array([0.9], np.float32), np.zeros(1, np.int32),
                                   ["person"], keypoints=kp[None])

        def get_model_info(self):
            return {"device": "cpu", "model_name": "fake", "has_keypoints": True}

    cfg = PharmaGuardConfig()
    cfg.objects.enabled = False
    cfg.recording.enabled = False
    p = PharmaGuardPipeline(cfg, detector=OnePerson(cfg))
    frame = np.zeros((480, 640, 3), np.uint8)
    for i in range(5):
        p.process_frame(frame, timestamp=i * 0.1)
    assert p.behavior_analyzer.zone_activity() == {"Prescription_Shelf_A": "CONTACT"}
    p.set_zones([ShelfZone(name="Left_Only", polygon_normalized=[[0, 0.1], [0.3, 0.1], [0.3, 0.5], [0, 0.5]])])
    for i in range(5, 10):
        p.process_frame(frame, timestamp=i * 0.1)
    assert set(p.behavior_analyzer.zone_activity()) == {"Left_Only"}
    assert p.behavior_analyzer.zone_activity()["Left_Only"] != "CONTACT"
    assert p.last_raw_frame is not None and p.last_raw_frame.shape == (480, 640, 3)


# --------------------------------------------------------------------------- review: pure logic

def _alert(uid, severity, tid=1):
    from pharmaguard.alerts import SecurityAlert
    import time
    return SecurityAlert(alert_id=uid, timestamp_iso="2026-11-19 10:00:00", epoch_timestamp=time.time(),
                         severity=severity, zone_name="Mock_Shelf", tracker_id=tid, title=f"{severity} <b>x</b>",
                         message="m", score=1.1, reasons=["hand <script> to bag"], incident_id=1,
                         event_type="SUSPICION_ALERT")


def test_review_table_filters_and_false_alarm_rate(tmp_path):
    store = EventStore(str(tmp_path))
    for i in range(4):
        store.record_alert(_alert(f"A{i}", "ALERT"))
    store.record_alert(_alert("W0", "WARNING"))
    rows, uids = events_table(store)
    assert len(rows) == 5 and len(uids) == 5
    assert len(events_table(store, severity="WARNING")[0]) == 1
    assert review_stats(store)["alerts"]["false_alarm_rate"] is None          # nothing reviewed yet
    store.set_review("A0", "confirmed", "op")
    store.set_review("A1", "false_alarm", "op")
    store.set_review("A2", "false_alarm", "op")
    store.set_review("A3", "dismissed", "op")                                  # excluded from the rate
    stats = review_stats(store)
    assert stats["alerts"]["false_alarm_rate"] == pytest.approx(2 / 3)
    assert stats["alerts"]["unreviewed"] == 0 and stats["reviews"]["unreviewed"] == 1
    assert len(events_table(store, status="false_alarm")[0]) == 2


def test_event_details_escape_and_framing(tmp_path):
    store = EventStore(str(tmp_path))
    store.record_alert(_alert("A0", "ALERT"))
    html = format_event_details(store.get_event("A0"))
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "does not determine theft" in html and "not acknowledged" in html
    assert clip_value(store.get_event("A0")) is None


# --------------------------------------------------------------------------- review + zones: dashboard handlers

class Sel:
    """Minimal stand-in for gr.SelectData."""
    def __init__(self, index):
        self.index = index


def test_review_tab_end_to_end_with_recorded_clip(tmp_path):
    from behavior_scenarios import Actor, FPS
    from test_live_streaming_integration import FixedBagDetector, ScriptedPoseDetector
    from pharmaguard.pipeline import PharmaGuardPipeline
    from pharmaguard.ui import create_pharmaguard_ui

    actor = Actor(1, 320, [(1.0, "rest", None), (1.5, "reach_R", None), (0.3, "rest", None),
                           (1.2, "bag_R", None), (2.0, "rest", None)], bag=True)
    frames = actor.frames()
    cfg = PharmaGuardConfig()
    cfg.objects.enabled = False
    cfg.recording.data_dir = str(tmp_path)
    pipeline = PharmaGuardPipeline(cfg, detector=ScriptedPoseDetector(frames, cfg))
    pipeline.object_detector = FixedBagDetector(actor)
    blank = np.zeros((480, 640, 3), np.uint8)
    for i in range(len(frames) + int(12 * FPS)):
        pipeline.process_frame(blank, timestamp=i / FPS)
    pipeline.evidence.wait_idle(120)

    demo = create_pharmaguard_ui(pipeline)
    fns = {getattr(f.fn, "__name__", ""): f.fn for f in demo.fns.values()}
    rows, uids, stats = fns["load"]("All", "All")
    assert len(rows) == 2 and "2</b> recorded events" in stats
    alert_row = next(i for i, r in enumerate(rows) if r[1] == "ALERT")
    uid, details, clip, verdict, note = fns["select"](uids, Sel([alert_row, 0]))
    assert "Why it was flagged" in details and "bag" in details
    assert clip and os.path.exists(clip)                                     # playable clip served to gr.Video
    msg, details, rows, uids, stats = fns["save"](uid, "false_alarm", "staged test", "All", "All")
    assert "False alarm" in msg and "staged test" in details
    assert "false-alarm rate <b>100%</b>" in stats
    assert pipeline.store.get_event(uid).review_status == "false_alarm"


def test_zone_tab_handlers_add_apply_save(tmp_path, monkeypatch):
    import pharmaguard.ui_zones as uz
    from pharmaguard.pipeline import PharmaGuardPipeline
    from pharmaguard.ui import create_pharmaguard_ui

    monkeypatch.setattr(uz, "CONFIGS_DIR", str(tmp_path))
    cfg = PharmaGuardConfig()
    cfg.objects.enabled = False
    cfg.recording.enabled = False
    pipeline = PharmaGuardPipeline(cfg)
    pipeline.process_frame(np.zeros((480, 640, 3), np.uint8))
    demo = create_pharmaguard_ui(pipeline)
    fns = {getattr(f.fn, "__name__", ""): f.fn for f in demo.fns.values()}

    frame, draft, canvas, msg = fns["on_snapshot"]([])
    assert msg == "Snapshot captured."
    for x, y in [(64, 48), (576, 48), (576, 264), (64, 264)]:
        draft, canvas, _ = fns["on_click"](frame, [], draft, Sel([x, y]))
    zones, draft, canvas, status, _, _ = fns["on_add"](frame, [], draft, "Counter_Shelf", 25)
    assert [z["name"] for z in zones] == ["Counter_Shelf"] and draft["points"] == []
    status, _ = fns["on_apply"]([])
    assert "Refusing" in status and pipeline.config.zones[0].name == "Prescription_Shelf_A"
    status, _ = fns["on_apply"](zones)
    assert [z.name for z in pipeline.config.zones] == ["Counter_Shelf"]
    assert pipeline.behavior_analyzer.zones[0].name == "Counter_Shelf"
    assert "Saved 1 zone" in fns["on_save"](zones, "staged.yaml")
    assert "Save failed" in fns["on_save"](zones, "../../outside.txt")
    assert PharmaGuardConfig.load_from_yaml(str(tmp_path / "staged.yaml")).zones[0].name == "Counter_Shelf"
