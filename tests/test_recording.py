"""
Event recording tests (Priority 5): SQLite store, evidence clips, retention, Drive-style mirroring,
and full-pipeline integration.
"""

import os
import sqlite3
import time

import cv2
import numpy as np
import pytest

from pharmaguard.alerts import SecurityAlert
from pharmaguard.config import PharmaGuardConfig, RecordingConfig
from pharmaguard.evidence import ClipWriter, EvidenceRecorder
from pharmaguard.storage import EventStore


def alert(uid="PG-1-0001", severity="ALERT", epoch=None, tid=4, incident=2):
    return SecurityAlert(alert_id=uid, timestamp_iso="2026-11-19 10:00:00",
                         epoch_timestamp=epoch if epoch is not None else time.time(), severity=severity,
                         zone_name="Mock_Shelf", tracker_id=tid, title="SUSPICIOUS SEQUENCE - review now",
                         message="msg", score=1.1, reasons=["R hand moved from Mock_Shelf to a bag"],
                         incident_id=incident, event_type="SUSPICION_ALERT")


# --------------------------------------------------------------------------- store

def test_store_records_required_fields_and_is_idempotent(tmp_path):
    store = EventStore(str(tmp_path), camera_id="cam-7")
    assert store.record_alert(alert()) is True
    assert store.record_alert(alert()) is False              # same alert id: no duplicate row
    ev = store.get_event("PG-1-0001")
    assert (ev.camera_id, ev.tracker_id, ev.incident_id, ev.category, ev.severity) == \
           ("cam-7", 4, 2, "SUSPICION_ALERT", "ALERT")
    assert ev.score == pytest.approx(1.1) and ev.reasons == ["R hand moved from Mock_Shelf to a bag"]
    assert ev.created_at == "2026-11-19 10:00:00" and ev.review_status == "open" and ev.clip_status == "none"
    assert store.counts() == {"ALERT": 1}


def test_store_ack_reset_review_and_audit(tmp_path):
    store = EventStore(str(tmp_path))
    store.record_alert(alert())
    store.mark_acknowledged(["PG-1-0001"], "pharmacist_a")
    store.mark_acknowledged(["PG-1-0001"], "someone_else")    # first acknowledgement is kept
    store.mark_reset(["PG-1-0001"])
    assert store.set_review("PG-1-0001", "false_alarm", "pharmacist_a", "customer reading label")
    with pytest.raises(ValueError):
        store.set_review("PG-1-0001", "guilty", "x")
    ev = store.get_event("PG-1-0001")
    assert ev.acknowledged_by == "pharmacist_a" and ev.reset_at
    assert ev.review_status == "false_alarm" and ev.review_note == "customer reading label"
    assert store.list_events(review_status="false_alarm")[0].event_uid == "PG-1-0001"
    assert store.list_audit()[0]["action"] == "EVENT_REVIEWED"


def test_store_persists_across_restart(tmp_path):
    EventStore(str(tmp_path)).record_alert(alert())
    assert EventStore(str(tmp_path)).get_event("PG-1-0001") is not None


def test_retention_purge_deletes_old_events_and_their_clips(tmp_path):
    store = EventStore(str(tmp_path))
    store.record_alert(alert("OLD", epoch=time.time() - 10 * 86400))
    store.record_alert(alert("NEW"))
    clip = tmp_path / "old.mp4"
    clip.write_bytes(b"x")
    store.set_clip("OLD", "saved", str(clip), "h264")
    assert store.purge_older_than(7) == 1
    assert store.get_event("OLD") is None and store.get_event("NEW") is not None
    assert not clip.exists()


def test_mirror_produces_a_valid_database_copy(tmp_path):
    store = EventStore(str(tmp_path / "local"), mirror_dir=str(tmp_path / "drive"))
    store.record_alert(alert())
    target = store.mirror()
    assert target and os.path.exists(target)
    rows = sqlite3.connect(target).execute("SELECT event_uid, severity FROM events").fetchall()
    assert rows == [("PG-1-0001", "ALERT")]


# --------------------------------------------------------------------------- clip writer

def _frames(n=20, w=640, h=360):
    base = np.zeros((h, w, 3), np.uint8)
    out = []
    for i in range(n):
        f = base.copy()
        cv2.rectangle(f, (10 + 20 * i, 100), (90 + 20 * i, 300), (0, 200, 255), -1)
        out.append(f)
    return out


def _read_count(path):
    cap = cv2.VideoCapture(path)
    n = 0
    while cap.read()[0]:
        n += 1
    return n


@pytest.mark.parametrize("prefer_ffmpeg", [True, False])
def test_clip_writer_produces_readable_video(tmp_path, prefer_ffmpeg):
    path, status, codec = ClipWriter(prefer_ffmpeg=prefer_ffmpeg).write(_frames(), 10.0, str(tmp_path / "clip"))
    assert status in ("saved", "saved_not_browser_playable") and path and os.path.exists(path)
    assert _read_count(path) == 20
    print("codec used:", codec, status)


def test_clip_writer_handles_odd_dimensions(tmp_path):
    path, status, _ = ClipWriter().write(_frames(w=641, h=361), 10.0, str(tmp_path / "odd"))
    assert path and _read_count(path) == 20


# --------------------------------------------------------------------------- evidence recorder

class CapturingWriter:
    def __init__(self):
        self.calls = []

    def write(self, frames, fps, out_base):
        self.calls.append((len(frames), fps, out_base))
        return out_base + ".mp4", "saved", "fake"


def _recorder(tmp_path, **cfg):
    done, writer = [], CapturingWriter()
    rec = EvidenceRecorder(RecordingConfig(**cfg), str(tmp_path / "clips"),
                           on_clip_done=lambda *a: done.append(a), writer=writer)
    return rec, writer, done


def test_clip_contains_pre_and_post_event_frames_including_trigger_frame(tmp_path):
    rec, writer, done = _recorder(tmp_path, pre_event_sec=5, post_event_sec=5, max_clip_fps=15)
    frame = np.zeros((10, 10, 3), np.uint8)
    for i in range(200):                      # 10 fps for 20 s
        t = i / 10
        if i == 60:
            rec.trigger("EV1", t)             # alert at t = 6.0, before that frame is added (pipeline order)
        rec.add_frame(frame, t)
    rec.wait_idle(10)
    assert done == [("EV1", "saved", str(tmp_path / "clips" / "EV1") + ".mp4", "fake")]
    n, fps, _ = writer.calls[0]
    assert n == 101                           # t = 1.0 .. 11.0 inclusive at 10 fps
    assert fps == pytest.approx(10.0, abs=0.1)


def test_ring_buffer_is_bounded_and_rate_capped(tmp_path):
    rec, writer, _ = _recorder(tmp_path, pre_event_sec=2, post_event_sec=0, max_clip_fps=10)
    frame = np.zeros((10, 10, 3), np.uint8)
    for i in range(300):                      # 30 fps input
        rec.add_frame(frame, i / 30)
    assert len(rec._buffer) <= 2 * 10 + 1
    rec.trigger("EV2", 300 / 30)
    rec.add_frame(frame, 300 / 30 + 0.2)
    rec.wait_idle(10)
    assert writer.calls[0][0] <= 22


def test_flush_finalises_partial_clip(tmp_path):
    rec, writer, done = _recorder(tmp_path, pre_event_sec=1, post_event_sec=30)
    frame = np.zeros((10, 10, 3), np.uint8)
    for i in range(20):
        rec.add_frame(frame, i / 10)
    rec.trigger("EV3", 2.0)
    rec.add_frame(frame, 2.0)
    assert rec.pending_jobs == 1 and done == []
    rec.flush()
    rec.wait_idle(10)
    assert done and done[0][0] == "EV3"


def test_writer_crash_is_reported_and_worker_survives(tmp_path):
    class Boom:
        def write(self, *a):
            raise RuntimeError("codec exploded")
    done = []
    rec = EvidenceRecorder(RecordingConfig(pre_event_sec=1, post_event_sec=0), str(tmp_path),
                           on_clip_done=lambda *a: done.append(a), writer=Boom())
    f = np.zeros((10, 10, 3), np.uint8)
    rec.add_frame(f, 0.0)
    rec.trigger("A", 0.5); rec.add_frame(f, 0.5)
    rec.trigger("B", 1.0); rec.add_frame(f, 1.0)
    rec.wait_idle(10)
    assert [d[:2] for d in done] == [("A", "failed"), ("B", "failed")]


# --------------------------------------------------------------------------- full pipeline

def test_pipeline_records_events_clips_ack_and_mirrors(tmp_path):
    from behavior_scenarios import Actor, FPS
    from test_live_streaming_integration import FixedBagDetector, ScriptedPoseDetector
    from pharmaguard.pipeline import PharmaGuardPipeline

    actor = Actor(1, 320, [(1.0, "rest", None), (1.5, "reach_R", None), (0.3, "rest", None),
                           (1.2, "bag_R", None), (2.0, "rest", None)], bag=True)
    frames = actor.frames()
    cfg = PharmaGuardConfig()
    cfg.objects.enabled = False
    cfg.recording.data_dir = str(tmp_path / "local")
    cfg.recording.mirror_dir = str(tmp_path / "drive")
    cfg.recording.camera_id = "mock-shelf-cam"
    pipeline = PharmaGuardPipeline(cfg, detector=ScriptedPoseDetector(frames, cfg))
    pipeline.object_detector = FixedBagDetector(actor)
    blank = np.zeros((480, 640, 3), np.uint8)
    for i in range(len(frames) + int(12 * FPS)):          # long enough for post-event windows to close
        pipeline.process_frame(blank, timestamp=i / FPS)
    pipeline.alarm.acknowledge("pharmacist_b")
    store = pipeline.store
    pipeline.evidence.wait_idle(120)

    events = store.list_events()
    assert sorted(e.severity for e in events) == ["ALERT", "WARNING"]
    alert_ev = next(e for e in events if e.severity == "ALERT")
    assert alert_ev.camera_id == "mock-shelf-cam" and alert_ev.tracker_id == 1 and alert_ev.incident_id >= 1
    assert alert_ev.score >= 1.0 and any("bag" in r for r in alert_ev.reasons)
    assert alert_ev.acknowledged_by == "pharmacist_b"
    for e in events:
        assert e.clip_status == "saved", (e.event_uid, e.clip_status, e.clip_codec)
        assert os.path.exists(e.clip_path) and _read_count(e.clip_path) > 50     # ~5 s pre + 5 s post at 10 fps
    actions = [a["action"] for a in store.list_audit()]
    assert "ALARM_RAISED" in actions and "ALARM_ACKNOWLEDGED" in actions

    pipeline.close()
    mirrored = sqlite3.connect(os.path.join(cfg.recording.mirror_dir, "pharmaguard.db"))
    assert mirrored.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 2
    assert len(os.listdir(os.path.join(cfg.recording.mirror_dir, "clips"))) == 2
