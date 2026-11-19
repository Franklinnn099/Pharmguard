"""
Alarm system tests (Priority 4): sounding rules, duplicate prevention, acknowledge/reset, manual test,
auto-silence, audit trail, dashboard rendering, and full-pipeline integration.
"""

import json

import numpy as np

from pharmaguard.alarm import ACKNOWLEDGED, ACTIVE, IDLE, TEST, AlarmController
from pharmaguard.alerts import SecurityAlert
from pharmaguard.config import AlarmConfig, PharmaGuardConfig


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


def alert(tid=1, incident=1, severity="ALERT", score=1.1, n=1):
    return SecurityAlert(alert_id=f"PG-{tid}-{incident}-{n}", timestamp_iso="2026-11-19 10:00:00",
                         epoch_timestamp=1.0, severity=severity, zone_name="Mock_Shelf", tracker_id=tid,
                         title="t", message="m", score=score,
                         reasons=["R hand moved from Mock_Shelf to a bag 0.4s after shelf contact"],
                         incident_id=incident)


def make(**cfg):
    clk = Clock()
    actions = []
    ctl = AlarmController(AlarmConfig(**cfg), clock=clk, on_action=lambda a, d: actions.append((a, d)))
    return ctl, clk, actions


# --------------------------------------------------------------------------- sounding rules

def test_alert_sounds_review_does_not():
    ctl, _, _ = make()
    ctl.on_alerts([alert(severity="WARNING", score=0.7)])
    s = ctl.snapshot()
    assert s.state == IDLE and not s.sounding and s.latest_review is not None
    ctl.on_alerts([alert()])
    s = ctl.snapshot()
    assert s.state == ACTIVE and s.sounding and s.kind == "ALERT" and s.sound_seq == 1


def test_same_incident_never_sounds_twice_even_after_reset():
    ctl, _, _ = make()
    assert len(ctl.on_alerts([alert()])) == 1
    assert ctl.on_alerts([alert(n=2)]) == []            # duplicate (e.g. cooldown expired upstream)
    ctl.acknowledge("op")
    ctl.reset("op")
    assert ctl.on_alerts([alert(n=3)]) == []            # still the same incident
    assert ctl.snapshot().state == IDLE
    assert ctl.total_alarms == 1


def test_new_incident_while_active_joins_and_restarts_siren():
    ctl, _, _ = make()
    ctl.on_alerts([alert(tid=1, incident=1)])
    seq = ctl.snapshot().sound_seq
    ctl.on_alerts([alert(tid=2, incident=2)])
    s = ctl.snapshot()
    assert s.state == ACTIVE and len(s.active_alerts) == 2 and s.sound_seq == seq + 1


def test_new_incident_after_acknowledge_resounds_and_keeps_history():
    ctl, _, _ = make()
    ctl.on_alerts([alert(tid=1, incident=1)])
    ctl.acknowledge("op")
    assert not ctl.snapshot().sounding
    ctl.on_alerts([alert(tid=2, incident=2)])
    s = ctl.snapshot()
    assert s.state == ACTIVE and s.sounding and len(s.active_alerts) == 2


# --------------------------------------------------------------------------- acknowledge / reset

def test_acknowledge_silences_and_records_operator_then_reset_clears():
    ctl, clk, actions = make()
    ctl.on_alerts([alert()])
    clk.t += 4
    assert ctl.acknowledge("night_pharmacist")
    s = ctl.snapshot()
    assert s.state == ACKNOWLEDGED and not s.sounding and s.acknowledged_by == "night_pharmacist"
    assert s.active_alerts                           # still shown until reset
    assert ctl.reset("night_pharmacist")
    s = ctl.snapshot()
    assert s.state == IDLE and not s.active_alerts
    kinds = [a for a, _ in actions]
    assert kinds == ["ALARM_RAISED", "ALARM_ACKNOWLEDGED", "ALARM_RESET"]
    assert actions[1][1]["operator"] == "night_pharmacist"


def test_acknowledge_with_nothing_active_is_a_noop():
    ctl, _, actions = make()
    assert ctl.acknowledge("op") is False
    assert ctl.reset("op") is False
    assert actions == []


def test_unacknowledged_alarm_auto_silences_but_stays_active():
    ctl, clk, _ = make(max_sound_sec=30)
    ctl.on_alerts([alert()])
    clk.t += 31
    s = ctl.snapshot()
    assert s.state == ACTIVE and not s.sounding and s.silenced_reason == "timeout"


def test_review_banner_expires():
    ctl, clk, _ = make(review_banner_sec=10)
    ctl.on_alerts([alert(severity="WARNING")])
    assert ctl.snapshot().latest_review is not None
    clk.t += 11
    assert ctl.snapshot().latest_review is None


# --------------------------------------------------------------------------- manual test

def test_manual_test_sounds_then_auto_ends_and_is_not_a_security_alert():
    ctl, clk, actions = make(test_duration_sec=5)
    assert ctl.test("op")
    s = ctl.snapshot()
    assert s.state == TEST and s.sounding and s.kind == "TEST"
    clk.t += 5.1
    s = ctl.snapshot()
    assert s.state == IDLE and not s.sounding
    assert ctl.total_alarms == 0
    assert [a for a, _ in actions] == ["ALARM_TEST"]


def test_manual_test_refused_while_real_alarm_active():
    ctl, _, _ = make()
    ctl.on_alerts([alert()])
    assert ctl.test("op") is False
    assert ctl.snapshot().kind == "ALERT"


def test_real_alert_during_test_takes_over():
    ctl, _, _ = make()
    ctl.test("op")
    ctl.on_alerts([alert()])
    s = ctl.snapshot()
    assert s.state == ACTIVE and s.kind == "ALERT" and s.sounding


def test_sound_can_be_disabled_by_config_but_banner_remains():
    ctl, _, _ = make(sound_enabled=False)
    ctl.on_alerts([alert()])
    s = ctl.snapshot()
    assert s.state == ACTIVE and not s.sounding


# --------------------------------------------------------------------------- dashboard rendering

def test_banner_carries_machine_readable_siren_state():
    from pharmaguard.ui import format_alarm_html
    ctl, _, _ = make()
    idle = format_alarm_html(ctl.snapshot())
    assert "id='pg-alarm'" in idle and "data-sounding='0'" in idle and "Monitoring" in idle
    ctl.on_alerts([alert()])
    active = format_alarm_html(ctl.snapshot(), volume=0.3)
    assert "data-sounding='1'" in active and "data-seq='1'" in active and "data-kind='ALERT'" in active
    assert "data-volume='0.30'" in active and "Person #1" in active and "bag" in active
    assert "does not establish theft" in active


def test_banner_escapes_untrusted_text():
    from pharmaguard.ui import format_alarm_html
    ctl, _, _ = make()
    a = alert()
    a.zone_name = "<script>x</script>"
    ctl.on_alerts([a])
    assert "<script>x" not in format_alarm_html(ctl.snapshot())


def test_dashboard_handlers_ack_reset_test():
    from pharmaguard.pipeline import PharmaGuardPipeline
    from pharmaguard.ui import create_pharmaguard_ui
    cfg = PharmaGuardConfig()
    cfg.objects.enabled = False
    pipeline = PharmaGuardPipeline(cfg)
    demo = create_pharmaguard_ui(pipeline)
    fns = {getattr(f.fn, "__name__", ""): f.fn for f in demo.fns.values()}
    html, _ = fns["on_alarm_test"]()
    assert "ALARM TEST" in html and "data-sounding='1'" in html
    pipeline.alarm.on_alerts([alert()])                  # real alarm takes over the test
    html, _ = fns["on_acknowledge"]()
    assert "acknowledged by operator" in html.lower() and "data-sounding='0'" in html
    html, _ = fns["on_alarm_reset"]()
    assert "Monitoring" in html
    first, cache = fns["refresh_alarm"]("")
    unchanged, _ = fns["refresh_alarm"](cache)
    assert isinstance(first, str) and not isinstance(unchanged, str)   # unchanged banner is skipped


# --------------------------------------------------------------------------- full pipeline

def test_pipeline_shelf_to_bag_sounds_once_and_audit_log_records_ack(tmp_path):
    """Scripted shelf->bag->leave through the real tracker/behaviour/alerts/alarm: exactly one alarm,
    REVIEW banner first, flashing border in the frame, acknowledgement persisted."""
    from behavior_scenarios import Actor, FPS
    from test_live_streaming_integration import FixedBagDetector, ScriptedPoseDetector
    from pharmaguard.pipeline import PharmaGuardPipeline

    actor = Actor(1, 320, [(1.0, "rest", None), (1.5, "reach_R", None), (0.3, "rest", None),
                           (1.2, "bag_R", None), (2.0, "rest", None)], bag=True)
    frames = actor.frames()
    cfg = PharmaGuardConfig()
    cfg.objects.enabled = False
    cfg.alert.alert_log_file = str(tmp_path / "alerts.jsonl")
    pipeline = PharmaGuardPipeline(cfg, detector=ScriptedPoseDetector(frames, cfg))
    pipeline.object_detector = FixedBagDetector(actor)
    blank = np.zeros((480, 640, 3), np.uint8)

    seqs, review_seen, border_frames = set(), False, 0
    for i in range(len(frames) + int(6 * FPS)):
        out = pipeline.process_frame(blank, timestamp=i / FPS)
        s = pipeline.alarm.snapshot()
        review_seen |= s.latest_review is not None and s.state == IDLE
        if s.sounding:
            seqs.add(s.sound_seq)
        if s.state == ACTIVE and out.annotated_frame_rgb[2, 320, 0] > 200:   # red border (RGB)
            border_frames += 1
    assert review_seen
    assert seqs == {1}                                  # sounded, exactly once
    assert pipeline.alarm.total_alarms == 1
    assert border_frames > 0

    pipeline.alarm.acknowledge("tester")
    records = [json.loads(line) for line in open(cfg.alert.alert_log_file, encoding="utf-8")]
    assert [r.get("severity") for r in records if "severity" in r] == ["WARNING", "ALERT"]
    ack = [r for r in records if r.get("record") == "ALARM_ACKNOWLEDGED"]
    assert ack and ack[0]["operator"] == "tester"
    timeline = [e.category for e in pipeline.event_manager.get_recent(100)]
    assert timeline.count("ALARM") >= 2
