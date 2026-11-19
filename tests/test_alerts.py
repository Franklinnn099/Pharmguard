"""
Unit tests for PharmaGuard alert management: severity filtering, frame-time cooldown, persistence.
"""

import json
import os

from pharmaguard.config import AlertConfig
from pharmaguard.behavior import BehaviorEvent, BehaviorEventType
from pharmaguard.alerts import AlertManager


def _alert_event(t: float, tid: int = 1) -> BehaviorEvent:
    return BehaviorEvent(
        event_type=BehaviorEventType.SUSPICION_ALERT,
        tracker_id=tid,
        zone_name="Shelf_A",
        timestamp=t,
        description="Suspicious interaction sequence: Person #1, score 1.10.",
        severity="ALERT",
        score=1.1,
        reasons=["R hand moved from Shelf_A to a bag 0.4s after shelf contact"],
        incident_id=3,
    )


def test_alert_generation_and_cooldown(tmp_path):
    log_file = str(tmp_path / "test_alerts.jsonl")
    manager = AlertManager(AlertConfig(cooldown_seconds=5.0, alert_log_file=log_file))

    alerts = manager.process_behavior_events([_alert_event(100.0)])
    assert len(alerts) == 1
    a = alerts[0]
    assert a.tracker_id == 1 and a.severity == "ALERT" and a.score == 1.1 and a.incident_id == 3
    assert "review" in a.title.lower()

    # Same alert within cooldown (frame time) is suppressed; after cooldown it is allowed.
    assert manager.process_behavior_events([_alert_event(102.0)]) == []
    assert len(manager.process_behavior_events([_alert_event(105.5)])) == 1
    assert len(manager.alert_history) == 2

    records = [json.loads(line) for line in open(log_file, encoding="utf-8")]
    assert records[0]["score"] == 1.1 and records[0]["incident_id"] == 3
    assert records[0]["reasons"] and records[0]["event_type"] == "SUSPICION_ALERT"


def test_info_severity_suppression():
    manager = AlertManager(AlertConfig(alert_log_file=None))
    info_event = BehaviorEvent(
        event_type=BehaviorEventType.SHELF_CONTACT,
        tracker_id=2,
        zone_name="Shelf_A",
        timestamp=0.0,
        description="Person #2 R hand reached into Shelf_A",
        severity="INFO",
    )
    assert manager.process_behavior_events([info_event]) == []


def test_persistence_failure_is_logged_not_raised(tmp_path, caplog):
    bad_path = str(tmp_path / "is_a_dir")
    os.makedirs(bad_path)
    manager = AlertManager(AlertConfig(alert_log_file=bad_path))
    alerts = manager.process_behavior_events([_alert_event(1.0)])
    assert len(alerts) == 1
    assert any("Failed to persist alert" in r.message for r in caplog.records)
