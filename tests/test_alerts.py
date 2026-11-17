"""
Unit tests for PharmaGuard alert management and throttling.
"""

import os
import time
import pytest
from pharmaguard.config import AlertConfig
from pharmaguard.behavior import BehaviorEvent, BehaviorEventType
from pharmaguard.alerts import AlertManager, SecurityAlert


def test_alert_generation_and_cooldown(tmp_path):
    log_file = str(tmp_path / "test_alerts.jsonl")
    config = AlertConfig(cooldown_seconds=5.0, alert_log_file=log_file)
    manager = AlertManager(config)

    event = BehaviorEvent(
        event_type=BehaviorEventType.SUSPICIOUS_LOITERING,
        tracker_id=1,
        zone_name="Shelf_A",
        dwell_time_seconds=9.5,
        description="Person #1 loitering in Shelf_A",
        severity="ALERT"
    )

    # First event should trigger an alert
    alerts = manager.process_behavior_events([event])
    assert len(alerts) == 1
    assert alerts[0].tracker_id == 1
    assert alerts[0].severity == "ALERT"
    assert len(manager.alert_history) == 1
    assert os.path.exists(log_file)

    # Immediate second identical event should be suppressed by cooldown
    alerts_suppressed = manager.process_behavior_events([event])
    assert len(alerts_suppressed) == 0
    assert len(manager.alert_history) == 1


def test_info_severity_suppression():
    config = AlertConfig()
    manager = AlertManager(config)

    # INFO severity event (routine entry) should not trigger security alert
    info_event = BehaviorEvent(
        event_type=BehaviorEventType.ZONE_ENTER,
        tracker_id=2,
        zone_name="Shelf_A",
        description="Person #2 entered zone",
        severity="INFO"
    )
    alerts = manager.process_behavior_events([info_event])
    assert len(alerts) == 0
