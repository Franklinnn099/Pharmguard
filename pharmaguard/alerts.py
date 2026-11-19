"""
Alert management module for PharmaGuard.
Handles alert generation, throttling/cooldowns, severity grading,
and audit persistence for pharmacy staff.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import os
import json
import logging
import time

from pharmaguard.config import AlertConfig
from pharmaguard.behavior import BehaviorEvent, BehaviorEventType

logger = logging.getLogger(__name__)


@dataclass
class SecurityAlert:
    """A human-actionable security alert for pharmacy personnel."""
    alert_id: str
    timestamp_iso: str
    epoch_timestamp: float
    severity: str  # "INFO", "WARNING", "ALERT"
    zone_name: str
    tracker_id: int
    title: str
    message: str
    dwell_time_seconds: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp_iso,
            "severity": self.severity,
            "zone": self.zone_name,
            "tracker_id": self.tracker_id,
            "title": self.title,
            "message": self.message,
            "dwell_time_sec": self.dwell_time_seconds
        }


class AlertManager:
    """Manages alert lifecycle, prevents alert spamming via cooldowns, and maintains logs."""

    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self.alert_history: List[SecurityAlert] = []
        # Maps (tracker_id, event_type) -> last_triggered_epoch
        self._last_alert_triggered: Dict[tuple, float] = {}
        self._alert_counter = 0

        if self.config.alert_log_file:
            log_dir = os.path.dirname(os.path.abspath(self.config.alert_log_file))
            os.makedirs(log_dir, exist_ok=True)

    def process_behavior_events(self, events: List[BehaviorEvent]) -> List[SecurityAlert]:
        """Convert behavior events into actionable alerts if cooldown has expired."""
        generated_alerts: List[SecurityAlert] = []
        now = time.time()

        for event in events:
            # We only generate alerts for WARNING and ALERT events, or significant status changes
            if event.severity not in ("WARNING", "ALERT"):
                continue

            cooldown_key = (event.tracker_id, event.zone_name, event.event_type)
            last_time = self._last_alert_triggered.get(cooldown_key, 0.0)

            if (now - last_time) < self.config.cooldown_seconds:
                # In cooldown period; suppress to avoid overwhelming the operator
                continue

            self._last_alert_triggered[cooldown_key] = now
            self._alert_counter += 1

            now_iso = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
            alert_id = f"PG-{int(now)}-{self._alert_counter:04d}"

            title = self._format_alert_title(event)

            alert = SecurityAlert(
                alert_id=alert_id,
                timestamp_iso=now_iso,
                epoch_timestamp=now,
                severity=event.severity,
                zone_name=event.zone_name,
                tracker_id=event.tracker_id,
                title=title,
                message=event.description,
                dwell_time_seconds=event.dwell_time_seconds
            )

            self.alert_history.insert(0, alert)
            if len(self.alert_history) > self.config.max_history_entries:
                self.alert_history.pop()

            generated_alerts.append(alert)
            self._persist_alert(alert)

        return generated_alerts

    def _format_alert_title(self, event: BehaviorEvent) -> str:
        """Create a clear, high-signal alert title."""
        if event.event_type == BehaviorEventType.SUSPICIOUS_LOITERING:
            return f"PROLONGED DWELL: Shelf {event.zone_name}"
        elif event.event_type == BehaviorEventType.RAPID_RETREAT:
            return f"RAPID RETREAT: Shelf {event.zone_name}"
        elif event.event_type == BehaviorEventType.MULTI_PERSON_RESTRICTED:
            return f"CONGREGATION: Shelf {event.zone_name}"
        return f"ACTIVITY: Shelf {event.zone_name}"

    def _persist_alert(self, alert: SecurityAlert) -> None:
        """Append alert record to JSON Lines log."""
        if not self.config.alert_log_file:
            return
        try:
            with open(self.config.alert_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")
        except OSError as exc:
            # Non-fatal for the live stream, but must never be silent: it means evidence is being lost.
            logger.warning("Failed to persist alert %s to %s: %s", alert.alert_id, self.config.alert_log_file, exc)

    def get_recent_alerts(self, limit: int = 15) -> List[SecurityAlert]:
        """Return the most recent alerts for UI presentation."""
        return self.alert_history[:limit]

    def clear(self) -> None:
        """Clear alert history and cooldown state."""
        self.alert_history.clear()
        self._last_alert_triggered.clear()
        self._alert_counter = 0
