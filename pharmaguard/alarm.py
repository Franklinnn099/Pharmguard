"""
Operator alarm controller for PharmaGuard.

State machine (server-side, shared by every connected dashboard):

    IDLE --ALERT--> ACTIVE (sounding) --acknowledge--> ACKNOWLEDGED --reset--> IDLE
      |                 |  (auto-silence after max_sound_sec; banner remains)
      +--test-------> TEST (sounding, auto-ends after test_duration_sec; never a security record)

Duplicate prevention: an incident (tracker_id, incident_id) can sound at most ONCE, ever. This is on top
of the behaviour engine's per-incident latch and AlertManager's cooldown. A new incident while an alarm is
active joins the active list and restarts the siren (new ``sound_seq``) so the operator hears it.

REVIEW (WARNING) alerts never sound; they are shown in the banner for ``review_banner_sec``.

The browser plays the siren; it learns what to do from :class:`AlarmSnapshot`
(``sounding`` + ``sound_seq``) rendered into the dashboard.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set, Tuple
import threading
import time

from pharmaguard.alerts import SecurityAlert
from pharmaguard.config import AlarmConfig

IDLE, ACTIVE, ACKNOWLEDGED, TEST = "IDLE", "ACTIVE", "ACKNOWLEDGED", "TEST"


@dataclass
class AlarmSnapshot:
    state: str
    sounding: bool
    sound_seq: int                      # increments each time the browser should (re)start the siren
    kind: str                           # "ALERT", "TEST" or "" (what is sounding / shown)
    active_alerts: List[SecurityAlert] = field(default_factory=list)
    latest_review: Optional[SecurityAlert] = None
    raised_at_iso: str = ""
    acknowledged_by: str = ""
    acknowledged_at_iso: str = ""
    silenced_reason: str = ""           # "", "acknowledged", "timeout"


ActionCallback = Callable[[str, Dict], None]


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


class AlarmController:
    """Thread-safe alarm state shared by the pipeline (raises) and the dashboard (ack/reset/test)."""

    def __init__(self, config: Optional[AlarmConfig] = None,
                 clock: Callable[[], float] = time.time,
                 on_action: Optional[ActionCallback] = None):
        self.config = config or AlarmConfig()
        self._clock = clock
        self._on_action = on_action
        self._lock = threading.Lock()
        self._alarmed_incidents: Set[Tuple[int, int]] = set()
        self._reset_state()
        self.sound_seq = 0
        self.total_alarms = 0

    def _reset_state(self) -> None:
        self.state = IDLE
        self.active_alerts: List[SecurityAlert] = []
        self.raised_at: Optional[float] = None
        self.sound_started_at: Optional[float] = None
        self.ack_by = ""
        self.ack_at: Optional[float] = None
        self.silenced_reason = ""
        self.test_started_at: Optional[float] = None
        self.latest_review: Optional[SecurityAlert] = None
        self.latest_review_at: Optional[float] = None

    def _emit(self, action: str, details: Dict) -> None:
        if self._on_action is not None:
            self._on_action(action, details)

    # ------------------------------------------------------------------ inputs
    def on_alerts(self, alerts: List[SecurityAlert]) -> List[SecurityAlert]:
        """Feed new SecurityAlerts. Returns the ALERT-level ones that actually started/joined an alarm."""
        sounded: List[SecurityAlert] = []
        with self._lock:
            now = self._clock()
            for a in alerts:
                if a.severity == "WARNING":
                    self.latest_review, self.latest_review_at = a, now
                    continue
                if a.severity != "ALERT":
                    continue
                key = (a.tracker_id, a.incident_id) if a.incident_id else (a.tracker_id, -int(a.epoch_timestamp))
                if key in self._alarmed_incidents:
                    continue                                    # this incident already sounded once
                self._alarmed_incidents.add(key)
                if self.state != ACTIVE:
                    if self.state in (IDLE, TEST):
                        self.active_alerts = []      # ACKNOWLEDGED: keep earlier alerts listed until reset
                    self.state = ACTIVE
                    self.raised_at = now
                    self.ack_by, self.ack_at = "", None
                self.active_alerts.append(a)
                self.sound_started_at = now
                self.silenced_reason = ""
                self.test_started_at = None
                self.sound_seq += 1
                self.total_alarms += 1
                sounded.append(a)
        for a in sounded:
            self._emit("ALARM_RAISED", {"alert_id": a.alert_id, "tracker_id": a.tracker_id,
                                        "incident_id": a.incident_id, "score": a.score})
        return sounded

    def acknowledge(self, operator: str = "operator") -> bool:
        """Silence the siren. Alerts stay on screen until reset. Returns False if nothing to acknowledge."""
        with self._lock:
            if self.state == TEST:
                self.state, self.test_started_at = IDLE, None
                self.sound_seq += 1
                return True
            if self.state != ACTIVE:
                return False
            now = self._clock()
            self.state = ACKNOWLEDGED
            self.ack_by, self.ack_at = operator or "operator", now
            self.silenced_reason = "acknowledged"
            ids = [a.alert_id for a in self.active_alerts]
        self._emit("ALARM_ACKNOWLEDGED", {"operator": operator, "alert_ids": ids, "at": _iso(now)})
        return True

    def reset(self, operator: str = "operator") -> bool:
        """Return to monitoring. Unacknowledged alarms are acknowledged implicitly (and recorded as such)."""
        with self._lock:
            if self.state == IDLE and self.latest_review is None:
                return False
            ids = [a.alert_id for a in self.active_alerts]
            was = self.state
            self._reset_state()
            self.sound_seq += 1
        self._emit("ALARM_RESET", {"operator": operator, "alert_ids": ids, "previous_state": was})
        return True

    def test(self, operator: str = "operator") -> bool:
        """Manual sound/visual check. Refused while a real alarm is active (never mask a real alarm)."""
        with self._lock:
            if self.state == ACTIVE:
                return False
            self.state = TEST
            self.test_started_at = self._clock()
            self.sound_seq += 1
        self._emit("ALARM_TEST", {"operator": operator})
        return True

    # ------------------------------------------------------------------ output
    def snapshot(self) -> AlarmSnapshot:
        with self._lock:
            now = self._clock()
            # Time-based transitions are evaluated lazily here.
            if self.state == TEST and self.test_started_at is not None \
                    and now - self.test_started_at >= self.config.test_duration_sec:
                self.state, self.test_started_at = IDLE, None
            if self.state == ACTIVE and not self.silenced_reason and self.sound_started_at is not None \
                    and now - self.sound_started_at >= self.config.max_sound_sec:
                self.silenced_reason = "timeout"
            review = self.latest_review
            if review is not None and self.latest_review_at is not None \
                    and now - self.latest_review_at > self.config.review_banner_sec:
                review = None
            sounding = self.config.sound_enabled and (
                self.state == TEST or (self.state == ACTIVE and not self.silenced_reason))
            kind = "TEST" if self.state == TEST else ("ALERT" if self.state in (ACTIVE, ACKNOWLEDGED) else "")
            return AlarmSnapshot(
                state=self.state, sounding=sounding, sound_seq=self.sound_seq, kind=kind,
                active_alerts=list(self.active_alerts), latest_review=review,
                raised_at_iso=_iso(self.raised_at) if self.raised_at else "",
                acknowledged_by=self.ack_by,
                acknowledged_at_iso=_iso(self.ack_at) if self.ack_at else "",
                silenced_reason=self.silenced_reason,
            )
