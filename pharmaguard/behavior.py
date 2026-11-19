"""
Behaviour analysis for PharmaGuard: temporal interaction sequences and suspicion scoring.

The engine does NOT recognise theft or intent. It accumulates evidence about interaction sequences and
raises a score for STAFF REVIEW:

    shelf contact (hand in shelf zone, debounced)
        -> same hand reaches waist/pocket region or a bag within ``carry_window_sec``   [concealment-like]
        -> person leaves the shelf area / camera view                                    [exit]

Scoring
  * Every signal adds weighted evidence that decays with half-life ``decay_half_life_sec``.
  * Browsing signals (contacts, repeated contacts, prolonged presence) are capped at ``browsing_cap`` which
    is below ``review_threshold``: browsing alone can never produce a review or an alert.
  * Levels: REVIEW (visual) at ``review_threshold``; ALERT (alarm) at ``alert_threshold``.
  * One event per level per incident (latched). An incident closes only when the score decays below
    ``reset_threshold``; only then can a new incident for the same person fire again.

All timing uses the FRAME timestamp passed to ``analyze`` (live clock or video time).
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Sequence, Tuple
import time

import numpy as np

from pharmaguard.config import BehaviorConfig, ShelfZone
from pharmaguard.interaction import HandObservation, PersonObservation, observe_person, zone_polygons_px
from pharmaguard.tracker import TrackingResult


class BehaviorEventType(str, Enum):
    SHELF_CONTACT = "SHELF_CONTACT"
    RETURNED_TO_SHELF = "RETURNED_TO_SHELF"        # pocket-like movement cancelled by a new shelf contact
    HAND_TO_POCKET = "HAND_TO_POCKET"
    HAND_TO_BAG = "HAND_TO_BAG"
    BAG_REACH = "BAG_REACH"                        # bag touched without a preceding shelf contact
    PROLONGED_PRESENCE = "PROLONGED_PRESENCE"
    REPEATED_CONTACTS = "REPEATED_CONTACTS"
    LEFT_AFTER_CONCEALMENT = "LEFT_AFTER_CONCEALMENT"
    SUSPICION_REVIEW = "SUSPICION_REVIEW"          # WARNING: staff should review
    SUSPICION_ALERT = "SUSPICION_ALERT"            # ALERT: alarm
    INCIDENT_CLOSED = "INCIDENT_CLOSED"


LEVEL_NONE, LEVEL_REVIEW, LEVEL_ALERT = 0, 1, 2
LEVEL_NAMES = {LEVEL_NONE: "NORMAL", LEVEL_REVIEW: "REVIEW", LEVEL_ALERT: "ALERT"}


@dataclass
class BehaviorEvent:
    """A detected interaction or an alert candidate."""
    event_type: BehaviorEventType
    tracker_id: int
    zone_name: str
    timestamp: float = field(default_factory=time.monotonic)
    dwell_time_seconds: float = 0.0
    velocity: float = 0.0
    description: str = ""
    severity: str = "INFO"  # "INFO", "WARNING", "ALERT"
    score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    incident_id: int = 0


@dataclass
class Evidence:
    t: float
    kind: str
    weight: float
    browsing: bool
    reason: str
    zone: str
    side: str = ""


@dataclass
class HandState:
    candidate_zone: Optional[str] = None
    candidate_since: Optional[float] = None
    contact_zone: Optional[str] = None
    outside_since: Optional[float] = None
    last_contact_end: Optional[float] = None
    last_contact_zone: Optional[str] = None
    carry_consumed: bool = True
    pocket_since: Optional[float] = None
    bag_since: Optional[float] = None
    bag_reach_reported: bool = False


@dataclass
class PersonState:
    hands: Dict[str, HandState] = field(default_factory=lambda: {"L": HandState(), "R": HandState()})
    evidence: List[Evidence] = field(default_factory=list)
    contact_times: Deque[float] = field(default_factory=deque)
    repeated_awarded_at: float = -1e9
    near_since: Dict[str, float] = field(default_factory=dict)
    presence_awarded_at: Dict[str, float] = field(default_factory=dict)
    last_concealment_t: Optional[float] = None
    last_concealment_zone: str = ""
    exit_awarded: bool = True
    away_since: Optional[float] = None
    level: int = LEVEL_NONE
    incident_id: int = 0
    last_seen: float = 0.0
    last_obs: Optional[PersonObservation] = None


@dataclass
class PersonView:
    """Read-only per-person summary for annotation and the dashboard."""
    score: float
    level: int
    hands: List[HandObservation]
    observation: Optional[PersonObservation]

    @property
    def level_name(self) -> str:
        return LEVEL_NAMES[self.level]


class PharmacyBehaviorAnalyzer:
    """Temporal interaction analysis and suspicion scoring over tracked people."""

    def __init__(self, zones: List[ShelfZone], config: Optional[BehaviorConfig] = None):
        self.zones = zones
        self.config = config or BehaviorConfig()
        self.people: Dict[int, PersonState] = {}
        self._incident_counter = 0
        self._zone_activity: Dict[str, str] = {}
        self._now = 0.0

    # ------------------------------------------------------------------ scoring
    def _decay(self, age: float) -> float:
        return 0.5 ** (max(age, 0.0) / self.config.decay_half_life_sec)

    def score(self, tid: int, now: Optional[float] = None) -> float:
        st = self.people.get(tid)
        if st is None:
            return 0.0
        now = self._now if now is None else now
        browsing = sum(e.weight * self._decay(now - e.t) for e in st.evidence if e.browsing)
        conceal = sum(e.weight * self._decay(now - e.t) for e in st.evidence if not e.browsing)
        return min(browsing, self.config.browsing_cap) + conceal

    def concealment_score(self, tid: int, now: Optional[float] = None) -> float:
        """Score from concealment-type evidence only (transfers to bag/pocket and subsequent exit)."""
        st = self.people.get(tid)
        if st is None:
            return 0.0
        now = self._now if now is None else now
        return sum(e.weight * self._decay(now - e.t) for e in st.evidence if not e.browsing)

    def _reasons(self, st: PersonState, now: float) -> List[str]:
        return [f"{e.reason} (t-{now - e.t:.0f}s)" for e in st.evidence if self._decay(now - e.t) > 0.1]

    def _add(self, st: PersonState, now: float, kind: str, weight: float, browsing: bool,
             reason: str, zone: str, side: str = "") -> None:
        st.evidence.append(Evidence(now, kind, weight, browsing, reason, zone, side))

    # ------------------------------------------------------------------ views
    def person_view(self, tid: int) -> Optional[PersonView]:
        st = self.people.get(tid)
        if st is None:
            return None
        return PersonView(score=round(self.score(tid), 2), level=st.level,
                          hands=st.last_obs.hands if st.last_obs else [], observation=st.last_obs)

    def needs_bag_context(self, now: Optional[float] = None) -> bool:
        """True while bag positions can change the score: a hand is entering / touching a shelf, or has left
        one within the carry window. Lets the pipeline skip the bag detector the rest of the time."""
        now = self._now if now is None else now
        cfg = self.config
        horizon = cfg.carry_window_sec + max(cfg.pocket_hold_sec, cfg.bag_hold_sec)
        for st in self.people.values():
            for hs in st.hands.values():
                if hs.contact_zone or hs.candidate_zone:
                    return True
                if not hs.carry_consumed and hs.last_contact_end is not None and now - hs.last_contact_end <= horizon:
                    return True
        return False

    def zone_activity(self) -> Dict[str, str]:
        """Per zone: 'CONTACT' (a hand is in it), 'PRESENCE' (someone at it) or 'IDLE'."""
        return dict(self._zone_activity)

    def reset(self) -> None:
        self.people.clear()
        self._zone_activity = {}

    # ------------------------------------------------------------------ main
    def analyze(self, tracking_result: TrackingResult, frame_width: int, frame_height: int,
                timestamp: Optional[float] = None,
                bag_boxes: Optional[Sequence[np.ndarray]] = None) -> List[BehaviorEvent]:
        now = time.monotonic() if timestamp is None else float(timestamp)
        self._now = now
        cfg = self.config
        events: List[BehaviorEvent] = []
        polys = zone_polygons_px(self.zones, frame_width, frame_height)
        zones_by_name = {z.name: z for z in self.zones}
        activity = {z.name: "IDLE" for z in self.zones}

        for tid, person in tracking_result.tracked_persons.items():
            st = self.people.setdefault(tid, PersonState())
            if person.frames_lost > 0:
                # Unseen: hands cannot be observed; drop in-progress holds, keep contact history.
                for hs in st.hands.values():
                    hs.pocket_since = hs.bag_since = None
                    hs.candidate_zone = hs.candidate_since = None
                self._check_away(st, tid, now, near=False, events=events)
                self._update_level(st, tid, now, events)
                continue

            st.last_seen = now
            obs = observe_person(person, polys, bag_boxes or [], cfg)
            st.last_obs = obs
            for z in obs.near_zones:
                if activity[z] == "IDLE":
                    activity[z] = "PRESENCE"

            for hand in obs.hands:
                self._update_hand(st, st.hands[hand.side], hand, tid, now, events)
                if hand.shelf_zone:
                    activity[hand.shelf_zone] = "CONTACT"
            for hs in st.hands.values():
                if hs.contact_zone:
                    activity[hs.contact_zone] = "CONTACT"

            self._update_presence(st, tid, obs, zones_by_name, now, events)
            self._check_away(st, tid, now, near=bool(obs.near_zones), events=events)
            self._update_level(st, tid, now, events)
            person.in_shelf_zone = any(h.contact_zone for h in st.hands.values())

        # Removed tracks: they have left the camera view.
        for tid in tracking_result.lost_ids:
            st = self.people.get(tid)
            if st is None:
                continue
            self._award_exit(st, tid, now, "left the camera view", events)
            self._update_level(st, tid, now, events)
            del self.people[tid]

        # Forget stale evidence.
        for st in self.people.values():
            st.evidence = [e for e in st.evidence if self._decay(now - e.t) > 0.01]

        self._zone_activity = activity
        return events

    # ------------------------------------------------------------------ hands
    def _update_hand(self, st: PersonState, hs: HandState, hand: HandObservation, tid: int,
                     now: float, events: List[BehaviorEvent]) -> None:
        cfg = self.config
        zone = hand.shelf_zone

        # --- shelf contact state machine (debounced both ways)
        if hs.contact_zone is None:
            if zone is not None:
                if hs.candidate_zone != zone:
                    hs.candidate_zone, hs.candidate_since = zone, now
                if now - hs.candidate_since >= cfg.contact_debounce_sec:
                    hs.contact_zone, hs.outside_since = zone, None
                    hs.candidate_zone = hs.candidate_since = None
                    self._on_contact(st, tid, hand.side, zone, now, events)
            else:
                hs.candidate_zone = hs.candidate_since = None
        else:
            if zone == hs.contact_zone:
                hs.outside_since = None
            else:
                if hs.outside_since is None:
                    hs.outside_since = now
                if now - hs.outside_since >= cfg.contact_debounce_sec:
                    hs.last_contact_end = hs.outside_since
                    hs.last_contact_zone = hs.contact_zone
                    hs.carry_consumed = False
                    hs.contact_zone = None
                    hs.outside_since = None

        # --- transfer from shelf to bag / pocket
        carrying = (not hs.carry_consumed and hs.last_contact_end is not None
                    and hs.contact_zone is None)

        if hand.in_bag:
            if hs.bag_since is None:
                hs.bag_since = now
            if now - hs.bag_since >= cfg.bag_hold_sec:
                if carrying and hs.bag_since - hs.last_contact_end <= cfg.carry_window_sec:
                    gap = max(0.0, hs.bag_since - hs.last_contact_end)
                    self._on_concealment(st, tid, hand.side, "bag", hs.last_contact_zone, gap, now, events)
                    hs.carry_consumed = True
                    hs.bag_reach_reported = True   # same hand-in-bag episode: not a separate bag reach
                elif not hs.bag_reach_reported and hs.contact_zone is None:
                    hs.bag_reach_reported = True
                    events.append(BehaviorEvent(
                        BehaviorEventType.BAG_REACH, tid, hs.last_contact_zone or "", now,
                        description=f"Person #{tid} {hand.side} hand at bag (no preceding shelf contact).",
                    ))
        else:
            hs.bag_since = None
            hs.bag_reach_reported = False

        if hand.in_pocket and not hand.in_bag:
            if hs.pocket_since is None:
                hs.pocket_since = now
            if (carrying and now - hs.pocket_since >= cfg.pocket_hold_sec
                    and hs.pocket_since - hs.last_contact_end <= cfg.carry_window_sec):
                gap = max(0.0, hs.pocket_since - hs.last_contact_end)
                self._on_concealment(st, tid, hand.side, "pocket", hs.last_contact_zone, gap, now, events)
                hs.carry_consumed = True
        else:
            hs.pocket_since = None

        if carrying and now - hs.last_contact_end > cfg.carry_window_sec + max(cfg.pocket_hold_sec, cfg.bag_hold_sec):
            hs.carry_consumed = True   # carry window expired without a transfer: ordinary handling

    def _on_contact(self, st: PersonState, tid: int, side: str, zone: str, now: float,
                    events: List[BehaviorEvent]) -> None:
        cfg = self.config
        # Same hand back at the shelf soon after a pocket-like movement: item examined and put back.
        cancelled = [e for e in st.evidence if e.kind == "pocket" and e.side == side
                     and now - e.t <= cfg.return_cancels_pocket_sec]
        if cancelled:
            st.evidence = [e for e in st.evidence
                           if e not in cancelled and not (e.kind == "exit" and e.t >= cancelled[0].t)]
            if not any(e.kind in ("pocket", "bag") for e in st.evidence):
                st.last_concealment_t, st.exit_awarded = None, True
            events.append(BehaviorEvent(BehaviorEventType.RETURNED_TO_SHELF, tid, zone, now,
                                        description=f"Person #{tid} {side} hand returned to {zone}: "
                                                    f"earlier waist/pocket movement discounted."))
        self._add(st, now, "contact", cfg.w_contact, True, f"{side} hand in {zone}", zone, side)
        events.append(BehaviorEvent(BehaviorEventType.SHELF_CONTACT, tid, zone, now,
                                    description=f"Person #{tid} {side} hand reached into {zone}.",
                                    score=self.score(tid, now)))
        st.contact_times.append(now)
        while st.contact_times and now - st.contact_times[0] > cfg.repeated_contacts_window_sec:
            st.contact_times.popleft()
        if (len(st.contact_times) >= cfg.repeated_contacts
                and now - st.repeated_awarded_at > cfg.repeated_contacts_window_sec):
            st.repeated_awarded_at = now
            n = len(st.contact_times)
            self._add(st, now, "repeated", cfg.w_repeated_contacts, True,
                      f"{n} shelf contacts in {cfg.repeated_contacts_window_sec:.0f}s", zone)
            events.append(BehaviorEvent(BehaviorEventType.REPEATED_CONTACTS, tid, zone, now,
                                        description=f"Person #{tid}: {n} shelf contacts within "
                                                    f"{cfg.repeated_contacts_window_sec:.0f}s."))

    def _on_concealment(self, st: PersonState, tid: int, side: str, target: str, zone: Optional[str],
                        gap: float, now: float, events: List[BehaviorEvent]) -> None:
        cfg = self.config
        zone = zone or ""
        if target == "bag":
            etype, weight, what = BehaviorEventType.HAND_TO_BAG, cfg.w_hand_to_bag, "a bag"
        else:
            etype, weight, what = BehaviorEventType.HAND_TO_POCKET, cfg.w_hand_to_pocket, "waist/pocket area"
        reason = f"{side} hand moved from {zone} to {what} {gap:.1f}s after shelf contact"
        self._add(st, now, target, weight, False, reason, zone, side)
        st.last_concealment_t, st.last_concealment_zone = now, zone
        st.exit_awarded = False
        st.away_since = None
        events.append(BehaviorEvent(etype, tid, zone, now, description=f"Person #{tid}: {reason}.",
                                    score=self.score(tid, now)))

    # ------------------------------------------------------------------ presence / exit
    def _update_presence(self, st: PersonState, tid: int, obs: PersonObservation,
                         zones_by_name: Dict[str, ShelfZone], now: float, events: List[BehaviorEvent]) -> None:
        for name in list(st.near_since):
            if name not in obs.near_zones:
                del st.near_since[name]
        for name in obs.near_zones:
            since = st.near_since.setdefault(name, now)
            limit = zones_by_name[name].prolonged_presence_sec
            last = st.presence_awarded_at.get(name, -1e9)
            if now - since >= limit and now - last >= limit:
                st.presence_awarded_at[name] = now
                self._add(st, now, "presence", self.config.w_prolonged_presence, True,
                          f"at {name} for {now - since:.0f}s", name)
                events.append(BehaviorEvent(BehaviorEventType.PROLONGED_PRESENCE, tid, name, now,
                                            dwell_time_seconds=round(now - since, 1),
                                            description=f"Person #{tid} at {name} for {now - since:.0f}s."))

    def _check_away(self, st: PersonState, tid: int, now: float, near: bool,
                    events: List[BehaviorEvent]) -> None:
        if st.exit_awarded or st.last_concealment_t is None:
            return
        if near:
            st.away_since = None
            return
        if st.away_since is None:
            st.away_since = now
        if now - st.away_since >= 1.5:
            self._award_exit(st, tid, now, "moved away from the shelf", events)

    def _award_exit(self, st: PersonState, tid: int, now: float, how: str, events: List[BehaviorEvent]) -> None:
        if st.exit_awarded or st.last_concealment_t is None:
            return
        since = (st.away_since or st.last_seen or now) - st.last_concealment_t
        if since > self.config.exit_window_sec:
            st.exit_awarded = True
            return
        st.exit_awarded = True
        reason = f"{how} {max(0.0, since):.1f}s after concealment-like movement"
        self._add(st, now, "exit", self.config.w_exit_after_concealment, False, reason, st.last_concealment_zone)
        events.append(BehaviorEvent(BehaviorEventType.LEFT_AFTER_CONCEALMENT, tid, st.last_concealment_zone, now,
                                    description=f"Person #{tid} {reason}.", score=self.score(tid, now)))

    # ------------------------------------------------------------------ levels / incidents
    def _update_level(self, st: PersonState, tid: int, now: float, events: List[BehaviorEvent]) -> None:
        cfg = self.config
        s = self.score(tid, now)
        zone = st.last_concealment_zone or next((e.zone for e in reversed(st.evidence)), "")
        if st.level == LEVEL_NONE and s >= cfg.review_threshold:
            self._incident_counter += 1
            st.incident_id = self._incident_counter
        if st.level < LEVEL_ALERT and s >= cfg.alert_threshold:
            st.level = LEVEL_ALERT
            events.append(self._level_event(BehaviorEventType.SUSPICION_ALERT, "ALERT", st, tid, zone, s, now))
        elif st.level < LEVEL_REVIEW and s >= cfg.review_threshold:
            st.level = LEVEL_REVIEW
            events.append(self._level_event(BehaviorEventType.SUSPICION_REVIEW, "WARNING", st, tid, zone, s, now))
        elif st.level > LEVEL_NONE and self.concealment_score(tid, now) < cfg.reset_threshold:
            # Close on concealment evidence only: continued browsing must not keep an incident latched
            # (which would silently swallow a genuinely new concealment minutes later).
            events.append(BehaviorEvent(BehaviorEventType.INCIDENT_CLOSED, tid, zone, now, score=round(s, 2),
                                        incident_id=st.incident_id,
                                        description=f"Person #{tid}: incident {st.incident_id} closed "
                                                    f"(score decayed to {s:.2f})."))
            st.level = LEVEL_NONE

    def _level_event(self, etype: BehaviorEventType, severity: str, st: PersonState, tid: int,
                     zone: str, s: float, now: float) -> BehaviorEvent:
        reasons = self._reasons(st, now)
        label = "Suspicious interaction sequence" if severity == "ALERT" else "Interaction worth reviewing"
        return BehaviorEvent(
            etype, tid, zone, now, severity=severity, score=round(s, 2), reasons=reasons,
            incident_id=st.incident_id,
            description=f"{label}: Person #{tid}, score {s:.2f}. " + "; ".join(reasons),
        )
