"""
Unit tests for PharmaGuard behavioral interaction analysis.
"""

import time
import numpy as np
import pytest
from pharmaguard.config import ShelfZone
from pharmaguard.tracker import TrackedPerson, TrackingResult
from pharmaguard.behavior import PharmacyBehaviorAnalyzer, BehaviorEventType


def test_behavior_analyzer_zone_entry_and_exit():
    # Zone spanning 10% to 50% horizontally, 20% to 80% vertically
    zone = ShelfZone(
        name="Shelf_Narcotics",
        polygon_normalized=[[0.1, 0.2], [0.5, 0.2], [0.5, 0.8], [0.1, 0.8]],
        dwell_alert_threshold_sec=5.0,
        rapid_retreat_threshold_sec=2.0
    )
    analyzer = PharmacyBehaviorAnalyzer([zone])
    w, h = 1000, 1000

    # Step 1: Person outside zone (e.g. at x=700, y=700)
    person1 = TrackedPerson(
        tracker_id=1,
        current_box=np.array([650, 600, 750, 800], dtype=np.float32),
        confidence=0.9,
        centroid=(700, 700)
    )
    res_outside = TrackingResult(
        tracked_persons={1: person1},
        active_ids=[1],
        new_ids=[],
        lost_ids=[]
    )
    events = analyzer.analyze(res_outside, frame_width=w, frame_height=h)
    assert len(events) == 0
    assert not person1.in_shelf_zone

    # Step 2: Person enters zone (e.g. feet at x=300, y=500)
    person1.current_box = np.array([250, 400, 350, 500], dtype=np.float32)
    person1.centroid = (300, 450)
    res_inside = TrackingResult(
        tracked_persons={1: person1},
        active_ids=[1],
        new_ids=[],
        lost_ids=[]
    )
    events = analyzer.analyze(res_inside, frame_width=w, frame_height=h)
    assert len(events) == 1
    assert events[0].event_type == BehaviorEventType.ZONE_ENTER
    assert events[0].tracker_id == 1
    assert events[0].zone_name == "Shelf_Narcotics"

    # Step 3: Person stays inside and exceeds dwell alert threshold
    # Manually adjust entry time to simulate 6 seconds passing
    key = (1, "Shelf_Narcotics")
    analyzer.zone_entry_times[key] = time.time() - 6.0
    events_dwell = analyzer.analyze(res_inside, frame_width=w, frame_height=h)
    assert any(e.event_type == BehaviorEventType.SUSPICIOUS_LOITERING for e in events_dwell)

    # Step 4: Person exits zone
    person1.current_box = np.array([650, 600, 750, 800], dtype=np.float32)
    events_exit = analyzer.analyze(res_outside, frame_width=w, frame_height=h)
    assert any(e.event_type in (BehaviorEventType.ZONE_EXIT, BehaviorEventType.RAPID_RETREAT) for e in events_exit)
