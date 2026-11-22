"""
Behaviour-engine tests mapped to the mandate's staged scenarios (§6).

Pass criteria: ordinary interactions (walk past, browse, pick up & return, reach own bag) produce NO
review/alert; shelf->bag->exit produces exactly ONE alert; shelf->pocket->exit produces a REVIEW only.
"""

import numpy as np
import pytest

from behavior_scenarios import Actor, FPS, H, W, run_actors, skeleton, person_box
from pharmaguard.behavior import BehaviorEventType as E, PharmacyBehaviorAnalyzer
from pharmaguard.config import BehaviorConfig, ShelfZone
from pharmaguard.interaction import observe_person, zone_polygons_px
from pharmaguard.tracker import TrackedPerson, TrackingResult

CFG = BehaviorConfig()


# --------------------------------------------------------------------------- geometry

def _obs(pose, cx=320.0, bag=False):
    kp = skeleton(cx, pose)
    p = TrackedPerson(tracker_id=1, current_box=person_box(kp), confidence=0.9, centroid=(0, 0), keypoints=kp)
    bags = [np.array([cx + 60, 250, cx + 150, 360], np.float32)] if bag else []
    return observe_person(p, zone_polygons_px([ShelfZone()], W, H), bags, CFG)


def test_geometry_reach_is_shelf_contact():
    r = {h.side: h for h in _obs("reach_R").hands}["R"]
    assert r.shelf_zone == "Prescription_Shelf_A" and not r.in_pocket


def test_geometry_hands_at_chest_are_not_shelf_contact_even_though_shelf_is_behind():
    for h in _obs("examine").hands:
        assert h.in_zone_raw == "Prescription_Shelf_A"   # polygon is behind the person...
        assert h.shelf_zone is None                      # ...but torso exclusion rejects it
        assert not h.in_pocket                           # and chest height is not a pocket


def test_geometry_resting_hands_are_neither_shelf_nor_pocket():
    for h in _obs("rest").hands:
        assert h.shelf_zone is None and not h.in_pocket and not h.in_bag


def test_geometry_pocket_and_bag():
    assert {h.side: h for h in _obs("pocket_R").hands}["R"].in_pocket
    r = {h.side: h for h in _obs("bag_R", bag=True).hands}["R"]
    assert r.in_bag and not r.in_pocket


def test_geometry_low_confidence_wrists_are_ignored():
    for h in _obs("hands_hidden").hands:
        assert not h.visible and h.shelf_zone is None


# --------------------------------------------------------------------------- mandate scenarios

def test_s3_walking_past_shelf_raises_nothing():
    run = run_actors([Actor(1, -60, [(4.0, "rest", 700)])])
    assert run.of(E.SHELF_CONTACT) == []
    assert run.severities() == []
    assert run.max_score.get(1, 0) < 0.1


def test_s4_browsing_with_repeated_contacts_stays_below_review():
    script = [(1.0, "rest", None)]
    for _ in range(5):
        script += [(1.5, "reach_R", None), (2.0, "examine", None), (1.0, "rest", None)]
    run = run_actors([Actor(1, 320, script + [(2.0, "rest", 700)])])
    assert len(run.of(E.SHELF_CONTACT)) == 5
    assert run.of(E.REPEATED_CONTACTS)
    assert run.severities() == []
    assert run.max_score[1] <= CFG.browsing_cap + 1e-6


def test_s4b_long_browse_prolonged_presence_capped():
    script = [(1.0, "rest", None)] + [(1.5, "reach_R", None), (3.0, "examine", None)] * 10
    run = run_actors([Actor(1, 320, script)])
    assert run.of(E.PROLONGED_PRESENCE)
    assert run.severities() == []


def test_s5_pick_up_examine_at_waist_and_return_raises_nothing():
    script = [(1.0, "rest", None), (1.5, "reach_R", None), (0.5, "examine", None),
              (2.0, "pocket_R", None),             # held low at the hip while reading the label
              (1.5, "reach_R", None),              # puts it back on the shelf
              (1.0, "rest", None), (3.0, "rest", 750)]
    run = run_actors([Actor(1, 320, script)])
    assert run.of(E.HAND_TO_POCKET)
    assert run.of(E.RETURNED_TO_SHELF)
    assert run.of(E.LEFT_AFTER_CONCEALMENT) == []
    assert run.severities() == []


def test_s6_reaching_into_own_bag_without_shelf_contact_raises_nothing():
    script = [(1.0, "rest", None), (2.0, "bag_R", None), (1.0, "rest", None), (3.0, "rest", 750)]
    run = run_actors([Actor(1, 320, script, bag=True)])
    assert run.of(E.BAG_REACH)
    assert run.of(E.HAND_TO_BAG) == []
    assert run.severities() == []


def test_s7_shelf_to_bag_then_leave_escalates_review_then_exactly_one_alert():
    script = [(1.0, "rest", None), (1.5, "reach_R", None), (0.3, "rest", None), (1.2, "bag_R", None),
              (1.0, "rest", None), (3.0, "rest", 750)]
    run = run_actors([Actor(1, 320, script, bag=True)])
    assert len(run.of(E.HAND_TO_BAG)) == 1
    assert run.of(E.BAG_REACH) == []          # the transfer is not also reported as a plain bag reach
    assert len(run.of(E.LEFT_AFTER_CONCEALMENT)) == 1
    alerts = run.of(E.SUSPICION_ALERT)
    assert len(alerts) == 1
    a = alerts[0]
    assert a.severity == "ALERT" and a.score >= CFG.alert_threshold
    assert any("bag" in r for r in a.reasons)
    assert any("not returned" in r or "left view" in r or "after concealment" in r for r in a.reasons)
    # Escalation within ONE incident: visual REVIEW at the transfer, ALERT once the item is kept.
    assert run.severities() == ["WARNING", "ALERT"]
    review = run.of(E.SUSPICION_REVIEW)[0]
    assert review.incident_id == a.incident_id and review.timestamp < a.timestamp


def test_s7b_shelf_to_pocket_then_leave_raises_one_alert():
    script = [(1.0, "rest", None), (1.5, "reach_R", None), (0.3, "rest", None), (1.5, "pocket_R", None),
              (1.0, "rest", None), (3.0, "rest", 750)]
    run = run_actors([Actor(1, 320, script)])
    assert len(run.of(E.SUSPICION_ALERT)) == 1
    assert len(run.of(E.ITEM_NOT_RETURNED)) == 1


@pytest.mark.parametrize("target,bag", [("pocket_R", False), ("bag_R", True)])
def test_shelf_to_pocket_or_bag_and_staying_put_raises_alarm(target, bag):
    """Realistic trigger: take an item and put it away; no need to walk off. Alarm after keep_confirm_sec."""
    script = [(1.0, "rest", None), (1.5, "reach_R", None), (0.3, "rest", None), (1.2, target, None),
              (8.0, "rest", None)]
    run = run_actors([Actor(1, 320, script, bag=bag, leave_at_end=False)])
    alerts = run.of(E.SUSPICION_ALERT)
    assert len(alerts) == 1, run.severities()
    assert run.of(E.LEFT_AFTER_CONCEALMENT) == []
    kept = run.of(E.ITEM_NOT_RETURNED)[0]
    conceal = (run.of(E.HAND_TO_POCKET) + run.of(E.HAND_TO_BAG))[0]
    assert kept.timestamp - conceal.timestamp == pytest.approx(CFG.keep_confirm_sec, abs=0.2)
    assert alerts[0].timestamp >= kept.timestamp


@pytest.mark.parametrize("target,bag", [("pocket_R", False), ("bag_R", True)])
def test_item_put_back_within_keep_window_raises_no_alarm(target, bag):
    script = [(1.0, "rest", None), (1.5, "reach_R", None), (0.3, "rest", None), (1.0, target, None),
              (1.5, "reach_R", None), (8.0, "rest", None)]
    run = run_actors([Actor(1, 320, script, bag=bag, leave_at_end=False)])
    assert run.of(E.SUSPICION_ALERT) == []
    assert run.of(E.ITEM_NOT_RETURNED) == []
    assert run.of(E.RETURNED_TO_SHELF)


def test_slow_transfer_outside_carry_window_is_not_concealment():
    script = [(1.0, "rest", None), (1.5, "reach_R", None), (5.0, "examine", None), (2.0, "bag_R", None),
              (3.0, "rest", 750)]
    run = run_actors([Actor(1, 320, script, bag=True)])
    assert run.of(E.HAND_TO_BAG) == []
    assert run.severities() == []


def test_one_incident_one_alert_then_new_incident_after_decay():
    conceal = [(1.5, "reach_R", None), (0.3, "rest", None), (1.0, "bag_R", None), (0.5, "rest", None)]
    # Conceal, move away (exit evidence) -> ALERT; conceal twice more while score is high -> no new alert.
    script = [(1.0, "rest", None)] + conceal + [(2.5, "rest", 600), (1.0, "rest", 320)] + conceal * 2
    script += [(120.0, "rest", None)]                      # long calm period -> decays, incident closes
    script += conceal + [(4.0, "rest", 720)]               # new incident, then walks off
    run = run_actors([Actor(1, 320, script, bag=True, leave_at_end=False)], tail_seconds=3.0)
    alerts = run.of(E.SUSPICION_ALERT)
    assert len(alerts) == 2, [(round(e.timestamp, 1), e.score) for e in alerts]
    assert alerts[0].incident_id != alerts[1].incident_id
    assert len(run.of(E.INCIDENT_CLOSED)) >= 1


def test_hidden_hands_produce_no_contacts():
    run = run_actors([Actor(1, 320, [(10.0, "hands_hidden", None)])])
    assert run.of(E.SHELF_CONTACT) == [] and run.severities() == []


def test_two_people_only_the_concealing_one_is_flagged():
    browser = Actor(1, 160, [(1.0, "rest", None)] + [(1.5, "reach_R", None), (1.5, "examine", None)] * 3
                    + [(3.0, "rest", -100)])
    concealer = Actor(2, 400, [(1.0, "rest", None), (1.5, "reach_R", None), (0.3, "rest", None),
                               (1.2, "bag_R", None), (1.0, "rest", None), (3.0, "rest", 800)], bag=True)
    run = run_actors([browser, concealer])
    assert run.severities(tid=1) == []
    assert run.severities(tid=2) == ["WARNING", "ALERT"]


def test_zone_activity_is_per_zone():
    zones = [ShelfZone(name="Left", polygon_normalized=[[0, .1], [.45, .1], [.45, .55], [0, .55]]),
             ShelfZone(name="Right", polygon_normalized=[[.55, .1], [1, .1], [1, .55], [.55, .55]])]
    analyzer = PharmacyBehaviorAnalyzer(zones, CFG)
    kp = skeleton(320, "reach_R")          # right wrist at x=440 -> Right zone
    p = TrackedPerson(tracker_id=1, current_box=person_box(kp), confidence=0.9, centroid=(0, 0), keypoints=kp)
    for i in range(5):
        analyzer.analyze(TrackingResult({1: p}, [1], [], []), W, H, timestamp=i / FPS)
    act = analyzer.zone_activity()
    assert act["Right"] == "CONTACT"
    assert act["Left"] in ("PRESENCE", "IDLE")


def test_no_pose_model_degrades_safely():
    analyzer = PharmacyBehaviorAnalyzer([ShelfZone()], CFG)
    p = TrackedPerson(tracker_id=1, current_box=np.array([260, 80, 380, 460], np.float32),
                      confidence=0.9, centroid=(0, 0), keypoints=None)
    events = []
    for i in range(300):
        events += analyzer.analyze(TrackingResult({1: p}, [1], [], []), W, H, timestamp=i / FPS)
    assert all(e.severity == "INFO" for e in events)
