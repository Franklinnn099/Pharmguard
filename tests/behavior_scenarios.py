"""
Scripted skeleton actors for behaviour-engine tests (webcam facing a mock shelf, 640x480 frame).

Geometry (person centred at cx, standing in front of the shelf; default shelf zone x 96-544, y 48-264):
  shoulders (cx±35, 150), hips (cx±25, 290)  -> torso length 140 px
  rest      : wrists hanging at mid-thigh (cx±45, 355)            - not shelf, not pocket
  reach_R   : right arm extended into shelf (cx+120, 180)         - shelf contact (outside torso box)
  examine   : both hands in front of chest (cx±10, 225)           - inside shelf polygon BUT inside torso box
  pocket_R  : right wrist at right hip (cx+22, 298)               - pocket region
  bag_R     : right wrist inside the bag carried at right side    - bag
A bag (if carried) hangs at (cx+60..cx+150, 250..360).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from pharmaguard.behavior import BehaviorEvent, BehaviorEventType, PharmacyBehaviorAnalyzer
from pharmaguard.config import BehaviorConfig, ShelfZone
from pharmaguard.tracker import TrackedPerson, TrackingResult

W, H = 640, 480
FPS = 10.0

POSES: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {
    # (left wrist offset, right wrist offset) relative to (cx, 0)
    "rest": ((-45, 355), (45, 355)),
    "reach_R": ((-45, 355), (120, 180)),
    "examine": ((-10, 225), (10, 225)),
    "pocket_R": ((-45, 355), (22, 298)),
    "bag_R": ((-45, 355), (105, 310)),
    "hands_hidden": ((-45, 355), (45, 355)),
    # Realistic relaxed hanging arm: wrist only ~0.13 torso below the hip (inside the pocket band), arm straight.
    "hang_R": ((-45, 355), (45, 308)),
    # Hand pushed into the right front pocket: wrist at the hip, elbow bent and flared out.
    "pocket_bent_R": ((-45, 355), (22, 298)),
}

# Optional elbow positions (left, right) per pose; poses without an entry have no visible elbows.
ELBOWS: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {
    "hang_R": ((-42, 245), (42, 245)),
    "pocket_bent_R": ((-42, 245), (60, 240)),
}


def skeleton(cx: float, pose: str, wrist_conf: float = 0.9) -> np.ndarray:
    kp = np.zeros((17, 3), np.float32)
    kp[5] = (cx - 35, 150, 0.95)   # L shoulder
    kp[6] = (cx + 35, 150, 0.95)   # R shoulder
    kp[11] = (cx - 25, 290, 0.9)   # L hip
    kp[12] = (cx + 25, 290, 0.9)   # R hip
    (lx, ly), (rx, ry) = POSES[pose]
    c = 0.05 if pose == "hands_hidden" else wrist_conf
    kp[9] = (cx + lx, ly, c)
    kp[10] = (cx + rx, ry, c)
    if pose in ELBOWS:
        (elx, ely), (erx, ery) = ELBOWS[pose]
        kp[7] = (cx + elx, ely, 0.9)
        kp[8] = (cx + erx, ery, 0.9)
    return kp


def person_box(kp: np.ndarray) -> np.ndarray:
    vis = kp[kp[:, 2] > 0.3][:, :2]
    x1 = min(vis[:, 0].min() - 15, kp[5, 0] - 25)
    x2 = max(vis[:, 0].max() + 15, kp[6, 0] + 25)
    return np.array([x1, 80, x2, 460], np.float32)


@dataclass
class Actor:
    tid: int
    cx: float
    script: List[Tuple[float, str, Optional[float]]]   # (duration s, pose, walk-to cx or None)
    bag: bool = False
    leave_at_end: bool = True

    def frames(self) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        """[(t, keypoints, box)] for the scripted performance."""
        out, t, cx = [], 0.0, self.cx
        for duration, pose, target in self.script:
            n = max(1, int(round(duration * FPS)))
            start = cx
            for i in range(n):
                if target is not None:
                    cx = start + (target - start) * (i + 1) / n
                kp = skeleton(cx, pose)
                out.append((t, kp, person_box(kp)))
                t += 1.0 / FPS
        self.cx = cx
        return out

    def bag_box(self, cx: float) -> np.ndarray:
        return np.array([cx + 60, 250, cx + 150, 360], np.float32)


@dataclass
class Run:
    events: List[BehaviorEvent] = field(default_factory=list)
    max_score: Dict[int, float] = field(default_factory=dict)
    analyzer: Optional[PharmacyBehaviorAnalyzer] = None

    def of(self, etype: BehaviorEventType, tid: Optional[int] = None) -> List[BehaviorEvent]:
        return [e for e in self.events if e.event_type == etype and (tid is None or e.tracker_id == tid)]

    def severities(self, tid: Optional[int] = None) -> List[str]:
        return [e.severity for e in self.events if e.severity != "INFO" and (tid is None or e.tracker_id == tid)]


def run_actors(actors: List[Actor], config: Optional[BehaviorConfig] = None,
               zones: Optional[List[ShelfZone]] = None, tail_seconds: float = 3.0) -> Run:
    """Drive the analyzer frame by frame. Actors who ``leave_at_end`` are removed (left view) afterwards."""
    analyzer = PharmacyBehaviorAnalyzer(zones or [ShelfZone()], config or BehaviorConfig())
    run = Run(analyzer=analyzer)
    tracks = {a.tid: (a, a.frames()) for a in actors}
    n = max(len(f) for _, f in tracks.values())
    persons: Dict[int, TrackedPerson] = {}
    t = 0.0
    for i in range(n):
        t = i / FPS
        bags, lost = [], []
        for tid, (actor, frames) in tracks.items():
            if i < len(frames):
                _, kp, box = frames[i]
                p = persons.get(tid) or TrackedPerson(tracker_id=tid, current_box=box, confidence=0.9,
                                                      centroid=(0, 0))
                p.current_box, p.keypoints, p.frames_lost = box, kp, 0
                persons[tid] = p
                if actor.bag:
                    bags.append(actor.bag_box(float((kp[5, 0] + kp[6, 0]) / 2)))
            elif tid in persons and actor.leave_at_end:
                lost.append(tid)
                del persons[tid]
        res = TrackingResult(tracked_persons=dict(persons), active_ids=list(persons), new_ids=[], lost_ids=lost)
        run.events += analyzer.analyze(res, W, H, timestamp=t, bag_boxes=bags)
        for tid in persons:
            run.max_score[tid] = max(run.max_score.get(tid, 0.0), analyzer.score(tid, t))
    # Remaining leavers, then let time pass.
    leavers = [tid for tid, (a, _) in tracks.items() if a.leave_at_end and tid in persons]
    for tid in leavers:
        del persons[tid]
    t += 1.0 / FPS
    run.events += analyzer.analyze(TrackingResult(dict(persons), list(persons), [], leavers), W, H, timestamp=t)
    for k in range(int(tail_seconds * FPS)):
        t += 1.0 / FPS
        run.events += analyzer.analyze(TrackingResult(dict(persons), list(persons), [], []), W, H, timestamp=t)
    return run
