"""
Ground-truth tracking scenarios (synthetic detections with known identities).

Each scenario yields frames of (timestamp, [(gt_person, box_xyxy, confidence), ...]).
``evaluate`` runs a tracker and reports, for every ground-truth person, the set of tracker IDs it was
given, which exposes ID switches (a person with >1 ID) and swaps (two people sharing an ID).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from pharmaguard.detector import DetectionResult
from pharmaguard.tracker import box_iou_matrix

Frame = Tuple[float, List[Tuple[str, Tuple[float, float, float, float], float]]]


def det(entries: List[Tuple[str, Tuple[float, float, float, float], float]]) -> DetectionResult:
    if not entries:
        return DetectionResult()
    return DetectionResult(
        xyxy=np.array([e[1] for e in entries], np.float32),
        confidence=np.array([e[2] for e in entries], np.float32),
        class_id=np.zeros(len(entries), np.int32),
        class_name=["person"] * len(entries),
    )


@dataclass
class ScenarioResult:
    ids_per_person: Dict[str, Set[int]] = field(default_factory=dict)
    frames_visible: Dict[str, int] = field(default_factory=dict)     # GT frames where person was detected
    frames_with_id: Dict[str, int] = field(default_factory=dict)     # ... and carried a visible track ID
    all_ids: Set[int] = field(default_factory=set)
    removed_ids: List[int] = field(default_factory=list)

    def id_switches(self) -> int:
        return sum(max(0, len(v) - 1) for v in self.ids_per_person.values())

    def shared_ids(self) -> Set[int]:
        seen: Dict[int, str] = {}
        shared: Set[int] = set()
        for person, ids in self.ids_per_person.items():
            for i in ids:
                if i in seen and seen[i] != person:
                    shared.add(i)
                seen[i] = person
        return shared


def evaluate(update: Callable[[DetectionResult, float], object], frames: Iterable[Frame]) -> ScenarioResult:
    """``update(detections, timestamp) -> TrackingResult``."""
    res = ScenarioResult()
    for ts, entries in frames:
        tr = update(det(entries), ts)
        res.removed_ids.extend(tr.lost_ids)
        visible = {tid: p for tid, p in tr.tracked_persons.items() if p.frames_lost == 0}
        res.all_ids.update(tr.tracked_persons.keys())
        if not entries:
            continue
        gt_boxes = np.array([e[1] for e in entries], np.float32)
        tids = list(visible)
        tboxes = np.array([visible[t].current_box for t in tids], np.float32) if tids else np.empty((0, 4), np.float32)
        iou = box_iou_matrix(gt_boxes, tboxes)
        for gi, (person, _, _) in enumerate(entries):
            res.frames_visible[person] = res.frames_visible.get(person, 0) + 1
            res.ids_per_person.setdefault(person, set())
            if iou.shape[1] and iou[gi].max() > 0.5:
                res.ids_per_person[person].add(tids[int(iou[gi].argmax())])
                res.frames_with_id[person] = res.frames_with_id.get(person, 0) + 1
    return res


# --------------------------------------------------------------------------- scenario builders

def walker(x0: float, y0: float, vx: float, vy: float, w: float = 120, h: float = 300):
    """Box generator: position at time t (seconds), velocity in px/s."""
    def box(t: float) -> Tuple[float, float, float, float]:
        x, y = x0 + vx * t, y0 + vy * t
        return (x, y, x + w, y + h)
    return box


def walk_speed(px_per_frame: float, fps: float = 10.0, n: int = 25) -> List[Frame]:
    b = walker(20, 100, px_per_frame * fps, 0)
    return [(i / fps, [("A", b(i / fps), 0.8)]) for i in range(n)]


def walk_body_heights(hps: float, fps: float, seconds: float = 4.0, h: float = 300.0) -> List[Frame]:
    """Walker moving ``hps`` body-heights per second (normal walk ~0.8, brisk ~1.2)."""
    b = walker(20, 100, hps * h, 0, w=0.4 * h, h=h)
    return [(i / fps, [("A", b(i / fps), 0.8)]) for i in range(int(seconds * fps))]


def parallel_walkers(fps: float, gap_px: float = 15.0, hps: float = 1.0) -> List[Frame]:
    """Two people walking side by side in the same direction, boxes nearly touching."""
    h, w = 300.0, 120.0
    a = walker(20, 100, hps * h, 0, w=w, h=h)
    b = walker(20 + w + gap_px, 110, hps * h, 0, w=w, h=h)
    return [(i / fps, [("A", a(i / fps), 0.8), ("B", b(i / fps), 0.8)]) for i in range(int(3 * fps))]


def crossing(fps: float = 10.0, occlusion: str = "low_conf") -> List[Frame]:
    """A walks right, B walks left on the same line; B is in front.
    While boxes overlap heavily, A is either detected with low confidence or not at all."""
    a = walker(40, 120, 150, 0, w=100, h=280)
    b = walker(840, 130, -150, 0, w=100, h=280)
    frames: List[Frame] = []
    for i in range(int(6 * fps)):
        t = i / fps
        ba, bb = a(t), b(t)
        overlap = box_iou_matrix(np.array([ba], np.float32), np.array([bb], np.float32))[0, 0]
        entries = [("B", bb, 0.85)]
        if overlap < 0.3:
            entries.append(("A", ba, 0.8))
        elif occlusion == "low_conf":
            entries.append(("A", ba, 0.25))
        # occlusion == "missing": A not detected at all while hidden
        frames.append((t, entries))
    return frames


def occlusion_gap(gap_seconds: float, fps: float = 10.0, moving: bool = True) -> List[Frame]:
    """Person visible 2 s, fully hidden for ``gap_seconds``, then visible 2 s again."""
    b = walker(50, 100, 60 if moving else 0, 0)
    frames: List[Frame] = []
    total = 4.0 + gap_seconds
    for i in range(int(total * fps)):
        t = i / fps
        hidden = 2.0 <= t < 2.0 + gap_seconds
        frames.append((t, [] if hidden else [("A", b(t), 0.8)]))
    return frames


def partial_occlusion_low_conf(fps: float = 10.0) -> List[Frame]:
    """Person's confidence drops to 0.2 for 1.5 s (e.g. half hidden behind a shelf)."""
    b = walker(100, 100, 40, 0)
    return [(i / fps, [("A", b(i / fps), 0.2 if 10 <= i < 25 else 0.8)]) for i in range(40)]


def crowd(fps: float = 10.0) -> List[Frame]:
    walkers = {
        "P1": walker(20, 20, 80, 10), "P2": walker(600, 40, -90, 5), "P3": walker(300, 250, 0, -40),
        "P4": walker(50, 350, 120, -20), "P5": walker(700, 300, -60, 0),
    }
    frames: List[Frame] = []
    for i in range(int(4 * fps)):
        t = i / fps
        frames.append((t, [(k, f(t), 0.75) for k, f in walkers.items()]))
    return frames


def jitter(n: int = 100, fps: float = 10.0, px: float = 6.0, seed: int = 0) -> List[Frame]:
    rng = np.random.default_rng(seed)
    base = np.array([200, 100, 320, 400], np.float32)
    return [(i / fps, [("A", tuple(base + rng.uniform(-px, px, 4)), float(rng.uniform(0.5, 0.9)))]) for i in range(n)]


def irregular_rate(seed: int = 1, speed_px_s: float = 150.0) -> List[Frame]:
    """Tunnel-like delivery: frame gaps uniformly 50-400 ms."""
    rng = np.random.default_rng(seed)
    b = walker(20, 100, speed_px_s, 0)
    t, frames = 0.0, []
    while t < 5.0:
        frames.append((t, [("A", b(t), 0.8)]))
        t += float(rng.uniform(0.05, 0.4))
    return frames


def single_frame_false_positive(fps: float = 10.0) -> List[Frame]:
    frames: List[Frame] = [(i / fps, []) for i in range(10)]
    frames[4] = (0.4, [("FP", (300, 100, 380, 300), 0.9)])
    return frames
