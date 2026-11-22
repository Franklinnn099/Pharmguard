"""
Per-frame interaction geometry for PharmaGuard (stateless).

Turns a tracked person's pose keypoints into hand observations:
  * which shelf zone (if any) each wrist is inside,
  * whether a wrist is in the person's waist/pocket region (disc around a hip keypoint),
  * whether a wrist is inside a bag carried by / next to this person.

Temporal reasoning (debouncing, sequences, scoring) lives in ``behavior.py``.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from pharmaguard.config import BehaviorConfig, ShelfZone
from pharmaguard.detector import KP_L_HIP, KP_L_SHOULDER, KP_L_WRIST, KP_R_HIP, KP_R_SHOULDER, KP_R_WRIST
from pharmaguard.tracker import TrackedPerson

Point = Tuple[float, float]


@dataclass
class HandObservation:
    side: str                       # "L" or "R" (person's own left/right, COCO convention)
    point: Optional[Point]          # None if the wrist keypoint is not confidently visible
    confidence: float
    shelf_zone: Optional[str] = None
    in_zone_raw: Optional[str] = None   # zone containing the wrist before torso exclusion (diagnostics)
    in_pocket: bool = False
    in_bag: bool = False
    # Wrist position accepted for the pocket/bag test only (lower confidence: a hand going INTO a pocket
    # or bag is partly hidden, so the pose model's wrist confidence drops). None if below even that.
    reach_point: Optional[Point] = None

    @property
    def visible(self) -> bool:
        return self.point is not None


@dataclass
class PersonObservation:
    tracker_id: int
    box: np.ndarray
    has_pose: bool
    hands: List[HandObservation] = field(default_factory=list)
    near_zones: List[str] = field(default_factory=list)   # body overlaps these shelf zones
    pocket_centers: List[Point] = field(default_factory=list)
    pocket_radius: float = 0.0
    torso_box: Optional[np.ndarray] = None
    bag_boxes: List[np.ndarray] = field(default_factory=list)
    hips_visible: bool = False


def zone_polygons_px(zones: Sequence[ShelfZone], width: int, height: int) -> Dict[str, np.ndarray]:
    return {
        z.name: np.array([[p[0] * width, p[1] * height] for p in z.polygon_normalized], dtype=np.float32)
        for z in zones
    }


def point_in_polygon(pt: Point, poly: np.ndarray) -> bool:
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


def box_overlap_fraction_with_polygon(box: np.ndarray, poly: np.ndarray, samples: int = 7) -> float:
    """Fraction of a grid of points inside ``box`` that fall in ``poly`` (cheap area-overlap estimate)."""
    xs = np.linspace(box[0], box[2], samples)
    ys = np.linspace(box[1], box[3], samples)
    inside = sum(point_in_polygon((x, y), poly) for x in xs for y in ys)
    return inside / float(samples * samples)


def _kp(kps: np.ndarray, idx: int, min_conf: float) -> Optional[Point]:
    x, y, c = kps[idx]
    return (float(x), float(y)) if c >= min_conf else None


def _expand(box: np.ndarray, frac: float) -> np.ndarray:
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] - w * frac, box[1] - h * frac, box[2] + w * frac, box[3] + h * frac], np.float32)


def _in_box(pt: Point, box: np.ndarray) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def bags_for_person(person_box: np.ndarray, bag_boxes: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Bags overlapping the person's box (expanded 25%): carried or set down right beside them."""
    region = _expand(person_box, 0.25)
    out = []
    for b in bag_boxes:
        if b[0] < region[2] and b[2] > region[0] and b[1] < region[3] and b[3] > region[1]:
            out.append(b)
    return out


def observe_person(person: TrackedPerson, zone_polys: Dict[str, np.ndarray],
                   bag_boxes: Sequence[np.ndarray], cfg: BehaviorConfig,
                   near_zone_min_fraction: float = 0.15) -> PersonObservation:
    box = person.current_box
    obs = PersonObservation(tracker_id=person.tracker_id, box=box, has_pose=person.keypoints is not None)
    obs.near_zones = [name for name, poly in zone_polys.items()
                      if box_overlap_fraction_with_polygon(box, poly) >= near_zone_min_fraction]
    if person.keypoints is None:
        return obs

    kps = person.keypoints
    mc = cfg.keypoint_min_conf
    # Torso length for scale: shoulder midpoint to hip midpoint; fall back to box height.
    shoulders = [p for p in (_kp(kps, KP_L_SHOULDER, mc), _kp(kps, KP_R_SHOULDER, mc)) if p]
    hips = [p for p in (_kp(kps, KP_L_HIP, mc), _kp(kps, KP_R_HIP, mc)) if p]
    if shoulders and hips:
        sm = np.mean(shoulders, axis=0)
        hm = np.mean(hips, axis=0)
        torso = float(np.linalg.norm(sm - hm))
    else:
        torso = 0.0
    if torso < 1.0:
        torso = 0.35 * float(box[3] - box[1])
    # Torso exclusion box: hands held in front of the body are not "in the shelf" behind it.
    body = shoulders + hips
    if body:
        bx = [p[0] for p in body]
        by = [p[1] for p in body]
        m = cfg.torso_exclusion_torso * torso
        obs.torso_box = np.array([min(bx) - m, min(by) - 0.5 * m, max(bx) + m, max(by) + 0.5 * m], np.float32)
    # Pocket regions only where hips are actually detected (never invented). The hip threshold is a little
    # lower than for other joints: hips are often partly covered by clothing / a hand going into a pocket.
    pocket_hips = [p for p in (_kp(kps, KP_L_HIP, cfg.hip_min_conf), _kp(kps, KP_R_HIP, cfg.hip_min_conf)) if p]
    obs.pocket_centers = pocket_hips
    obs.hips_visible = bool(pocket_hips)
    hip_line = max(p[1] for p in pocket_hips) if pocket_hips else None
    obs.pocket_radius = cfg.pocket_radius_torso * torso
    obs.bag_boxes = [_expand(b, cfg.bag_box_expand) for b in bags_for_person(box, bag_boxes)]

    for side, idx in (("L", KP_L_WRIST), ("R", KP_R_WRIST)):
        conf = float(kps[idx][2])
        pt = _kp(kps, idx, mc)
        reach = pt or _kp(kps, idx, cfg.pocket_wrist_min_conf)
        hand = HandObservation(side=side, point=pt, confidence=conf, reach_point=reach)
        if pt is not None:
            for name, poly in zone_polys.items():
                if point_in_polygon(pt, poly):
                    hand.in_zone_raw = name
                    if obs.torso_box is None or not _in_box(pt, obs.torso_box):
                        hand.shelf_zone = name
                    break
        if reach is not None:
            low_enough = (hip_line is not None and
                          reach[1] >= min(c[1] for c in pocket_hips) - cfg.pocket_max_above_hip_torso * torso)
            hand.in_pocket = low_enough and any(
                np.hypot(reach[0] - c[0], reach[1] - c[1]) <= obs.pocket_radius for c in pocket_hips)
            hand.in_bag = any(_in_box(reach, b) for b in obs.bag_boxes)
        obs.hands.append(hand)
    return obs
