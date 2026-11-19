"""
Multi-object tracking module for PharmaGuard.

ByteTrack (Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022):

1. Every track is propagated with a constant-velocity Kalman filter in (cx, cy, aspect, height) space.
2. Stage 1: tracked + lost tracks are matched to HIGH-confidence detections (score-fused IoU, Hungarian).
3. Stage 2: still-unmatched *tracked* tracks are matched to LOW-confidence detections. This is what keeps
   partially occluded people (whose detector confidence collapses) on their existing ID.
4. Candidate (unconfirmed) tracks are matched to the remaining high detections; a candidate becomes a
   visible track with an ID only after ``min_confirm_hits`` consecutive detections.
5. Unmatched high detections above ``new_track_threshold`` start candidates. Lost tracks are removed
   after ``lost_track_seconds`` of real time.

Deviation from the paper: the Kalman step is scaled by the real time between frames (``dt * frame_rate``),
because browser-tunnel delivery is irregular, and lost-track velocity decays so prediction does not run away.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from pharmaguard.config import TrackerConfig
from pharmaguard.detector import DetectionResult


class TrackState(str, Enum):
    NEW = "NEW"            # candidate, not yet confirmed, no visible ID
    TRACKED = "TRACKED"    # matched this frame
    LOST = "LOST"          # unmatched, kept alive for re-association
    REMOVED = "REMOVED"


# --------------------------------------------------------------------------- geometry

def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix between two sets of [x1, y1, x2, y2] bounding boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / np.clip(union, a_min=1e-6, a_max=None)
    return iou.astype(np.float32)


def xyxy_to_xyah(box: np.ndarray) -> np.ndarray:
    w = max(float(box[2] - box[0]), 1e-3)
    h = max(float(box[3] - box[1]), 1e-3)
    return np.array([box[0] + w / 2.0, box[1] + h / 2.0, w / h, h], dtype=np.float64)


def xyah_to_xyxy(xyah: np.ndarray) -> np.ndarray:
    cx, cy, a, h = (float(v) for v in xyah[:4])
    h = max(h, 1e-3)
    w = max(a, 1e-3) * h
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)


# --------------------------------------------------------------------------- Kalman filter

class KalmanBoxFilter:
    """Constant-velocity Kalman filter on (cx, cy, a, h), noise model as in ByteTrack/DeepSORT.

    ``steps`` is the elapsed time expressed in nominal frames, so irregular frame spacing is handled:
    velocity is in pixels per nominal frame, and process noise grows with elapsed time.
    """

    STD_POS = 1.0 / 20.0
    STD_VEL = 1.0 / 160.0
    _H = np.hstack([np.eye(4), np.zeros((4, 4))])

    def initiate(self, xyah: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.r_[xyah, np.zeros(4)]
        h = xyah[3]
        std = [2 * self.STD_POS * h, 2 * self.STD_POS * h, 1e-2, 2 * self.STD_POS * h,
               10 * self.STD_VEL * h, 10 * self.STD_VEL * h, 1e-5, 10 * self.STD_VEL * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray, steps: float) -> Tuple[np.ndarray, np.ndarray]:
        steps = max(float(steps), 1e-3)
        F = np.eye(8)
        F[:4, 4:] = np.eye(4) * steps
        h = mean[3]
        std = [self.STD_POS * h, self.STD_POS * h, 1e-2, self.STD_POS * h,
               self.STD_VEL * h, self.STD_VEL * h, 1e-5, self.STD_VEL * h]
        Q = np.diag(np.square(std)) * steps
        mean = F @ mean
        cov = F @ cov @ F.T + Q
        mean[3] = max(mean[3], 1.0)
        mean[2] = max(mean[2], 1e-3)
        return mean, cov

    def update(self, mean: np.ndarray, cov: np.ndarray, xyah: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        R = np.diag(np.square([self.STD_POS * h, self.STD_POS * h, 1e-1, self.STD_POS * h]))
        H = self._H
        S = H @ cov @ H.T + R
        K = np.linalg.solve(S.T, (cov @ H.T).T).T
        mean = mean + K @ (xyah - H @ mean)
        cov = (np.eye(8) - K @ H) @ cov
        return mean, cov


# --------------------------------------------------------------------------- track objects

@dataclass
class TrackedPerson:
    """A tracked individual. ``current_box`` is the last detection while TRACKED and the
    Kalman prediction while LOST (useful for drawing an occluded person's expected position)."""
    tracker_id: int
    current_box: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    centroid: Tuple[float, float]
    state: TrackState = TrackState.NEW
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))  # px / nominal frame
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    first_seen_time: float = field(default_factory=time.monotonic)
    last_seen_time: float = field(default_factory=time.monotonic)
    frames_active: int = 1
    frames_lost: int = 0
    hits: int = 1
    in_shelf_zone: bool = False
    shelf_entry_timestamp: Optional[float] = None
    accumulated_dwell_seconds: float = 0.0
    # COCO-17 keypoints (17, 3) from the matched detection; None while LOST or with a non-pose model.
    keypoints: Optional[np.ndarray] = field(default=None, repr=False)
    # Kalman state (internal)
    kf_mean: Optional[np.ndarray] = field(default=None, repr=False)
    kf_cov: Optional[np.ndarray] = field(default=None, repr=False)

    def predict_box(self) -> np.ndarray:
        """Current Kalman-predicted box (after this frame's predict step)."""
        if self.kf_mean is None:
            return self.current_box
        return xyah_to_xyxy(self.kf_mean)

    def seconds_since_seen(self, now: float) -> float:
        return max(0.0, now - self.last_seen_time)


@dataclass
class TrackingResult:
    """Output bundle from tracker for current frame.

    ``tracked_persons`` holds CONFIRMED tracks only (TRACKED or LOST). ``lost_ids`` lists tracks
    REMOVED this frame (kept name for downstream compatibility: behaviour state is purged for them).
    """
    tracked_persons: Dict[int, TrackedPerson]
    active_ids: List[int]
    new_ids: List[int]
    lost_ids: List[int]
    occluded_ids: List[int] = field(default_factory=list)
    raw_supervision_detections: Optional[Any] = None  # retained for API compatibility; unused


# --------------------------------------------------------------------------- tracker

class ByteTrackerWrapper:
    """ByteTrack with Kalman prediction, two-stage association, candidate confirmation, time-based expiry."""

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        self.kf = KalmanBoxFilter()
        self.reset()

    def reset(self) -> None:
        """Reset all tracked state."""
        self.tracked_registry: Dict[int, TrackedPerson] = {}   # confirmed tracks by visible ID
        self._candidates: List[TrackedPerson] = []
        self._next_id = 1
        self._last_ts: Optional[float] = None

    # ------------------------------------------------------------------ helpers
    def _new_candidate(self, box: np.ndarray, conf: float, ts: float,
                       kp: Optional[np.ndarray] = None) -> TrackedPerson:
        mean, cov = self.kf.initiate(xyxy_to_xyah(box))
        cx, cy = float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0)
        return TrackedPerson(tracker_id=0, current_box=box.astype(np.float32), confidence=conf,
                             centroid=(cx, cy), state=TrackState.NEW, trajectory=[(cx, cy)],
                             first_seen_time=ts, last_seen_time=ts, kf_mean=mean, kf_cov=cov,
                             keypoints=kp)

    def _predict(self, track: TrackedPerson, steps: float) -> None:
        if track.state == TrackState.LOST:
            # Damp extrapolation while unseen; zero height velocity as in ByteTrack.
            track.kf_mean[4:8] *= self.config.lost_velocity_decay ** steps
            track.kf_mean[7] = 0.0
        track.kf_mean, track.kf_cov = self.kf.predict(track.kf_mean, track.kf_cov, steps)

    def _apply_detection(self, track: TrackedPerson, box: np.ndarray, conf: float, ts: float,
                         kp: Optional[np.ndarray] = None) -> None:
        track.kf_mean, track.kf_cov = self.kf.update(track.kf_mean, track.kf_cov, xyxy_to_xyah(box))
        track.current_box = box.astype(np.float32)
        track.confidence = conf
        track.keypoints = kp
        cx, cy = float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0)
        track.centroid = (cx, cy)
        track.velocity = track.kf_mean[4:6].astype(np.float32)
        track.trajectory.append((cx, cy))
        if len(track.trajectory) > self.config.max_trace_points:
            del track.trajectory[0]
        track.last_seen_time = ts
        track.frames_active += 1
        track.frames_lost = 0
        track.hits += 1
        if track.state != TrackState.NEW:
            track.state = TrackState.TRACKED

    def _motion_gate(self, tracks: List[TrackedPerson], preds: np.ndarray, boxes: np.ndarray,
                     now: float) -> Tuple[np.ndarray, np.ndarray]:
        """Time-aware centre-distance gate for when IoU is uninformative (low FPS, fast motion, new
        candidates with no velocity yet). Returns (allowed mask, similarity in [0, min_iou))."""
        cfg = self.config
        pc = (preds[:, None, :2] + preds[:, None, 2:]) / 2.0
        dc = (boxes[None, :, :2] + boxes[None, :, 2:]) / 2.0
        ph = np.maximum(preds[:, 3] - preds[:, 1], 1.0)[:, None]
        dh = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)[None, :]
        dist = np.linalg.norm(pc - dc, axis=2) / ph                       # in body heights
        elapsed = np.array([max(now - t.last_seen_time, 1.0 / cfg.frame_rate) for t in tracks])[:, None]
        limit = np.minimum(cfg.max_center_speed * elapsed + cfg.center_gate_slack, 1.5)
        ratio = np.maximum(ph / dh, dh / ph)
        allowed = (dist <= limit) & (ratio <= cfg.max_height_ratio) & (cfg.max_center_speed > 0)
        sim = np.where(allowed, (1.0 - dist / np.maximum(limit, 1e-6)) * 0.1, 0.0)
        return allowed, sim

    def _associate(self, tracks: List[TrackedPerson], boxes: np.ndarray, scores: np.ndarray,
                   min_iou: float, fuse_score: bool, now: float = 0.0,
                   motion_fallback: bool = False) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Hungarian assignment on (score-fused) IoU; pairs below ``min_iou`` raw IoU are rejected
        unless ``motion_fallback`` admits them via the time-aware centre-distance gate (ranked below
        every IoU-qualified pair)."""
        if not tracks or len(boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(boxes)))
        preds = np.array([t.predict_box() for t in tracks], dtype=np.float32)
        boxes = boxes.astype(np.float32)
        iou = box_iou_matrix(preds, boxes)
        iou_ok = iou >= min_iou
        sim = np.where(iou_ok, 0.2 + (iou * scores[None, :] if fuse_score else iou), 0.0)
        gate_ok = np.zeros_like(iou_ok)
        if motion_fallback:
            gate_ok, gate_sim = self._motion_gate(tracks, preds, boxes, now)
            gate_ok &= ~iou_ok
            sim = np.where(gate_ok, gate_sim, sim)
        rows, cols = linear_sum_assignment(-sim)
        matches, used_t, used_d = [], set(), set()
        for r, c in zip(rows, cols):
            if iou_ok[r, c] or gate_ok[r, c]:
                matches.append((r, c))
                used_t.add(r)
                used_d.add(c)
        return (matches,
                [i for i in range(len(tracks)) if i not in used_t],
                [j for j in range(len(boxes)) if j not in used_d])

    # ------------------------------------------------------------------ main update
    def update(self, detections: DetectionResult, timestamp: Optional[float] = None) -> TrackingResult:
        """Advance the tracker by one frame.

        Args:
            detections: person detections for this frame (any confidence >= detector floor).
            timestamp: frame time in seconds (monotonic). Defaults to ``time.monotonic()``.
                       Pass video time when processing recorded footage.
        """
        cfg = self.config
        ts = time.monotonic() if timestamp is None else float(timestamp)
        dt = 1.0 / cfg.frame_rate if self._last_ts is None else max(ts - self._last_ts, 1e-3)
        self._last_ts = ts
        steps = dt * cfg.frame_rate

        boxes = detections.xyxy if not detections.is_empty() else np.empty((0, 4), np.float32)
        scores = detections.confidence if not detections.is_empty() else np.empty((0,), np.float32)
        kps = detections.keypoints if (detections.keypoints is not None and not detections.is_empty()) else None
        high = scores >= cfg.track_high_threshold
        low = (scores >= cfg.track_low_threshold) & ~high
        hi_boxes, hi_scores = boxes[high], scores[high]
        lo_boxes, lo_scores = boxes[low], scores[low]
        hi_kps = kps[high] if kps is not None else None
        lo_kps = kps[low] if kps is not None else None
        _kp = lambda arr, i: None if arr is None else arr[i]

        confirmed = list(self.tracked_registry.values())
        for t in confirmed + self._candidates:
            self._predict(t, steps)

        active_ids: List[int] = []

        # Stage 1: confirmed (tracked + lost) <-> high-confidence detections
        m1, un_trk, un_hi = self._associate(confirmed, hi_boxes, hi_scores, cfg.match_iou_threshold,
                                            fuse_score=True, now=ts, motion_fallback=True)
        for ti, di in m1:
            self._apply_detection(confirmed[ti], hi_boxes[di], float(hi_scores[di]), ts, _kp(hi_kps, di))
            active_ids.append(confirmed[ti].tracker_id)

        # Stage 2: remaining *currently tracked* tracks <-> low-confidence detections
        remaining = [confirmed[i] for i in un_trk]
        r_tracked = [t for t in remaining if t.state == TrackState.TRACKED]
        r_lost = [t for t in remaining if t.state != TrackState.TRACKED]
        m2, un_trk2, _ = self._associate(r_tracked, lo_boxes, lo_scores, cfg.low_match_iou_threshold, fuse_score=False)
        for ti, di in m2:
            self._apply_detection(r_tracked[ti], lo_boxes[di], float(lo_scores[di]), ts, _kp(lo_kps, di))
            active_ids.append(r_tracked[ti].tracker_id)

        # Unmatched confirmed tracks become / stay LOST; expire on real elapsed time.
        removed: List[int] = []
        occluded: List[int] = []
        for t in [r_tracked[i] for i in un_trk2] + r_lost:
            t.state = TrackState.LOST
            t.frames_lost += 1
            t.current_box = t.predict_box()
            t.keypoints = None
            if t.seconds_since_seen(ts) > cfg.lost_track_seconds:
                t.state = TrackState.REMOVED
                removed.append(t.tracker_id)
                self.tracked_registry.pop(t.tracker_id, None)
            else:
                occluded.append(t.tracker_id)

        # Stage 3: candidates <-> remaining high detections; confirm after enough consecutive hits.
        rem_hi_boxes, rem_hi_scores = hi_boxes[un_hi], hi_scores[un_hi]
        rem_hi_kps = hi_kps[un_hi] if hi_kps is not None else None
        m3, un_cand, un_hi2 = self._associate(self._candidates, rem_hi_boxes, rem_hi_scores,
                                              cfg.unconfirmed_match_iou_threshold, fuse_score=True,
                                              now=ts, motion_fallback=True)
        new_ids: List[int] = []
        survivors: List[TrackedPerson] = []
        for ci, di in m3:
            cand = self._candidates[ci]
            self._apply_detection(cand, rem_hi_boxes[di], float(rem_hi_scores[di]), ts, _kp(rem_hi_kps, di))
            if cand.hits >= cfg.min_confirm_hits:
                cand.tracker_id = self._next_id
                self._next_id += 1
                cand.state = TrackState.TRACKED
                self.tracked_registry[cand.tracker_id] = cand
                new_ids.append(cand.tracker_id)
                active_ids.append(cand.tracker_id)
            else:
                survivors.append(cand)
        # Unmatched candidates are discarded (a candidate must be seen on consecutive frames).

        # New candidates from leftover confident detections.
        for di in un_hi2:
            if rem_hi_scores[di] >= cfg.new_track_threshold:
                cand = self._new_candidate(rem_hi_boxes[di], float(rem_hi_scores[di]), ts, _kp(rem_hi_kps, di))
                if cfg.min_confirm_hits <= 1:
                    cand.tracker_id = self._next_id
                    self._next_id += 1
                    cand.state = TrackState.TRACKED
                    self.tracked_registry[cand.tracker_id] = cand
                    new_ids.append(cand.tracker_id)
                    active_ids.append(cand.tracker_id)
                else:
                    survivors.append(cand)
        self._candidates = survivors

        return TrackingResult(
            tracked_persons=self.tracked_registry,
            active_ids=active_ids,
            new_ids=new_ids,
            lost_ids=removed,
            occluded_ids=occluded,
        )
