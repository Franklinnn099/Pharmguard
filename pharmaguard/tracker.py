"""
Multi-object tracking module for PharmaGuard.
Implements the ByteTrack algorithm (Zhang et al., ECCV 2022) with two-stage association,
Kalman-smoothed trajectory prediction, persistent IDs, and occlusion buffering.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

from pharmaguard.config import TrackerConfig
from pharmaguard.detector import DetectionResult


class TrackState(str, Enum):
    NEW = "NEW"
    TRACKED = "TRACKED"
    LOST = "LOST"
    REMOVED = "REMOVED"


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix between two sets of [x1, y1, x2, y2] bounding boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    # Intersections
    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])  # [N, M, 2]
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])  # [N, M, 2]
    wh = np.clip(rb - lt, a_min=0, a_max=None)                   # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]                            # [N, M]

    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / np.clip(union, a_min=1e-6, a_max=None)
    return iou.astype(np.float32)


@dataclass
class TrackedPerson:
    """Represents a tracked individual over consecutive video frames."""
    tracker_id: int
    current_box: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    centroid: Tuple[float, float]
    state: TrackState = TrackState.NEW
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    first_seen_time: float = field(default_factory=time.time)
    last_seen_time: float = field(default_factory=time.time)
    frames_active: int = 1
    frames_lost: int = 0
    in_shelf_zone: bool = False
    shelf_entry_timestamp: Optional[float] = None
    accumulated_dwell_seconds: float = 0.0

    def predict_box(self) -> np.ndarray:
        """Predict expected bounding box location using current velocity."""
        return self.current_box + (self.velocity * 0.7)

    def update_position(self, box: np.ndarray, confidence: float, max_trace: int = 30) -> None:
        """Update person location, compute velocity delta, and update trajectory."""
        self.velocity = box - self.current_box
        self.current_box = box
        self.confidence = confidence
        self.state = TrackState.TRACKED

        cx = float((box[0] + box[2]) / 2.0)
        cy = float((box[1] + box[3]) / 2.0)
        self.centroid = (cx, cy)
        self.trajectory.append(self.centroid)
        if len(self.trajectory) > max_trace:
            self.trajectory.pop(0)

        self.last_seen_time = time.time()
        self.frames_active += 1
        self.frames_lost = 0

    def mark_lost(self) -> None:
        """Increment lost frames count when track is occluded or not detected."""
        self.frames_lost += 1
        self.state = TrackState.LOST


@dataclass
class TrackingResult:
    """Output bundle from tracker for current frame."""
    tracked_persons: Dict[int, TrackedPerson]
    active_ids: List[int]
    new_ids: List[int]
    lost_ids: List[int]
    raw_supervision_detections: Optional[Any] = None


class ByteTrackerWrapper:
    """ByteTrack algorithm: associates detections across two confidence tiers."""

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        self.tracked_registry: Dict[int, TrackedPerson] = {}
        self._next_id = 1

    def _match(
        self,
        tracks: List[TrackedPerson],
        boxes: np.ndarray,
        iou_threshold: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Perform bipartite matching between tracks and detection boxes."""
        if len(tracks) == 0:
            return [], [], list(range(len(boxes)))
        if len(boxes) == 0:
            return [], list(range(len(tracks))), []

        track_boxes = np.array([t.predict_box() for t in tracks], dtype=np.float32)
        ious = box_iou_matrix(track_boxes, boxes)
        cost_matrix = 1.0 - ious

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched: List[Tuple[int, int]] = []
        unmatched_tracks: List[int] = list(range(len(tracks)))
        unmatched_boxes: List[int] = list(range(len(boxes)))

        for r, c in zip(row_ind, col_ind):
            if ious[r, c] >= iou_threshold:
                matched.append((r, c))
                if r in unmatched_tracks:
                    unmatched_tracks.remove(r)
                if c in unmatched_boxes:
                    unmatched_boxes.remove(c)

        return matched, unmatched_tracks, unmatched_boxes

    def update(self, detections: DetectionResult) -> TrackingResult:
        """Execute two-stage ByteTrack association for the current frame."""
        new_ids: List[int] = []
        lost_ids: List[int] = []
        current_active_ids: List[int] = []

        if detections.is_empty():
            # All existing tracks become lost
            all_tids = list(self.tracked_registry.keys())
            for tid in all_tids:
                person = self.tracked_registry[tid]
                person.mark_lost()
                if person.frames_lost > self.config.lost_track_buffer:
                    lost_ids.append(tid)
                    del self.tracked_registry[tid]

            return TrackingResult(
                tracked_persons=self.tracked_registry,
                active_ids=[],
                new_ids=[],
                lost_ids=lost_ids
            )

        boxes = detections.xyxy
        confs = detections.confidence

        # Step 1: Split into High-confidence and Low-confidence detections
        high_mask = confs >= self.config.track_activation_threshold
        low_mask = ~high_mask

        high_boxes = boxes[high_mask]
        high_confs = confs[high_mask]
        low_boxes = boxes[low_mask]
        low_confs = confs[low_mask]

        existing_tracks = list(self.tracked_registry.values())

        # Step 2: First association (Active/Lost tracks <-> High-confidence detections)
        matched_1, unmatched_tracks_idx, unmatched_high_idx = self._match(
            existing_tracks, high_boxes, iou_threshold=self.config.minimum_matching_threshold * 0.5
        )

        # Update first-stage matched tracks
        for track_idx, box_idx in matched_1:
            track = existing_tracks[track_idx]
            track.update_position(high_boxes[box_idx], float(high_confs[box_idx]))
            current_active_ids.append(track.tracker_id)

        # Step 3: Second association (Remaining tracks <-> Low-confidence detections)
        remaining_tracks = [existing_tracks[i] for i in unmatched_tracks_idx]
        matched_2, unmatched_tracks_2_idx, _ = self._match(
            remaining_tracks, low_boxes, iou_threshold=0.3
        )

        for track_idx, box_idx in matched_2:
            track = remaining_tracks[track_idx]
            track.update_position(low_boxes[box_idx], float(low_confs[box_idx]))
            current_active_ids.append(track.tracker_id)

        # Step 4: Initialize new tracks for unmatched high-confidence detections
        for box_idx in unmatched_high_idx:
            box = high_boxes[box_idx]
            conf = float(high_confs[box_idx])
            tid = self._next_id
            self._next_id += 1

            cx = float((box[0] + box[2]) / 2.0)
            cy = float((box[1] + box[3]) / 2.0)
            new_person = TrackedPerson(
                tracker_id=tid,
                current_box=box,
                confidence=conf,
                centroid=(cx, cy),
                trajectory=[(cx, cy)],
                state=TrackState.TRACKED
            )
            self.tracked_registry[tid] = new_person
            new_ids.append(tid)
            current_active_ids.append(tid)

        # Step 5: Mark truly lost tracks and purge when buffer exceeded
        truly_unmatched_tracks = [remaining_tracks[i] for i in unmatched_tracks_2_idx]
        for track in truly_unmatched_tracks:
            track.mark_lost()
            if track.frames_lost > self.config.lost_track_buffer:
                lost_ids.append(track.tracker_id)
                self.tracked_registry.pop(track.tracker_id, None)

        return TrackingResult(
            tracked_persons=self.tracked_registry,
            active_ids=current_active_ids,
            new_ids=new_ids,
            lost_ids=lost_ids
        )

    def reset(self) -> None:
        """Reset all tracked state."""
        self.tracked_registry.clear()
        self._next_id = 1
