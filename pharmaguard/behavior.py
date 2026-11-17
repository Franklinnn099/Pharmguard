"""
Behavior analysis module for PharmaGuard.
Monitors person interactions around designated pharmacy shelf zones, calculates dwell times,
and flags suspicious movement patterns (e.g. prolonged loitering, rapid grab-and-retreat).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional
import time
import numpy as np
import cv2

from pharmaguard.config import ShelfZone
from pharmaguard.tracker import TrackedPerson, TrackingResult


class BehaviorEventType(str, Enum):
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    SUSPICIOUS_LOITERING = "SUSPICIOUS_LOITERING"
    RAPID_RETREAT = "RAPID_RETREAT"
    MULTI_PERSON_RESTRICTED = "MULTI_PERSON_RESTRICTED"


@dataclass
class BehaviorEvent:
    """Represents a detected behavioral interaction or alert candidate."""
    event_type: BehaviorEventType
    tracker_id: int
    zone_name: str
    timestamp: float = field(default_factory=time.time)
    dwell_time_seconds: float = 0.0
    velocity: float = 0.0
    description: str = ""
    severity: str = "INFO"  # "INFO", "WARNING", "ALERT"


class PharmacyBehaviorAnalyzer:
    """Analyzes spatial and temporal interactions of tracked individuals within pharmacy zones."""

    def __init__(self, zones: List[ShelfZone]):
        self.zones = zones
        # Maps (tracker_id, zone_name) -> entry_timestamp
        self.zone_entry_times: Dict[Tuple[int, str], float] = {}
        # Maps (tracker_id, zone_name) -> last_alert_time
        self.zone_alert_times: Dict[Tuple[int, str], float] = {}

    def _denormalize_polygon(self, norm_poly: List[List[float]], width: int, height: int) -> np.ndarray:
        """Convert normalized (0-1) coordinates into absolute pixel coordinates."""
        pts = [[int(pt[0] * width), int(pt[1] * height)] for pt in norm_poly]
        return np.array(pts, dtype=np.int32)

    def _is_point_in_zone(self, point: Tuple[float, float], zone_poly_px: np.ndarray) -> bool:
        """Test if a pixel point (x, y) falls inside the polygon."""
        dist = cv2.pointPolygonTest(zone_poly_px, (float(point[0]), float(point[1])), measureDist=False)
        return dist >= 0

    def analyze(self, tracking_result: TrackingResult, frame_width: int, frame_height: int) -> List[BehaviorEvent]:
        """Analyze tracking trajectories against configured pharmacy shelf zones.

        Args:
            tracking_result: Output from tracker update.
            frame_width: Width of incoming video frame.
            frame_height: Height of incoming video frame.

        Returns:
            List of detected BehaviorEvent instances.
        """
        events: List[BehaviorEvent] = []
        current_time = time.time()

        for zone in self.zones:
            zone_poly_px = self._denormalize_polygon(zone.polygon_normalized, frame_width, frame_height)
            persons_in_zone: List[int] = []

            for tid, person in tracking_result.tracked_persons.items():
                if person.frames_lost > 0:
                    continue

                # Use bottom center (ground contact / feet) for accurate floor zone intersection
                box = person.current_box
                ground_point = (float((box[0] + box[2]) / 2.0), float(box[3]))

                inside = self._is_point_in_zone(ground_point, zone_poly_px)
                key = (tid, zone.name)

                if inside:
                    persons_in_zone.append(tid)
                    if key not in self.zone_entry_times:
                        # Just entered
                        self.zone_entry_times[key] = current_time
                        events.append(
                            BehaviorEvent(
                                event_type=BehaviorEventType.ZONE_ENTER,
                                tracker_id=tid,
                                zone_name=zone.name,
                                timestamp=current_time,
                                dwell_time_seconds=0.0,
                                description=f"Person #{tid} entered {zone.name}.",
                                severity="INFO"
                            )
                        )
                    else:
                        # Already inside, compute dwell time
                        dwell_time = current_time - self.zone_entry_times[key]
                        person.accumulated_dwell_seconds = dwell_time
                        person.in_shelf_zone = True

                        # Check for suspicious loitering threshold
                        if dwell_time >= zone.dwell_alert_threshold_sec:
                            last_alert = self.zone_alert_times.get(key, 0.0)
                            # Alert if not already alerted within last 10 seconds
                            if (current_time - last_alert) >= 10.0:
                                self.zone_alert_times[key] = current_time
                                events.append(
                                    BehaviorEvent(
                                        event_type=BehaviorEventType.SUSPICIOUS_LOITERING,
                                        tracker_id=tid,
                                        zone_name=zone.name,
                                        timestamp=current_time,
                                        dwell_time_seconds=round(dwell_time, 1),
                                        description=(
                                            f"Suspicious dwell time: Person #{tid} lingering in {zone.name} "
                                            f"for {round(dwell_time, 1)}s (Threshold: {zone.dwell_alert_threshold_sec}s)."
                                        ),
                                        severity="ALERT"
                                    )
                                )
                else:
                    if key in self.zone_entry_times:
                        # Just exited
                        dwell_time = current_time - self.zone_entry_times[key]
                        del self.zone_entry_times[key]
                        person.in_shelf_zone = False

                        # Check for rapid grab and retreat
                        if 0.5 <= dwell_time <= zone.rapid_retreat_threshold_sec:
                            events.append(
                                BehaviorEvent(
                                    event_type=BehaviorEventType.RAPID_RETREAT,
                                    tracker_id=tid,
                                    zone_name=zone.name,
                                    timestamp=current_time,
                                    dwell_time_seconds=round(dwell_time, 1),
                                    description=(
                                        f"Rapid retreat detected: Person #{tid} left {zone.name} "
                                        f"after only {round(dwell_time, 1)}s interaction."
                                    ),
                                    severity="WARNING"
                                )
                            )
                        else:
                            events.append(
                                BehaviorEvent(
                                    event_type=BehaviorEventType.ZONE_EXIT,
                                    tracker_id=tid,
                                    zone_name=zone.name,
                                    timestamp=current_time,
                                    dwell_time_seconds=round(dwell_time, 1),
                                    description=f"Person #{tid} exited {zone.name} (Dwell: {round(dwell_time, 1)}s).",
                                    severity="INFO"
                                )
                            )

            # Check for multiple individuals congregating in sensitive shelf zone
            if len(persons_in_zone) >= 2:
                events.append(
                    BehaviorEvent(
                        event_type=BehaviorEventType.MULTI_PERSON_RESTRICTED,
                        tracker_id=persons_in_zone[0],
                        zone_name=zone.name,
                        timestamp=current_time,
                        dwell_time_seconds=0.0,
                        description=(
                            f"Multi-person interaction: {len(persons_in_zone)} individuals "
                            f"(IDs: {persons_in_zone}) present in {zone.name} simultaneously."
                        ),
                        severity="WARNING"
                    )
                )

        # Cleanup entries for lost tracks
        for lost_id in tracking_result.lost_ids:
            for zone in self.zones:
                key = (lost_id, zone.name)
                self.zone_entry_times.pop(key, None)
                self.zone_alert_times.pop(key, None)

        return events
