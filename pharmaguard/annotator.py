"""
Visual annotation module for PharmaGuard.
Renders professional security overlays, shelf ROI boundaries, person bounding boxes,
tracking traces, dwell badges, and security status HUD.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2

from pharmaguard.config import DisplayConfig, ShelfZone
from pharmaguard.tracker import TrackingResult, TrackedPerson


class SecurityAnnotator:
    """Renders professional pharmacy monitoring overlays on video frames."""

    def __init__(self, config: Optional[DisplayConfig] = None):
        self.config = config or DisplayConfig()

    def _denormalize_polygon(self, norm_poly: List[List[float]], width: int, height: int) -> np.ndarray:
        pts = [[int(pt[0] * width), int(pt[1] * height)] for pt in norm_poly]
        return np.array(pts, dtype=np.int32)

    def draw_shelf_zones(self, frame: np.ndarray, zones: List[ShelfZone], any_person_in_zone: Dict[str, bool]) -> np.ndarray:
        """Render semi-transparent pharmacy shelf zones with status badges."""
        if not self.config.show_shelf_zones:
            return frame

        h, w = frame.shape[:2]
        overlay = frame.copy()

        for zone in zones:
            pts = self._denormalize_polygon(zone.polygon_normalized, w, h)
            is_active = any_person_in_zone.get(zone.name, False)

            # Red/Amber when someone is interacting, Cyan/Blue when idle/secure
            fill_color = (20, 20, 180) if is_active else (180, 130, 20)  # BGR
            border_color = (50, 50, 255) if is_active else (255, 200, 50)

            # Fill zone with transparency
            cv2.fillPoly(overlay, [pts], fill_color)

            # Draw zone outline
            cv2.polylines(frame, [pts], isClosed=True, color=border_color, thickness=2, lineType=cv2.LINE_AA)

            # Draw zone header badge
            top_pt = pts[0]
            status_text = f"ZONE: {zone.name} [{'OCCUPIED' if is_active else 'SECURED'}]"
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            badge_x = max(10, min(w - text_size[0] - 14, top_pt[0]))
            badge_y = max(40, top_pt[1] - 8)

            cv2.rectangle(
                frame,
                (badge_x - 4, badge_y - text_size[1] - 4),
                (badge_x + text_size[0] + 4, badge_y + 4),
                (30, 30, 30),
                -1
            )
            cv2.putText(
                frame,
                status_text,
                (badge_x, badge_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                border_color,
                1,
                lineType=cv2.LINE_AA
            )

        # Blend semi-transparent fill
        alpha = 0.22
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def draw_tracked_persons(self, frame: np.ndarray, tracking_result: TrackingResult) -> np.ndarray:
        """Render person bounding boxes, persistent IDs, dwell time badges, and motion traces."""
        for tid, person in tracking_result.tracked_persons.items():
            if person.frames_lost > 0:
                continue

            box = person.current_box
            x1, y1, x2, y2 = [int(v) for v in box]

            # Choose color scheme based on zone interaction
            if person.in_shelf_zone:
                box_color = (0, 80, 255)  # Alert Orange/Red
                label_bg = (0, 60, 200)
            else:
                box_color = (255, 180, 40)  # Calm Teal/Blue
                label_bg = (40, 40, 50)

            # 1. Draw motion trace
            if self.config.show_track_traces and len(person.trajectory) > 1:
                pts = np.array(person.trajectory, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False, color=box_color, thickness=2, lineType=cv2.LINE_AA)
                # Small circle at centroid
                cx, cy = int(person.centroid[0]), int(person.centroid[1])
                cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)

            # 2. Draw bounding box corners / rect
            if self.config.show_boxes:
                # Clean sleek security box
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, self.config.box_thickness, lineType=cv2.LINE_AA)

            # 3. Draw ID and Dwell label
            if self.config.show_labels:
                dwell_str = f" | {person.accumulated_dwell_seconds:.1f}s" if person.in_shelf_zone else ""
                label_text = f"ID #{tid}{dwell_str}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.5
                thickness = 1
                t_size = cv2.getTextSize(label_text, font, scale, thickness)[0]

                # Label tag
                lbl_y1 = max(35, y1)
                cv2.rectangle(frame, (x1, lbl_y1 - t_size[1] - 8), (x1 + t_size[0] + 10, lbl_y1), label_bg, -1)
                cv2.rectangle(frame, (x1, lbl_y1 - t_size[1] - 8), (x1 + t_size[0] + 10, lbl_y1), box_color, 1)
                cv2.putText(frame, label_text, (x1 + 5, lbl_y1 - 4), font, scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

        return frame

    def draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        latency_ms: float,
        active_tracks: int,
        alert_count: int,
        connection_status: str = "LIVE"
    ) -> np.ndarray:
        """Render security dashboard header HUD with operational telemetry."""
        if not self.config.show_status_hud:
            return frame

        h, w = frame.shape[:2]
        hud_h = 32

        # Draw HUD bar background
        sub_img = frame[0:hud_h, 0:w]
        dark_rect = np.full(sub_img.shape, self.config.hud_bg_color, dtype=np.uint8)
        cv2.addWeighted(sub_img, 0.25, dark_rect, 0.75, 0, sub_img)

        # Border separator line
        cv2.line(frame, (0, hud_h), (w, hud_h), (60, 70, 80), 1)

        # Status text elements
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1

        # Left: Brand & Status dot
        status_color = (0, 220, 0) if connection_status == "LIVE" else (0, 0, 220)
        cv2.circle(frame, (14, 16), 4, status_color, -1, lineType=cv2.LINE_AA)
        cv2.putText(frame, f"PHARMAGUARD v0.1.0  [{connection_status}]", (26, 21), font, font_scale, (230, 230, 230), thickness, lineType=cv2.LINE_AA)

        # Middle: Active Tracks & Alerts
        stats_text = f"TRACKED: {active_tracks} | ALERTS: {alert_count}"
        t_size = cv2.getTextSize(stats_text, font, font_scale, thickness)[0]
        mid_x = (w - t_size[0]) // 2
        cv2.putText(frame, stats_text, (mid_x, 21), font, font_scale, (200, 200, 200), thickness, lineType=cv2.LINE_AA)

        # Right: FPS & Latency
        telemetry_text = f"FPS: {fps:.1f}  |  LATENCY: {latency_ms:.0f}ms"
        tel_size = cv2.getTextSize(telemetry_text, font, font_scale, thickness)[0]
        tel_x = max(mid_x + t_size[0] + 20, w - tel_size[0] - 15)
        cv2.putText(frame, telemetry_text, (tel_x, 21), font, font_scale, (0, 220, 220), thickness, lineType=cv2.LINE_AA)

        return frame
