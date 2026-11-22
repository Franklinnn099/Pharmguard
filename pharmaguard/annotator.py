"""
Visual annotation module for PharmaGuard.
Renders professional security overlays, shelf ROI boundaries, person bounding boxes,
tracking traces, dwell badges, and security status HUD.
"""

from typing import Callable, List, Dict, Tuple, Optional
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

    def draw_shelf_zones(self, frame: np.ndarray, zones: List[ShelfZone],
                         activity: Dict[str, object]) -> np.ndarray:
        """Semi-transparent shelf zones. ``activity`` maps zone -> 'CONTACT' | 'PRESENCE' | 'IDLE'
        (a bool is accepted as legacy: True = CONTACT)."""
        if not self.config.show_shelf_zones:
            return frame

        h, w = frame.shape[:2]
        overlay = frame.copy()
        styles = {  # BGR fill, border
            "CONTACT": ((20, 20, 180), (50, 50, 255)),
            "PRESENCE": ((20, 120, 180), (40, 180, 255)),
            "IDLE": ((180, 130, 20), (255, 200, 50)),
        }

        for zone in zones:
            pts = self._denormalize_polygon(zone.polygon_normalized, w, h)
            state = activity.get(zone.name, "IDLE")
            if isinstance(state, bool):
                state = "CONTACT" if state else "IDLE"
            fill_color, border_color = styles.get(state, styles["IDLE"])

            cv2.fillPoly(overlay, [pts], fill_color)
            cv2.polylines(frame, [pts], isClosed=True, color=border_color, thickness=2, lineType=cv2.LINE_AA)

            top_pt = pts[0]
            label = {"CONTACT": "HAND CONTACT", "PRESENCE": "PERSON AT SHELF", "IDLE": "CLEAR"}.get(state, state)
            status_text = f"ZONE: {zone.name} [{label}]"
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            badge_x = max(10, min(w - text_size[0] - 14, top_pt[0]))
            badge_y = max(40, top_pt[1] - 8)
            cv2.rectangle(frame, (badge_x - 4, badge_y - text_size[1] - 4),
                          (badge_x + text_size[0] + 4, badge_y + 4), (30, 30, 30), -1)
            cv2.putText(frame, status_text, (badge_x, badge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        border_color, 1, lineType=cv2.LINE_AA)

        alpha = 0.22
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def draw_bags(self, frame: np.ndarray, bag_boxes: List[np.ndarray]) -> np.ndarray:
        if not self.config.show_boxes:
            return frame
        for b in bag_boxes:
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 60, 200), 1, lineType=cv2.LINE_AA)
            cv2.putText(frame, "bag", (x1 + 3, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (200, 60, 200), 1, lineType=cv2.LINE_AA)
        return frame

    def draw_tracked_persons(self, frame: np.ndarray, tracking_result: TrackingResult,
                             views: Optional[Callable[[int], object]] = None) -> np.ndarray:
        """Person boxes coloured by suspicion level, ID + score label, wrist markers, motion traces."""
        level_colors = {0: (255, 180, 40), 1: (0, 190, 255), 2: (0, 0, 255)}   # BGR: blue, amber, red
        for tid, person in tracking_result.tracked_persons.items():
            if person.frames_lost > 0:
                self._draw_occluded(frame, tid, person)
                continue

            view = views(tid) if views else None
            level = getattr(view, "level", 0)
            score = getattr(view, "score", 0.0)
            box_color = level_colors.get(level, level_colors[0])
            if level == 0 and person.in_shelf_zone:
                box_color = (0, 140, 255)
            label_bg = (40, 40, 50) if level == 0 else tuple(int(c * 0.7) for c in box_color)

            x1, y1, x2, y2 = [int(v) for v in person.current_box]

            if self.config.show_track_traces and len(person.trajectory) > 1:
                pts = np.array(person.trajectory, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False, color=box_color, thickness=2, lineType=cv2.LINE_AA)

            if self.config.show_boxes:
                thick = self.config.box_thickness + (2 if level == 2 else 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thick, lineType=cv2.LINE_AA)

            # Pocket regions (thin magenta circles) so the operator can see where "pocket" is; or a warning
            # when the hips are not detected (then pocket detection cannot work).
            obs = getattr(view, "observation", None)
            if obs is not None and obs.has_pose:
                if obs.hips_visible:
                    for pb in obs.pocket_boxes:
                        cv2.rectangle(frame, (int(pb[0]), int(pb[1])), (int(pb[2]), int(pb[3])), (200, 0, 200), 1,
                                      lineType=cv2.LINE_AA)
                else:
                    cv2.putText(frame, "hips not visible: pocket check off", (x1 + 4, min(y2 - 6, frame.shape[0] - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, lineType=cv2.LINE_AA)

            # Wrists: green = free, orange = in shelf, magenta = at pocket/bag.
            # A hollow marker is a low-confidence wrist used only for the pocket/bag test.
            for hand in getattr(view, "hands", []) or []:
                if hand.point is None:
                    rp = getattr(hand, "reach_point", None)
                    if rp is not None:
                        col = (255, 0, 255) if (hand.in_pocket or hand.in_bag) else (80, 220, 80)
                        cv2.circle(frame, (int(rp[0]), int(rp[1])), 6, col, 2, lineType=cv2.LINE_AA)
                    continue
                color = (80, 220, 80)
                if hand.shelf_zone:
                    color = (0, 140, 255)
                if hand.in_pocket or hand.in_bag:
                    color = (255, 0, 255)
                cv2.circle(frame, (int(hand.point[0]), int(hand.point[1])), 6, color, -1, lineType=cv2.LINE_AA)
                cv2.circle(frame, (int(hand.point[0]), int(hand.point[1])), 6, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            if self.config.show_labels:
                tag = {1: " REVIEW", 2: " ALERT"}.get(level, "")
                label_text = f"ID #{tid} | {score:.2f}{tag}" if view is not None else f"ID #{tid}"
                font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                t_size = cv2.getTextSize(label_text, font, scale, thickness)[0]
                lbl_y1 = max(35, y1)
                cv2.rectangle(frame, (x1, lbl_y1 - t_size[1] - 8), (x1 + t_size[0] + 10, lbl_y1), label_bg, -1)
                cv2.rectangle(frame, (x1, lbl_y1 - t_size[1] - 8), (x1 + t_size[0] + 10, lbl_y1), box_color, 1)
                cv2.putText(frame, label_text, (x1 + 5, lbl_y1 - 4), font, scale, (255, 255, 255), thickness,
                            lineType=cv2.LINE_AA)

        return frame

    def _draw_occluded(self, frame: np.ndarray, tid: int, person: TrackedPerson) -> None:
        """Dashed grey box at the Kalman-predicted position of a temporarily unseen person."""
        if not self.config.show_boxes:
            return
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in person.current_box]
        x1, x2 = max(0, x1), min(w - 1, x2)
        y1, y2 = max(0, y1), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return
        color = (150, 150, 150)
        dash = 8
        for x in range(x1, x2, dash * 2):
            cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, 1)
            cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, 1)
        for y in range(y1, y2, dash * 2):
            cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, 1)
            cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, 1)
        if self.config.show_labels:
            cv2.putText(frame, f"ID #{tid} (occluded)", (x1 + 4, max(14, y1 + 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, lineType=cv2.LINE_AA)

    def draw_alarm_border(self, frame: np.ndarray, alarm, frame_index: int) -> np.ndarray:
        """Flashing border in the video itself so the alarm is visible even if the panel is off-screen."""
        state = getattr(alarm, "state", "IDLE")
        if state == "ACTIVE":
            color, flash = (0, 0, 255), True
        elif state == "ACKNOWLEDGED":
            color, flash = (0, 0, 160), False
        elif state == "TEST":
            color, flash = (220, 60, 200), True
        else:
            return frame
        if flash and frame_index % 2:
            return frame
        h, w = frame.shape[:2]
        t = max(6, w // 80)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, t)
        label = {"ACTIVE": "ALARM - REVIEW NOW", "ACKNOWLEDGED": "ALARM ACKNOWLEDGED",
                 "TEST": "ALARM TEST"}[state]
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        x, y = (w - size[0]) // 2, h - 14
        cv2.rectangle(frame, (x - 8, y - size[1] - 8), (x + size[0] + 8, y + 6), color, -1)
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)
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
        left_text = f"PHARMAGUARD [{connection_status}]"
        cv2.putText(frame, left_text, (26, 21), font, font_scale, (230, 230, 230), thickness, lineType=cv2.LINE_AA)
        left_end = 26 + cv2.getTextSize(left_text, font, font_scale, thickness)[0][0]

        # Middle: Active Tracks & Alerts (never overlapping the left block on narrow frames)
        stats_text = f"TRACKED: {active_tracks} | ALERTS: {alert_count}"
        t_size = cv2.getTextSize(stats_text, font, font_scale, thickness)[0]
        mid_x = max((w - t_size[0]) // 2, left_end + 14)
        cv2.putText(frame, stats_text, (mid_x, 21), font, font_scale, (200, 200, 200), thickness, lineType=cv2.LINE_AA)

        # Right: FPS & Latency
        # fps = delivered end-to-end frame rate; latency_ms = server processing time only.
        telemetry_text = f"FPS: {fps:.1f}  |  PROC: {latency_ms:.0f}ms"
        tel_size = cv2.getTextSize(telemetry_text, font, font_scale, thickness)[0]
        tel_x = max(mid_x + t_size[0] + 20, w - tel_size[0] - 15)
        cv2.putText(frame, telemetry_text, (tel_x, 21), font, font_scale, (0, 220, 220), thickness, lineType=cv2.LINE_AA)

        return frame
