"""
End-to-end computer vision pipeline for PharmaGuard.
Orchestrates frame ingestion, inference, tracking, behavioral analysis,
alert generation, fine-grained telemetry measurement, and visual annotation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import time
import numpy as np
import cv2
import torch

from pharmaguard.config import PharmaGuardConfig
from pharmaguard.detector import BaseDetector, YOLODetector, DetectionResult
from pharmaguard.tracker import ByteTrackerWrapper, TrackingResult, TrackedPerson
from pharmaguard.behavior import PharmacyBehaviorAnalyzer, BehaviorEvent
from pharmaguard.alerts import AlertManager, SecurityAlert
from pharmaguard.events import EventManager, SystemEvent
from pharmaguard.annotator import SecurityAnnotator


@dataclass
class PipelineOutput:
    """Output package produced for each processed video frame."""
    annotated_frame_rgb: np.ndarray
    fps: float
    latency_ms: float
    active_tracks_count: int
    active_ids: List[int]
    recent_alerts: List[SecurityAlert]
    recent_events: List[SystemEvent]
    connection_status: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PharmaGuardPipeline:
    """Core real-time processing pipeline."""

    def __init__(
        self,
        config: Optional[PharmaGuardConfig] = None,
        detector: Optional[BaseDetector] = None
    ):
        self.config = config or PharmaGuardConfig()

        # Component Initialization
        self.detector = detector or YOLODetector(self.config.detector)
        self.tracker = ByteTrackerWrapper(self.config.tracker)
        self.behavior_analyzer = PharmacyBehaviorAnalyzer(self.config.zones)
        self.alert_manager = AlertManager(self.config.alert)
        self.event_manager = EventManager()
        self.annotator = SecurityAnnotator(self.config.display)

        # Performance & Timing Metrics
        self._frame_times: List[float] = []
        self._last_frame_ts: float = time.time()
        self.fps: float = 0.0
        self.latency_ms: float = 0.0
        self.frame_counter: int = 0

        # Fine-grained stage latencies (ms)
        self.last_stage_timings: Dict[str, float] = {
            "detect_ms": 0.0,
            "track_ms": 0.0,
            "behavior_ms": 0.0,
            "annotate_ms": 0.0,
            "total_backend_ms": 0.0
        }

        model_info = self.detector.get_model_info()
        self.device = model_info.get("device", "cpu")
        self.model_name = model_info.get("model_name", "unknown")

        self.event_manager.log(
            category="SYSTEM",
            message=f"Pipeline initialized with model '{self.model_name}' on device '{self.device}'"
        )

    def process_frame(self, frame_rgb: np.ndarray) -> PipelineOutput:
        """Process a single frame captured from the webcam (RGB format).

        Args:
            frame_rgb: RGB image numpy array (H, W, 3) or (H, W, 4).

        Returns:
            PipelineOutput with annotated frame and monitoring metrics.
        """
        start_time = time.perf_counter()
        self.frame_counter += 1

        if frame_rgb is None or frame_rgb.size == 0:
            return self._create_empty_output(start_time, "NO_SIGNAL")

        # Handle RGBA frames from web browser canvas/WebRTC if 4-channel
        if len(frame_rgb.shape) == 3 and frame_rgb.shape[2] == 4:
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGBA2RGB)
        elif len(frame_rgb.shape) == 2:
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2RGB)

        h, w = frame_rgb.shape[:2]

        # Convert to BGR for OpenCV processing & standard model input
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 1. Person Detection (YOLO / RT-DETR)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_det_start = time.perf_counter()
        detection_result = self.detector.detect(frame_bgr)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_det_end = time.perf_counter()
        detect_ms = (t_det_end - t_det_start) * 1000.0

        # 2. Multi-Object Tracking (ByteTrack)
        t_track_start = time.perf_counter()
        tracking_result = self.tracker.update(detection_result)
        t_track_end = time.perf_counter()
        track_ms = (t_track_end - t_track_start) * 1000.0

        # 3. Behavioral Analysis (Shelf Zone monitoring)
        t_beh_start = time.perf_counter()
        behavior_events = self.behavior_analyzer.analyze(tracking_result, frame_width=w, frame_height=h)
        t_beh_end = time.perf_counter()
        behavior_ms = (t_beh_end - t_beh_start) * 1000.0

        # 4. Alert Management
        new_alerts = self.alert_manager.process_behavior_events(behavior_events)

        # Log new events to event timeline
        for b_evt in behavior_events:
            if b_evt.severity in ("WARNING", "ALERT"):
                self.event_manager.log(
                    category="SECURITY",
                    message=b_evt.description,
                    metadata={"zone": b_evt.zone_name, "tracker_id": b_evt.tracker_id}
                )

        # Check which zones are occupied
        zone_occupancy: Dict[str, bool] = {}
        for zone in self.config.zones:
            zone_occupancy[zone.name] = any(
                p.in_shelf_zone for p in tracking_result.tracked_persons.values() if p.frames_lost == 0
            )

        # 5. Visual Annotation
        t_ann_start = time.perf_counter()
        annotated_bgr = frame_bgr.copy()
        annotated_bgr = self.annotator.draw_shelf_zones(annotated_bgr, self.config.zones, zone_occupancy)
        annotated_bgr = self.annotator.draw_tracked_persons(annotated_bgr, tracking_result)

        # Calculate FPS and latency
        end_time = time.perf_counter()
        self.latency_ms = (end_time - start_time) * 1000.0

        now = time.time()
        delta = now - self._last_frame_ts
        self._last_frame_ts = now
        if delta > 0:
            instant_fps = 1.0 / delta
            self._frame_times.append(instant_fps)
            if len(self._frame_times) > 30:
                self._frame_times.pop(0)
            self.fps = float(np.mean(self._frame_times))
        else:
            self.fps = 30.0

        # Draw HUD overlay
        annotated_bgr = self.annotator.draw_hud(
            frame=annotated_bgr,
            fps=self.fps,
            latency_ms=self.latency_ms,
            active_tracks=len(tracking_result.active_ids),
            alert_count=len(self.alert_manager.alert_history),
            connection_status="LIVE"
        )
        t_ann_end = time.perf_counter()
        annotate_ms = (t_ann_end - t_ann_start) * 1000.0

        # Convert back to RGB for browser/Gradio presentation
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        self.last_stage_timings = {
            "detect_ms": round(detect_ms, 2),
            "track_ms": round(track_ms, 2),
            "behavior_ms": round(behavior_ms, 2),
            "annotate_ms": round(annotate_ms, 2),
            "total_backend_ms": round(self.latency_ms, 2)
        }

        metadata = {
            "resolution": f"{w}x{h}",
            "device": self.device,
            "model_name": self.model_name,
            "stage_timings": self.last_stage_timings,
            "detections_count": detection_result.count,
            "frame_counter": self.frame_counter
        }

        return PipelineOutput(
            annotated_frame_rgb=annotated_rgb,
            fps=round(self.fps, 1),
            latency_ms=round(self.latency_ms, 1),
            active_tracks_count=len(tracking_result.active_ids),
            active_ids=tracking_result.active_ids,
            recent_alerts=self.alert_manager.get_recent_alerts(limit=10),
            recent_events=self.event_manager.get_recent(limit=15),
            connection_status="ONLINE",
            metadata=metadata
        )

    def _create_empty_output(self, start_time: float, status: str) -> PipelineOutput:
        """Fallback when frame is invalid or dropped."""
        blank_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_rgb, f"SIGNAL LOSS: {status}", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 50, 50), 2)
        latency = (time.perf_counter() - start_time) * 1000.0
        return PipelineOutput(
            annotated_frame_rgb=blank_rgb,
            fps=0.0,
            latency_ms=round(latency, 1),
            active_tracks_count=0,
            active_ids=[],
            recent_alerts=self.alert_manager.get_recent_alerts(limit=10),
            recent_events=self.event_manager.get_recent(limit=15),
            connection_status=status,
            metadata={"device": getattr(self, "device", "unknown"), "status": status}
        )

    def reset_state(self) -> None:
        """Reset trackers and alerts for fresh session."""
        self.tracker.reset()
        self.alert_manager.clear()
        self.event_manager.clear()
        self._frame_times.clear()
        self.frame_counter = 0
