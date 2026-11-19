"""
Configuration module for PharmaGuard.
Defines typed schemas for detector, tracker, zones, alerts, and display settings.
"""

from typing import List, Tuple, Optional, Dict, Any
from pydantic import BaseModel, Field
import os
import yaml


class ShelfZone(BaseModel):
    """Configuration for a specific monitored pharmacy shelf zone."""
    name: str = "Prescription_Shelf_A"
    # Normalized coordinates [[x_min, y_min], [x_max, y_max]] or polygon [[x1, y1], [x2, y2], ...]
    polygon_normalized: List[List[float]] = Field(
        default_factory=lambda: [
            [0.10, 0.20],
            [0.50, 0.20],
            [0.50, 0.85],
            [0.10, 0.85]
        ]
    )
    dwell_alert_threshold_sec: float = Field(
        default=8.0,
        description="Dwell time in seconds near high-risk shelf that triggers an alert."
    )
    rapid_retreat_threshold_sec: float = Field(
        default=2.5,
        description="Abrupt grab-and-retreat time window in seconds."
    )


class DetectorConfig(BaseModel):
    """Object detection configuration."""
    model_name: str = Field(
        default="yolov8n.pt",
        description="Pretrained model weights (e.g. yolov8n.pt, yolo11n.pt, rtdetr-l.pt)."
    )
    confidence_threshold: float = Field(
        default=0.10, ge=0.05, le=1.0,
        description="Detector output floor. Must be <= tracker.track_low_threshold so ByteTrack's "
                    "low-confidence recovery stage receives partially occluded people. "
                    "Display/new-track confidence is governed by the tracker thresholds."
    )
    iou_threshold: float = Field(default=0.45, ge=0.1, le=1.0)
    target_classes: List[int] = Field(
        default_factory=lambda: [0],
        description="Class 0 in COCO corresponds to 'person'."
    )
    device: str = Field(
        default="auto",
        description="'auto', 'cuda', 'cpu', or 'mps'."
    )
    imgsz: int = Field(
        default=640, ge=160, le=1280,
        description="Model inference size (letterboxed). Lower = faster, less accurate on small/far people."
    )
    half_precision: bool = Field(
        default=True,
        description="Use FP16 inference when running on CUDA (ignored on CPU)."
    )


class TrackerConfig(BaseModel):
    """ByteTrack multi-object tracking configuration (Zhang et al., ECCV 2022).

    Defaults follow the published ByteTrack / Ultralytics settings, with time-based (not frame-based)
    lifetimes because tunnel-delivered frame rates vary between ~3 and ~25 FPS.
    """
    tracker_type: str = "bytetrack"
    track_high_threshold: float = Field(default=0.45, ge=0.05, le=1.0,
        description="Detections at/above this join the first association stage.")
    track_low_threshold: float = Field(default=0.10, ge=0.01, le=1.0,
        description="Detections in [low, high) are only used to keep EXISTING tracks alive (stage 2).")
    new_track_threshold: float = Field(default=0.50, ge=0.05, le=1.0,
        description="Minimum confidence for an unmatched detection to start a candidate track.")
    match_iou_threshold: float = Field(default=0.20, ge=0.0, le=1.0,
        description="Stage 1 minimum IoU between Kalman-predicted box and detection (ByteTrack: 1-0.8).")
    low_match_iou_threshold: float = Field(default=0.50, ge=0.0, le=1.0,
        description="Stage 2 minimum IoU for low-confidence detections (stricter: they are less reliable).")
    unconfirmed_match_iou_threshold: float = Field(default=0.30, ge=0.0, le=1.0,
        description="Minimum IoU to confirm a candidate track on its next detection.")
    max_center_speed: float = Field(default=2.0, ge=0.0,
        description="Fallback gate when IoU is too low (low FPS / fast motion): accept a match if the box centre "
                    "moved <= this many body-heights per second of elapsed time (walking ~0.8, brisk ~1.2). "
                    "0 disables the fallback (pure ByteTrack IoU gating).")
    center_gate_slack: float = Field(default=0.1, ge=0.0,
        description="Constant allowance (body heights) added to the motion gate for detector box jitter.")
    max_height_ratio: float = Field(default=1.35, ge=1.0,
        description="Fallback gate: detection height must be within this ratio of the predicted height.")
    min_confirm_hits: int = Field(default=2, ge=1,
        description="Consecutive detections before a track gets a visible ID (suppresses 1-frame false positives).")
    lost_track_seconds: float = Field(default=2.0, gt=0.0,
        description="How long an unseen (occluded) track is kept for re-association before removal.")
    lost_velocity_decay: float = Field(default=0.85, gt=0.0, le=1.0,
        description="Per nominal frame decay of predicted velocity while lost, so a person who stops "
                    "behind an occluder is not extrapolated off-screen.")
    frame_rate: int = Field(default=30, ge=1,
        description="Nominal rate the Kalman noise model is tuned for; real time gaps are scaled to it.")
    max_trace_points: int = 30


class AlertConfig(BaseModel):
    """Alert generation and throttling configuration."""
    cooldown_seconds: float = Field(
        default=10.0,
        description="Minimum seconds before repeating an alert for the same track ID."
    )
    max_history_entries: int = 100
    # Overridable via env so automated tests never write into the operational audit log.
    alert_log_file: Optional[str] = Field(
        default_factory=lambda: os.environ.get("PHARMAGUARD_ALERT_LOG", "logs/pharmaguard_alerts.jsonl")
    )


class StreamConfig(BaseModel):
    """Browser <-> server video transport settings (the real-time critical path)."""
    capture_width: int = Field(default=640, description="Ideal webcam capture width requested from the browser.")
    capture_height: int = Field(default=480, description="Ideal webcam capture height requested from the browser.")
    max_process_width: int = Field(
        default=640, ge=160,
        description="Frames wider than this are downscaled server-side before processing and return."
    )
    output_format: str = Field(
        default="jpeg",
        description="Encoding of the annotated frame sent back to the browser. "
                    "Measured: webp 527 ms vs jpeg 13 ms per 1280x720 frame (CPU) - keep jpeg."
    )
    stream_every_sec: float = Field(
        default=0.04, gt=0.0,
        description="Browser capture poll interval. Gradio skips capture while a frame is pending, "
                    "so this is a minimum spacing, not a queue rate."
    )
    signal_timeout_sec: float = Field(
        default=2.0, gt=0.0,
        description="No frame for this long => camera/connection reported as NO SIGNAL."
    )
    telemetry_window: int = Field(default=60, ge=5, description="Frames used for rolling FPS/latency stats.")


class DisplayConfig(BaseModel):
    """Visual rendering settings for annotated stream."""
    show_boxes: bool = True
    show_labels: bool = True
    show_track_traces: bool = True
    show_shelf_zones: bool = True
    show_status_hud: bool = True
    trace_length: int = 30
    box_thickness: int = 2
    hud_bg_color: List[int] = Field(default_factory=lambda: [20, 24, 30])  # Dark slate


class PharmaGuardConfig(BaseModel):
    """Root configuration for PharmaGuard application."""
    app_name: str = "PharmaGuard Pharmacy Security Monitor"
    version: str = "0.1.0"
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    zones: List[ShelfZone] = Field(default_factory=lambda: [ShelfZone()])
    alert: AlertConfig = Field(default_factory=AlertConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    stream: StreamConfig = Field(default_factory=StreamConfig)

    @classmethod
    def load_from_yaml(cls, path: str) -> "PharmaGuardConfig":
        """Load configuration from a YAML file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def save_to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False)
