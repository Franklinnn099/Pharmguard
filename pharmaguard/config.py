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
    confidence_threshold: float = Field(default=0.40, ge=0.05, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.1, le=1.0)
    target_classes: List[int] = Field(
        default_factory=lambda: [0],
        description="Class 0 in COCO corresponds to 'person'."
    )
    device: str = Field(
        default="auto",
        description="'auto', 'cuda', 'cpu', or 'mps'."
    )


class TrackerConfig(BaseModel):
    """ByteTrack multi-object tracking configuration."""
    tracker_type: str = "bytetrack"
    track_activation_threshold: float = 0.35
    lost_track_buffer: int = 30  # Number of frames to hold a lost track before discarding
    minimum_matching_threshold: float = 0.80
    frame_rate: int = 30


class AlertConfig(BaseModel):
    """Alert generation and throttling configuration."""
    cooldown_seconds: float = Field(
        default=10.0,
        description="Minimum seconds before repeating an alert for the same track ID."
    )
    max_history_entries: int = 100
    alert_log_file: Optional[str] = "logs/pharmaguard_alerts.jsonl"


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
