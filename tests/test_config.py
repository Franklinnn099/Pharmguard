"""
Unit tests for PharmaGuard configuration schemas.
"""

import os
import pytest
from pharmaguard.config import (
    PharmaGuardConfig,
    DetectorConfig,
    TrackerConfig,
    ShelfZone,
    AlertConfig,
    DisplayConfig
)


def test_default_config_initialization():
    config = PharmaGuardConfig()
    assert config.app_name == "PharmaGuard Pharmacy Security Monitor"
    assert config.detector.model_name == "yolov8n.pt"
    assert config.detector.target_classes == [0]
    assert config.tracker.tracker_type == "bytetrack"
    assert len(config.zones) == 1
    assert config.zones[0].name == "Prescription_Shelf_A"
    assert config.alert.cooldown_seconds == 10.0
    assert config.display.show_boxes is True


def test_yaml_serialization_deserialization(tmp_path):
    config = PharmaGuardConfig()
    config.detector.confidence_threshold = 0.55
    config.zones[0].dwell_alert_threshold_sec = 12.0

    yaml_path = str(tmp_path / "test_config.yaml")
    config.save_to_yaml(yaml_path)

    assert os.path.exists(yaml_path)
    loaded_config = PharmaGuardConfig.load_from_yaml(yaml_path)
    assert loaded_config.detector.confidence_threshold == 0.55
    assert loaded_config.zones[0].dwell_alert_threshold_sec == 12.0
