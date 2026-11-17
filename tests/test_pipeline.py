"""
End-to-end integration tests for PharmaGuard pipeline.
"""

import numpy as np
import pytest
from pharmaguard.config import PharmaGuardConfig
from pharmaguard.pipeline import PharmaGuardPipeline, PipelineOutput


def test_pipeline_end_to_end_synthetic_frame():
    config = PharmaGuardConfig()
    config.detector.device = "cpu"
    pipeline = PharmaGuardPipeline(config)

    # Synthetic RGB frame (640x480)
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)

    # Process frame
    output = pipeline.process_frame(frame)

    assert isinstance(output, PipelineOutput)
    assert output.annotated_frame_rgb.shape == (480, 640, 3)
    assert output.connection_status == "ONLINE"
    assert output.latency_ms > 0
    assert output.fps >= 0
    assert isinstance(output.active_ids, list)


def test_pipeline_reset_state():
    config = PharmaGuardConfig()
    config.detector.device = "cpu"
    pipeline = PharmaGuardPipeline(config)

    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    pipeline.process_frame(frame)
    pipeline.reset_state()

    assert pipeline.frame_counter == 0
    assert len(pipeline.alert_manager.alert_history) == 0
    assert len(pipeline.tracker.tracked_registry) == 0
