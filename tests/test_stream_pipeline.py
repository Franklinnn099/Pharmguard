"""
Tests for the Priority-1 real-time video path: telemetry maths, frame normalisation,
real-image inference, log isolation, and a genuine HTTP round trip through a live Gradio server.
"""

import os
import socket
import sys

import cv2
import numpy as np
import pytest

from pharmaguard.config import AlertConfig, PharmaGuardConfig
from pharmaguard.pipeline import PharmaGuardPipeline, normalize_frame
from pharmaguard.telemetry import StreamTelemetry
from pharmaguard.detector import YOLODetector

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def _sample_image() -> np.ndarray:
    import ultralytics
    path = os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg")
    img = cv2.imread(path)
    assert img is not None, f"sample image missing: {path}"
    return img


# --------------------------------------------------------------------------- telemetry

def test_telemetry_cycle_processing_and_outside_split():
    clk = FakeClock()
    tel = StreamTelemetry(window=10, signal_timeout_sec=2.0, clock=clk)
    for _ in range(5):
        a = tel.mark_arrival()
        clk.t += 0.050          # 50 ms in pipeline
        tel.mark_return(a)
        clk.t += 0.150          # 150 ms encode + network + browser
    snap = tel.snapshot()
    assert snap.signal == "LIVE"
    assert snap.cycle_p50_ms == pytest.approx(200.0, abs=0.5)
    assert snap.processing_p50_ms == pytest.approx(50.0, abs=0.5)
    assert snap.outside_p50_ms == pytest.approx(150.0, abs=0.5)
    assert snap.delivered_fps == pytest.approx(5.0, abs=0.05)


def test_telemetry_detects_interruption_and_excludes_gap_from_stats():
    clk = FakeClock()
    tel = StreamTelemetry(window=10, signal_timeout_sec=2.0, clock=clk)
    assert tel.snapshot().signal == "STANDBY"
    for _ in range(3):
        tel.mark_return(tel.mark_arrival())
        clk.t += 0.1
    clk.t += 5.0                                  # camera unplugged / tunnel stalls
    assert tel.snapshot().signal == "NO_SIGNAL"
    tel.mark_return(tel.mark_arrival())           # stream resumes
    snap = tel.snapshot()
    assert snap.signal == "LIVE"
    assert snap.interruptions == 1
    assert snap.cycle_p95_ms < 1000.0             # 5 s outage did not pollute latency stats


# --------------------------------------------------------------------------- frame handling

@pytest.mark.parametrize("shape", [(720, 1280, 3), (720, 1280, 4), (720, 1280)])
def test_normalize_frame_downscales_and_fixes_channels(shape):
    frame = np.zeros(shape, dtype=np.uint8)
    out = normalize_frame(frame, max_width=640)
    assert out.shape == (360, 640, 3)
    assert out.dtype == np.uint8


def test_normalize_frame_never_upscales():
    out = normalize_frame(np.zeros((240, 320, 3), np.uint8), max_width=640)
    assert out.shape == (240, 320, 3)


def test_pipeline_on_real_people_image_downscaled():
    """Real photo with people, HD input: must be downscaled and people must be detected and tracked."""
    pipeline = PharmaGuardPipeline(PharmaGuardConfig())
    rgb = cv2.cvtColor(cv2.resize(_sample_image(), (1280, 720)), cv2.COLOR_BGR2RGB)
    out = pipeline.process_frame(rgb)
    out = pipeline.process_frame(rgb)
    assert out.metadata["input_resolution"] == "1280x720"
    assert out.metadata["resolution"] == "640x360"
    assert out.annotated_frame_rgb.shape == (360, 640, 3)
    assert out.metadata["detections_count"] >= 2
    assert out.active_tracks_count >= 2


# --------------------------------------------------------------------------- config / safety

def test_tests_do_not_write_operational_audit_log():
    assert os.path.normpath(AlertConfig().alert_log_file) != os.path.normpath("logs/pharmaguard_alerts.jsonl")


def test_precision_kwargs_cpu_and_cuda():
    assert YOLODetector._resolve_precision_kwargs(False) == {}
    kw = YOLODetector._resolve_precision_kwargs(True)
    assert kw in ({"quantize": 16}, {"half": True})


# --------------------------------------------------------------------------- live HTTP transport

def test_live_server_http_round_trip_returns_annotated_jpeg():
    """Real server, real HTTP: upload a JPEG frame, run the pipeline, download the JPEG result."""
    from pharmaguard.ui import create_pharmaguard_ui
    from e2e_benchmark import GradioHTTP

    pipeline = PharmaGuardPipeline(PharmaGuardConfig())
    demo = create_pharmaguard_ui(pipeline)
    demo.queue()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    try:
        demo.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True)
        http = GradioHTTP(f"http://127.0.0.1:{port}", None, None)
        _, jpeg = cv2.imencode(".jpg", cv2.resize(_sample_image(), (640, 480)))
        for _ in range(3):
            stages = http.round_trip(jpeg.tobytes())
        assert set(stages) == {"upload", "submit", "wait_result", "download"}
        assert pipeline.frame_counter == 3
        assert pipeline.last_output is not None and pipeline.last_output.active_tracks_count >= 2
        tele = pipeline.telemetry_snapshot()
        assert tele.signal == "LIVE" and tele.cycle_p50_ms > 0
    finally:
        demo.close()
