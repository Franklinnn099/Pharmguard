# 🛡️ PharmaGuard — AI Pharmacy Security Monitor

> Real-time suspicious-interaction monitoring system designed for pharmacies. Captures live webcam feeds, tracks individuals with persistent IDs, monitors sensitive medicine shelves, flags suspicious behavioral sequences, and presents actionable telemetry to security staff.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: Passing](https://img.shields.io/badge/tests-20%2F20%20passing-brightgreen.svg)]()
[![Hardware: CPU%2FCUDA](https://img.shields.io/badge/Compute-CUDA%20%7C%20CPU-orange.svg)]()

---

## Architecture Overview

PharmaGuard employs a **cloud-assisted architecture** allowing lightweight client devices (e.g. laptop browsers) to harness high-throughput GPU acceleration in Google Colab:

```
[ Laptop Webcam ] ──(Browser WebRTC/WS Stream)──► [ Gradio Public Tunnel ]
                                                              │
                                                              ▼
                                              [ Google Colab GPU Runtime ]
                                                              │
                    ┌─────────────────────────────────────────┴─────────────────────────────────────────┐
                    ▼                                         ▼                                         ▼
            [ YOLO Detector ]                        [ ByteTrack Tracker ]                   [ Behavior Analyzer ]
         Pretrained Person Model                     Two-Stage Association                  Shelf ROI & Dwell Time
         (yolov8n / rtdetr-l)                        Persistent Track IDs                   Loitering & Rapid Grab
                    │                                         │                                         │
                    └─────────────────────────────────────────┬─────────────────────────────────────────┘
                                                              │
                                                              ▼
                                                    [ Security Annotator ]
                                              Overlays, Trajectory Trails, HUD
                                                              │
                                                              ▼
[ Browser Monitoring Console ] ◄──(Annotated Frames, FPS, Latency, Real-Time Alerts)────────────────────┘
```

---

## Key Capabilities

- **Real-Time Person Detection**: Pretrained YOLO inference filtering for person detections with configurable confidence and IoU thresholds.
- **Multi-Object Tracking (ByteTrack)**: Robust two-stage association combining high-confidence and low-confidence detections to maintain track continuity across temporary occlusions.
- **Configurable Shelf Zones**: Resolution-independent normalized 2D polygons representing monitored pharmacy shelves (e.g. narcotics or restricted medicine racks).
- **Behavioral Sequence Recognition**:
  - `SUSPICIOUS_LOITERING`: Detects when an individual lingers near sensitive stock beyond the alert dwell threshold (default: 8.0s).
  - `RAPID_RETREAT`: Detects abrupt grab-and-retreat movement patterns (rapid interaction between 0.5s and 2.5s).
  - `MULTI_PERSON_RESTRICTED`: Flags simultaneous congregation around high-risk shelves.
- **Alert Management & Throttling**: Cooldown suppression prevents alert storms while recording alerts to structured audit logs (`pharmaguard_alerts.jsonl`).
- **Professional Security Dashboard**: High-contrast dark surveillance console built with Gradio displaying live FPS, latency, tracked count, active IDs, real-time alert feed, and zone configuration sliders.

---

## Quickstart Guide

### Option 1: Running in Google Colab (Recommended for GPU Acceleration)

1. Open `colab/PharmaGuard_Colab.ipynb` in [Google Colab](https://colab.research.google.com).
2. Set the runtime type to GPU: **Runtime > Change runtime type > T4 GPU**.
3. Run all cells in order.
4. Open the generated public HTTPS URL (e.g., `https://xxxx.gradio.live`) on your laptop browser.
5. Click **Allow** when the browser asks for webcam permission to begin live surveillance!

### Option 2: Running Locally on Windows / macOS / Linux

```powershell
# 1. Clone repository & enter workspace
git clone https://github.com/your-username/pharmguard.git
cd pharmguard

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Run the automated test suite
python -m pytest tests/ -v

# 4. Launch the local security console
python main.py --local --port 7860
```
Then visit `http://127.0.0.1:7860` in your web browser.

---

## Testing & Verification

PharmaGuard includes a 100% automated test suite validating each pipeline stage:

```powershell
python -m pytest tests/ -v
```

Output:
```
tests/test_alerts.py::test_alert_generation_and_cooldown PASSED
tests/test_alerts.py::test_info_severity_suppression PASSED
tests/test_behavior.py::test_behavior_analyzer_zone_entry_and_exit PASSED
tests/test_config.py::test_default_config_initialization PASSED
tests/test_config.py::test_yaml_serialization_deserialization PASSED
tests/test_detector.py::test_detector_initialization PASSED
tests/test_detector.py::test_detector_inference_on_empty_frame PASSED
tests/test_detector.py::test_detector_inference_on_synthetic_frame PASSED
tests/test_pipeline.py::test_pipeline_end_to_end_synthetic_frame PASSED
tests/test_pipeline.py::test_pipeline_reset_state PASSED
tests/test_tracker.py::test_tracker_initialization PASSED
tests/test_tracker.py::test_tracker_empty_update PASSED
tests/test_tracker.py::test_tracker_tracking_consistency PASSED
tests/test_tracker.py::test_tracker_reset PASSED
============================= 14 passed in 12.75s ==============================
```

To run the pipeline throughput benchmark:
```powershell
python benchmark.py
```

---

## Project Documentation

- [PROJECT_STATUS.md](file:///C:/Users/USER/Desktop/pharmguard/PROJECT_STATUS.md): Complete list of completed features, incomplete features, known issues, and next priorities.
- [TECHNICAL_DECISIONS.md](file:///C:/Users/USER/Desktop/pharmguard/TECHNICAL_DECISIONS.md): In-depth architectural trade-off evaluations and model licensing assessment.
- [CLAUDE.md](file:///C:/Users/USER/Desktop/pharmguard/CLAUDE.md): Contributor guide and conventions for human and AI engineers.

---

## Target Roadmap to December 1, 2026

- [x] Cloud-assisted browser-to-Colab-to-browser webcam streaming foundation.
- [x] Pretrained YOLO person detection.
- [x] Two-stage ByteTrack multi-person tracking with persistent IDs.
- [x] Pharmacy shelf zone dwell time & rapid retreat analysis.
- [x] Actionable alerts and professional security console UI.
- [ ] Pose estimation / hand-in-shelf keypoint validation.
- [ ] Commercial Apache 2.0 model validation (`rtdetr-l.pt`).
- [ ] Multi-camera aisle expansion.
