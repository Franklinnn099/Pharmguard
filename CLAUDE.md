# PharmaGuard — Engineering & Contributor Guide (CLAUDE.md)

Welcome to **PharmaGuard**. This file establishes development guidelines, architectural rules, coding standards, commands, and workflows for human engineers and AI assistants (including Claude Code and Antigravity).

---

## 1. Project Overview & Philosophy

PharmaGuard is an AI-assisted real-time pharmacy surveillance prototype designed to detect suspicious interaction sequences around sensitive medicine shelves.
- **Deadline**: December 1, 2026.
- **Deliverable**: Functioning real-time demonstration streaming from laptop webcam to Google Colab GPU and back.
- **Key Principle**: *Reliability & genuine functioning over complex mock code.* Never fabricate test results or pretend mock detections are real AI outputs.

---

## 2. Repository Structure

```
pharmguard/
├── .gitignore                      # Python, models, caches, and logs exclusions
├── CLAUDE.md                       # Engineering guidelines & contributor instructions
├── PROJECT_STATUS.md               # Up-to-date tracking of completed & incomplete tasks
├── TECHNICAL_DECISIONS.md          # Architectural choices, licensing, & trade-off records
├── README.md                       # Public project documentation & quickstart
├── requirements.txt                # Pinned dependencies
├── main.py                         # Root entry point
├── benchmark.py                    # Server-side compute benchmark (incl. return-frame encode)
├── docs/AUDIT_2026-11-19.md        # Takeover audit: verified defects, measurements, risks
├── tools/e2e_benchmark.py          # Browser-equivalent HTTP round-trip benchmark (run from laptop)
│
├── pharmaguard/                    # Core Python package
│   ├── __init__.py                 # Package version declaration
│   ├── config.py                   # Pydantic schemas (detector, tracker, zones, alerts, UI)
│   ├── detector.py                 # BaseDetector interface & YOLODetector implementation
│   ├── tracker.py                  # IoU tracker, ByteTrack-style (no Kalman yet; see audit F1/F2)
│   ├── behavior.py                 # PharmacyBehaviorAnalyzer (dwell, loitering, rapid retreat)
│   ├── alerts.py                   # AlertManager (throttling, severity grading, JSONL logging)
│   ├── events.py                   # EventManager (chronological ring buffer)
│   ├── annotator.py                # SecurityAnnotator (shelf overlays, tracks, trails, HUD)
│   ├── pipeline.py                 # Per-frame orchestration, downscale, stage timings
│   ├── telemetry.py                # Server-measured end-to-end FPS/latency + signal state
│   ├── ui.py                       # Gradio Security Operations Dashboard
│   └── app.py                      # CLI launcher with local & Colab presets
│
├── colab/                          # Google Colab execution artifacts
│   ├── PharmaGuard_Colab.ipynb     # Turnkey Colab notebook with live tunnel (no Drive mount yet)
│   └── setup_colab.sh              # Shell script for automated Colab environment bootstrap
│
└── tests/                          # Automated test suite
    ├── test_config.py              # Configuration & YAML serialization tests
    ├── test_detector.py            # Pretrained YOLO inference & empty-frame tests
    ├── test_tracker.py             # ByteTrack ID consistency & recovery tests
    ├── test_behavior.py            # Spatial ROI, dwell time, & rapid retreat tests
    ├── test_alerts.py              # Cooldown throttling & persistence tests
    ├── test_pipeline.py            # End-to-end vision pipeline integration tests
    ├── test_live_streaming_integration.py  # UI handlers, transport config, live server
    ├── test_stream_pipeline.py     # Telemetry, downscale, real-image inference, HTTP round trip
    └── conftest.py                 # Redirects alert log so tests never touch logs/
```

---

## 3. Standard Commands

### Environment Setup
```powershell
pip install -r requirements.txt
```

### Running Tests
Always run the test suite before committing changes:
```powershell
python -m pytest tests/ -v
```

### Running Pipeline Benchmark
```powershell
python benchmark.py
```

### Launching the Application
- **Local Workstation Mode**:
  ```powershell
  python main.py --local --port 7860
  ```
- **Cloud / Colab Mode (Public HTTPS Tunnel)**:
  ```powershell
  python main.py --colab --model yolov8n.pt
  ```

---

## 4. Coding Conventions & Constraints

1. **Modular Architecture**: Keep functional boundaries clean. Do NOT combine UI, tracking, and detection logic into a single file.
2. **Abstract Detectors**: All detection backends must inherit from `BaseDetector` in `pharmaguard/detector.py`. This ensures we can switch to Apache 2.0 models (e.g. RT-DETR) without modifying downstream trackers.
3. **Normalized Coordinates**: Always represent visual zones and regions of interest (ROI) in normalized floats `[0.0, 1.0]` so they remain resolution-independent across varying webcam cameras.
4. **Feet Ground Contact for Spatial Intersections**: Test spatial zone occupancy against the bottom center of the bounding box `((x1+x2)/2, y2)` rather than the geometric center to prevent false hits when a person's upper body leans across a boundary. *Applies to full-body CCTV views only.* With the laptop webcam facing a mock shelf, feet are usually out of frame, so shelf interaction must be judged from hand/wrist positions (Priority 3).
5. **Alert Throttling**: Never emit unthrottled alert streams. Every security alert candidate must pass through `AlertManager.process_behavior_events()` with an enforceable cooldown window.
6. **No Silent Changes**: Whenever schemas, thresholds, or dependencies change, update `PROJECT_STATUS.md` and `TECHNICAL_DECISIONS.md`.
7. **Type Annotations**: Use Python type hints throughout all modules. Validate dataclasses or Pydantic models for configuration.
8. **Real-time transport rules (measured, see TECHNICAL_DECISIONS §3)**: return frames as JPEG (never Gradio's default WebP), request a 640x480 webcam capture, downscale server-side, and keep the per-frame stream event image-only (panels refresh via a 1 Hz timer).
9. **Measurement honesty**: never print or document a metric that was not measured (e.g. hard-coded "dropped frames"). Label compute-only vs end-to-end figures explicitly.
10. **Tests must not touch operational data**: `tests/conftest.py` redirects `PHARMAGUARD_ALERT_LOG`.

---

## 5. Development Workflow for Claude Code

1. Read `PROJECT_STATUS.md` and `TECHNICAL_DECISIONS.md` before planning any new feature.
2. Formulate an atomic checklist of tasks.
3. Implement changes in the corresponding modular component under `pharmaguard/`.
4. Add or update unit tests in `tests/`.
5. Execute `python -m pytest tests/ -v` and ensure all tests pass.
6. Update `PROJECT_STATUS.md` with true, verified progress.
