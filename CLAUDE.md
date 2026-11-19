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
├── docs/DEMO_STAGING.md            # Camera placement, zone setup, staged scenario protocol
├── configs/demo_shelf.yaml         # Demo zones/thresholds (pass with --config)
├── tools/e2e_benchmark.py          # Browser-equivalent HTTP round-trip benchmark (run from laptop)
├── tools/track_video.py            # Run pipeline on recorded video (video time); tracking report + annotated MP4
│
├── pharmaguard/                    # Core Python package
│   ├── __init__.py                 # Package version declaration
│   ├── config.py                   # Pydantic schemas (detector, tracker, zones, alerts, UI)
│   ├── detector.py                 # BaseDetector interface & YOLODetector implementation
│   ├── tracker.py                  # ByteTrack + time-scaled Kalman + motion fallback gate
│   ├── interaction.py              # Per-frame hand geometry: wrist in shelf / pocket / bag (stateless)
│   ├── behavior.py                 # Temporal sequences, decaying suspicion score, incident latching
│   ├── alerts.py                   # AlertManager (throttling, severity grading, JSONL logging)
│   ├── alarm.py                    # AlarmController: sound once per incident, ack/reset/test, audit
│   ├── storage.py                  # SQLite EventStore (events, audit, review, retention, Drive mirror)
│   ├── evidence.py                 # Ring buffer + background clip encoder (ffmpeg/avc1/VP8 fallbacks)
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
    ├── test_tracker.py             # Ground-truth scenarios: speed/FPS sweep, crossing, occlusion, swaps
    ├── scenario_utils.py           # Synthetic scenario builders + identity evaluation
    ├── test_behavior.py            # Mandate §6 scenarios on scripted skeletons (false-alert & miss checks)
    ├── behavior_scenarios.py       # Skeleton actor scripts for behaviour tests
    ├── test_alerts.py              # Cooldown throttling & persistence tests
    ├── test_alarm.py               # Alarm state machine, dedupe, ack/reset/test, pipeline integration
    ├── test_recording.py           # Event store, clips, retention, mirroring, pipeline integration
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
4. **Hands, not feet, for shelf interaction**: a shelf contact is a wrist keypoint inside a shelf polygon AND outside the person's own torso box. The old feet-point rule applies only to full-body CCTV views and is not used by the behaviour engine.
5. **Alert discipline**: alerts come only from the behaviour engine's per-incident latch (one REVIEW, one ALERT per incident), then pass through `AlertManager.process_behavior_events()` cooldowns as a safety net. Browsing evidence is capped below the review threshold, so never raise that cap without staged-footage evidence. Wording is 'for staff review', never theft or intent.
6. **No Silent Changes**: Whenever schemas, thresholds, or dependencies change, update `PROJECT_STATUS.md` and `TECHNICAL_DECISIONS.md`.
7. **Type Annotations**: Use Python type hints throughout all modules. Validate dataclasses or Pydantic models for configuration.
8. **Real-time transport rules (measured, see TECHNICAL_DECISIONS §3)**: return frames as JPEG (never Gradio's default WebP), request a 640x480 webcam capture, downscale server-side, and keep the per-frame stream event image-only (panels refresh via a 1 Hz timer).
9. **Measurement honesty**: never print or document a metric that was not measured (e.g. hard-coded "dropped frames"). Label compute-only vs end-to-end figures explicitly.
10. **Tests must not touch operational data**: `tests/conftest.py` redirects `PHARMAGUARD_ALERT_LOG` and `PHARMAGUARD_DATA_DIR`. Never commit `data/` (event DB and clips contain people's faces).
11. **Never open SQLite on a Drive/FUSE mount**: write locally and use `EventStore.mirror()` (backup API).

---

## 5. Development Workflow for Claude Code

1. Read `PROJECT_STATUS.md` and `TECHNICAL_DECISIONS.md` before planning any new feature.
2. Formulate an atomic checklist of tasks.
3. Implement changes in the corresponding modular component under `pharmaguard/`.
4. Add or update unit tests in `tests/`.
5. Execute `python -m pytest tests/ -v` and ensure all tests pass.
6. Update `PROJECT_STATUS.md` with true, verified progress.
