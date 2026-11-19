# 🛡️ PharmaGuard — AI Pharmacy Security Monitor

> Real-time suspicious-interaction monitoring system designed for pharmacies. Captures live webcam feeds, tracks individuals with persistent IDs, monitors sensitive medicine shelves, flags suspicious behavioral sequences, and presents actionable telemetry to security staff.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
> **Licence note:** no project licence has been chosen yet. Ultralytics YOLO is AGPL-3.0 (see TECHNICAL_DECISIONS.md §1).
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
- **Multi-Object Tracking**: ByteTrack with a time-scaled Kalman filter, low-confidence occlusion recovery, 2-frame ID confirmation and a motion gate for low tunnel frame rates (TECHNICAL_DECISIONS.md §6).
- **Configurable Shelf Zones**: Resolution-independent normalized 2D polygons representing monitored pharmacy shelves (e.g. narcotics or restricted medicine racks).
- **Hand-Based Interaction Analysis**: YOLOv8-pose wrist keypoints detect hands reaching into a configured shelf zone, then moving to the person's waist/pocket region or a detected bag within seconds.
- **Suspicion Scoring for Staff Review**: decaying, weighted evidence; browsing alone can never alert; REVIEW (visual) and ALERT (alarm) fire once per incident with human-readable reasons. The system flags interactions for review and does not determine theft or intent.
- **Operator Alarm**: browser siren on ALERT only (once per incident), visual banner and flashing frame border, Acknowledge / Reset with operator audit trail, and a manual Test alarm button. REVIEW warnings are visual only.
- **Event Recording**: every REVIEW/ALERT stored in SQLite (camera, track, incident, score, reasons, acknowledgement, staff review outcome) with a 10 s annotated evidence clip; mirrored to Google Drive in Colab; 7-day retention.
- **Event Review & Zone Editor**: review tab with clip playback and staff verdicts (confirmed / false alarm), which produce a measured false-alarm rate; draw shelf zones on a live snapshot and apply them without restarting.
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

### Option 2: Running Locally on Windows / macOS / Linux (CPU works; ~5 FPS measured on a laptop CPU)

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

The automated test suite covers each pipeline stage, the Gradio handlers, and a real HTTP frame round trip through a live local server:

```powershell
python -m pytest tests/ -v
```

Current results and measured performance are tracked in [PROJECT_STATUS.md](PROJECT_STATUS.md).

To benchmark:
```powershell
python benchmark.py                                   # server compute, incl. return-frame encoding
python tools/e2e_benchmark.py --url http://127.0.0.1:7860   # browser-equivalent HTTP round trip
```

---

## Project Documentation

- [PROJECT_STATUS.md](PROJECT_STATUS.md): Complete list of completed features, incomplete features, known issues, and next priorities.
- [TECHNICAL_DECISIONS.md](TECHNICAL_DECISIONS.md): In-depth architectural trade-off evaluations and model licensing assessment.
- [CLAUDE.md](CLAUDE.md): Contributor guide and conventions for human and AI engineers.

---

## Target Roadmap to December 1, 2026

- [x] Browser-to-server-to-browser streaming verified locally; Colab GPU run pending.
- [x] Pretrained YOLO person detection.
- [x] Robust multi-person tracking (ByteTrack + Kalman, occlusion, crossing): verified on synthetic ground truth; staged real-video check pending.
- [x] Shelf interaction analysis (hands, bag/pocket, scoring): verified on scripted scenarios; staged real-footage rates pending (docs/DEMO_STAGING.md).
- [x] Actionable alerts and professional security console UI.
- [ ] Commercial Apache 2.0 model validation (`rtdetr-l.pt`).
- [ ] Multi-camera aisle expansion.
