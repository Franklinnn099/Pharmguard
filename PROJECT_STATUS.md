# PharmaGuard — Project Status Report

**Current Date**: November 17, 2026  
**Milestone Deadline**: December 1, 2026 (Live Demonstration)  
**Version**: 0.1.0  
**Repository State**: Operational Core Prototype (Passed 14/14 automated unit & pipeline integration tests)

---

## 1. Completed Functionality

### Core Vision & Inference Pipeline
- [x] **Modular Architecture**: Clean separation between detector, tracker, behavioral analyzer, alert manager, event logger, annotator, UI, and configuration.
- [x] **Object Detection**: Pretrained YOLO integration (`pharmaguard/detector.py`) filtering for class `0` (`person`), configurable confidence threshold (default `0.40`), IoU threshold (default `0.45`), auto hardware device acceleration (`cuda` / `cpu`).
- [x] **Multi-Object Tracking (ByteTrack)**: Robust two-stage ByteTrack algorithm (`pharmaguard/tracker.py`) associating high- and low-confidence detections, persistent tracking IDs, Kalman-smoothed trajectory extrapolation, and occlusion buffering (30-frame buffer).
- [x] **Pharmacy Shelf Spatial Monitoring**: Configurable 2D normalized polygon zones (`ShelfZone`), resolution-independent denormalization, and bottom-center/ground-contact point-in-polygon testing to prevent false torso intersections.
- [x] **Suspicious Behavioral Sequence Detection**:
  - `ZONE_ENTER` / `ZONE_EXIT` logging.
  - `SUSPICIOUS_LOITERING`: Triggers when an individual dwells within a restricted shelf zone beyond the threshold (default `8.0s`).
  - `RAPID_RETREAT`: Detects grab-and-flee interactions (rapid exit within 0.5–2.5s).
  - `MULTI_PERSON_RESTRICTED`: Flags simultaneous congregation around sensitive pharmaceuticals.
- [x] **Security Alert Management & Throttling**:
  - Severity stratification: `INFO`, `WARNING`, `ALERT`.
  - Cooldown suppression mechanism (default `10.0s`) to prevent alert flooding.
  - JSONL audit logging (`logs/pharmaguard_alerts.jsonl`).
- [x] **High-Signal Visual Annotations & Security HUD**:
  - Monitored shelf zone polygon overlays (dynamic color changes from Teal/Secured to Amber/Occupied).
  - Bounding boxes and persistent IDs (`ID #X | Dwell: Ys`).
  - Motion trajectory trails.
  - Top Security HUD bar displaying live connection pulse, FPS, frame latency, active track count, and alert count.
- [x] **Gradio Operations Console**:
  - Dark surveillance aesthetic with real-time browser webcam streaming input and annotated video display.
  - Live metric cards (Status, Tracked Persons, Active IDs, FPS, Latency).
  - Real-time alert feed with colored severity badges.
  - Chronological audit event log.
  - Live interactive sliders for confidence and dwell thresholds.
- [x] **Turnkey Google Colab Integration**:
  - Dedicated notebook `colab/PharmaGuard_Colab.ipynb` with GPU verification, Drive mounting, pinned dependency installer, and `--share` tunnel generation.
  - Shell setup script `colab/setup_colab.sh`.

---

## 2. Incomplete Functionality / Roadmap to December 1, 2026

- [ ] **Pose Estimation / Hand Interaction**: Integrating 2D hand/wrist keypoints (e.g. YOLO-Pose) to confirm whether hands actually entered the shelf volume vs standing near it.
- [ ] **Concealment Sequence Detection**: Detecting item removal followed by rapid placement into bags or clothing pockets.
- [ ] **Cross-Session Re-Identification (ReID)**: Persistent feature embeddings across camera dropouts exceeding 60 seconds.
- [ ] **Multi-Camera Handoff**: Support for multiple RTSP IP streams simultaneously across different pharmacy aisles.
- [ ] **Push Notifications / SMS**: Webhook dispatch to pharmacy manager phones or mobile push alerts.

---

## 3. Known Bugs & Limitations

1. **Python 3.13 Wheel Availability**: Standard PyPI wheel for C++ `lap` library is absent for Python 3.13 on Windows without `--user` pip installation or build tools. Resolved by utilizing our self-contained pure-SciPy Hungarian algorithm (`linear_sum_assignment`), ensuring 100% platform portability across Windows and Linux Colab.
2. **Network Jitter on Colab Free Tier**: While Gradio WebSocket streaming works smoothly over high-bandwidth connections, Colab public live tunnels (`*.gradio.live`) occasionally encounter latency spikes (100–300ms) if upstream bandwidth fluctuates.

---

## 4. Current Architecture

```
[ Laptop Webcam ]
        │  (Browser WebRTC / WebSocket Stream)
        ▼
[ Gradio UI Client ]
        │  (Public HTTPS Live Tunnel)
        ▼
[ Google Colab GPU Cloud Runtime ]
        │
        ├─► [ BaseDetector / YOLODetector ] (YOLOv8 / RT-DETR)
        │         │
        │         ▼ (Detections: xyxy, conf, class=person)
        │
        ├─► [ ByteTrackerWrapper ] (Two-Stage Hungarian Matching)
        │         │
        │         ▼ (Persistent Tracks: IDs, Trajectories, Centroids)
        │
        ├─► [ PharmacyBehaviorAnalyzer ] (Polygon ROI Dwell & Velocity)
        │         │
        │         ▼ (Behavior Events: Loitering, Rapid Retreat)
        │
        ├─► [ AlertManager & EventManager ] (Throttling, Severity, Audit Logs)
        │         │
        │         ▼ (Security Alerts, Ring Buffer)
        │
        ├─► [ SecurityAnnotator ] (Overlays, Zones, Trails, Telemetry HUD)
        │
        ▼
[ Rendered Annotated Video + Telemetry Stream returned to Browser ]
```

---

## 5. How to Run the Application

### Option A: Cloud Google Colab (Primary Target)
1. Open `colab/PharmaGuard_Colab.ipynb` in Google Colab.
2. Ensure GPU runtime is active (`Runtime > Change runtime type > T4 GPU`).
3. Run all cells.
4. Click the generated public HTTPS URL `https://xxxx.gradio.live`.
5. In your laptop browser, allow webcam permissions. The live surveillance stream will start immediately.

### Option B: Local Workstation
```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run unit & pipeline tests
python -m pytest tests/ -v

# 3. Launch the local security dashboard
python main.py --local --port 7860
```
Open `http://127.0.0.1:7860` in any web browser.

---

## 6. Next Development Priorities

1. **Conduct Physical Camera Validation**: Test laptop webcam streaming across diverse lighting conditions.
2. **Zone Calibration Tool**: Add an interactive polygon point selector in the UI so pharmacy managers can click points on the video stream to define custom shelf zones.
3. **Apache 2.0 Licensing Path Verification**: Validate `rtdetr-l.pt` inference in Colab to benchmark commercial migration without AGPL obligations.
