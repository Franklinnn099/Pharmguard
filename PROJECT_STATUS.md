# PharmaGuard — Project Status Report

**Current Date**: November 17, 2026  
**Milestone Target**: December 1, 2026 (Live Demonstration)  
**Version**: 0.1.1  
**Test Suite Status**: 19/19 Passing Automated Tests (14 core unit + 5 live streaming integration tests)

---

## 1. Verified vs. Unverified Capabilities

### Genuinely Verified & Tested (Pass 19/19)
- [x] **Webcam Ingestion & Stream Disconnect Recovery**: Verified that the Gradio stream callback consumes RGB/RGBA webcam frames, handles `None` input during stream pause/disconnection without crashing, and smoothly resumes upon reconnection.
- [x] **Multi-Person Tracking Consistency**: Verified that multiple distinct individuals receive unique tracking IDs (e.g. ID #1 and ID #2) and maintain persistent identities during movement without identity switches.
- [x] **Shelf Zone Spatial Interaction**: Verified 2D normalized polygon boundaries (`Prescription_Shelf_A`) and bottom-center ground contact points for accurate floor projection.
- [x] **Behavioral Sequence & Dwell Time Calculation**: Verified `ZONE_ENTER`, dwell time accumulation, `SUSPICIOUS_LOITERING` alert emission when dwell threshold (default 8.0s) is exceeded, and `RAPID_RETREAT` alert emission for abrupt exits (0.5–2.5s).
- [x] **Alert Cooldown Throttling**: Verified that repeat alerts for the same track ID within the cooldown window (default 10.0s) are suppressed to avoid operator flooding, and logged to JSONL.
- [x] **Visual Annotations & Top HUD**: Verified that shelf polygons, bounding boxes, dwell badges, motion trails, and real-time operational telemetry (FPS, Latency, Connection Status, Track Count) are rendered accurately into RGB frames.
- [x] **Fine-Grained Stage Latency Instrumentation**: Pipeline output measures and returns individual stage latencies (`detect_ms`, `track_ms`, `behavior_ms`, `annotate_ms`, `total_backend_ms`).
- [x] **Gradio Low-Latency Streaming Configuration**: Fixed the default Gradio 0.5s (2 FPS) bottleneck by configuring `stream_every=0.04` (~25 FPS), `show_progress="hidden"`, and enabling WebSocket queueing via `demo.queue()`.

### Awaiting Live User Colab Runtime Execution
- [ ] **Physical Colab GPU Webcam Session**: The code and turnkey notebook `colab/PharmaGuard_Colab.ipynb` are prepared for the user to execute on Colab GPU with their physical laptop webcam. Actual GPU latency (expected 15–25ms on NVIDIA T4/L4) must be recorded during the live Colab run.

### Incomplete / Future Product Roadmap
- [ ] **Pose / Wrist Keypoint Estimation**: Validating physical reaching into shelf volume using YOLO-Pose keypoints.
- [ ] **Concealment Recognition**: Detection of pocketing or bagging sequences.
- [ ] **Multi-Camera Aisle Hand-off**: Cross-camera re-identification.

---

## 2. Issues Discovered & Fixed During Audit

1. **Gradio Streaming Rate Bottleneck**: `gr.Image.stream()` defaulted to `stream_every=0.5` (2 FPS) with progress spinners. Fixed by explicitly setting `stream_every=0.04` (~25 FPS), `show_progress="hidden"`, and `concurrency_limit=1`.
2. **WebSocket Queueing**: Added `demo.queue(default_concurrency_limit=2)` to `app.py` to prevent HTTP polling stalls during video streaming.
3. **RGBA 4-Channel Stream Compatibility**: Browser canvas and WebRTC streams occasionally deliver 4-channel RGBA numpy arrays. Added an automatic format converter in `pipeline.py` to normalize RGBA/Grayscale to 3-channel RGB.
4. **Stream Pause / Disconnection Handling**: When webcam is stopped or paused in the browser, Gradio sends `None`. Fixed `process_webcam_stream` in `ui.py` to return a clean `STANDBY` state with zero active tracks, preventing crash or stale state.
5. **Dashboard Security / Authentication**: Added optional basic authentication (`--auth`, default `admin:pharmguard2026`) in `app.py` to secure the public HTTPS Gradio link.
6. **Colab Directory & Dependency Robustness**: Updated `colab/PharmaGuard_Colab.ipynb` and `colab/setup_colab.sh` to handle workspace paths dynamically and ensure pre-cached weights and clean imports.

---

## 3. Actual Benchmark Measurements

Measured via `benchmark.py` across 50 frames (640x480 resolution):

| Metric | Measured Value (Host CPU) | Projected Colab GPU (NVIDIA T4/L4) |
| :--- | :--- | :--- |
| **Stage 1: YOLO Person Detection** | 105.54 ms | ~5–8 ms |
| **Stage 2: ByteTrack MOT** | 0.01 ms | <0.05 ms |
| **Stage 3: Shelf Behavior & Dwell** | 0.02 ms | <0.05 ms |
| **Stage 4: Visual HUD & Overlays** | 1.92 ms | ~1.5 ms |
| **Total Mean Backend Latency** | **107.86 ms** | **~7–10 ms** |
| **P95 Backend Latency** | 118.88 ms | ~12–15 ms |
| **Backend Processing Throughput** | 9.3 FPS (CPU limited) | >80 FPS (Engine capable) |
| **Target Browser Streaming Rate** | ~25 FPS (`stream_every=0.04`) | ~25 FPS (`stream_every=0.04`) |
| **Dropped Frames** | 0 / 50 | 0 |

---

## 4. Current Architecture

```
[ Laptop Webcam ]
        │  (Browser WebRTC / WebSocket Stream at 25 FPS)
        ▼
[ Gradio UI Client ]
        │  (Encrypted HTTPS Live Tunnel: https://xxxx.gradio.live)
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

### Google Colab GPU Execution (Live Cloud Demonstration)
1. Open `colab/PharmaGuard_Colab.ipynb` in [Google Colab](https://colab.research.google.com).
2. Set runtime to GPU: **Runtime > Change runtime type > T4 GPU**.
3. Run all cells in order.
4. Click the public **`https://xxxx.gradio.live`** link generated in Step 6.
5. In your laptop browser, click **Webcam** on the left panel and click **Allow**.
6. Live surveillance starts streaming immediately with GPU inference!

### Local Workstation Execution
```powershell
python -m pytest tests/ -v
python main.py --local --port 7860
```
Open `http://127.0.0.1:7860` in any web browser.
