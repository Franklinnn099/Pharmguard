# PharmaGuard — Technical Decisions & Architecture Rationale

## Document Purpose
This document records significant architectural decisions, technology selections, trade-off evaluations, model licensing assessments, and rejected alternatives for **PharmaGuard**.

---

## 1. Computer Vision Detection Backbone & Model Licensing Assessment

### Context
PharmaGuard requires real-time person detection from a live webcam feed. The target presentation deadline is December 1, 2026. Custom training is strictly ruled out for the initial milestone.

### Options Evaluated
1. **Ultralytics YOLO (YOLOv8 / YOLO11)**
   - *Pros*: SOTA accuracy, ultra-fast inference (TensorRT, ONNX, PyTorch), native integration with ByteTrack, automatic PyTorch device management, active community.
   - *Cons / Licensing*: Released under **AGPL-3.0**. Under AGPL-3.0, if PharmaGuard is offered as a network service (SaaS), the entire source code of the service must be made available to network users unless a commercial license is purchased from Ultralytics.
   - *Decision*: **Selected for initial research & December prototype**, with an architectural abstraction (`BaseDetector`) so the model backend can be decoupled.

2. **RT-DETR (Real-Time DEtection TRansformer / Baidu)**
   - *Pros*: **Apache 2.0 License** (permissive, enterprise-friendly, commercially clean). Eliminates NMS (Non-Maximum Suppression) bottlenecks, competitive accuracy. Supported natively via Ultralytics model loader (`YOLO("rtdetr-l.pt")`).
   - *Cons*: Slightly higher compute footprint on lower-tier GPUs compared to nano YOLO variants.
   - *Decision*: Designated as the primary commercial migration path.

3. **YOLOX (Megvii)**
   - *Pros*: **Apache 2.0 License**, anchor-free architecture.
   - *Cons*: Older codebase, more setup friction, slower development velocity in modern PyTorch ecosystems.

### Decision & Mitigations
We implement an abstract class `BaseDetector` in `pharmaguard/detector.py`. The initial implementation `YOLODetector` defaults to `yolov8n.pt` for maximum speed and simplicity in Colab, but allows swapping to Apache 2.0 `rtdetr-l.pt` without altering tracking, behavior analysis, or the user interface.

---

## 2. Multi-Object Tracking (MOT): ByteTrack Selection

### Context
We need to assign persistent IDs to people, maintain tracking through temporary occlusions, calculate dwell times, and avoid identity switching.

### Options Evaluated
1. **ByteTrack**
   - *Mechanism*: Associates almost every detection box (both high-confidence and low-confidence detections) using Kalman filtering and bipartite matching. Low-confidence detections recover occluded individuals without introducing false positives.
   - *Pros*: High FPS (runs purely on bounding box geometry without heavy deep appearance feature extraction), low latency, minimal identity switches. Available with MIT license via `supervision`.
   - *Decision*: **Selected as the default multi-object tracker.**
   - *History*: the prototype shipped a hand-written IoU tracker described as ByteTrack. It had no Kalman filter, its low-confidence stage could never run, and it switched IDs above ~55 px/frame of motion (audit F1-F4). Replaced on 2026-11-19, see §6.

2. **DeepSORT / BoT-SORT**
   - *Pros*: Uses deep ReID (Re-Identification) appearance embeddings.
   - *Cons*: ReID CNN feature extraction per bounding box adds 15–30 ms latency per frame on GPU and severely degrades Colab streaming throughput.
   - *Decision*: Rejected for the real-time prototype; may be revisited in future phases if cross-camera re-identification is required.

---

## 3. Streaming & Client-Server Architecture: Gradio Streaming

### Context
The client runs on a laptop with a webcam. The compute backend runs in Google Colab (with GPU). The browser must capture webcam frames, send them to Colab, and receive annotated frames with telemetry in real time.

### Options Evaluated
1. **Gradio with `gr.Image(sources=['webcam'], streaming=True)` + Public Live Tunnel**
   - *Mechanism*: Gradio runs a FastAPI/WebSocket server in Colab. `demo.launch(share=True)` provides an encrypted public HTTPS tunnel (`https://xxxx.gradio.live`). Modern browsers strictly require HTTPS to grant webcam access.
   - *Streaming Latency Optimization*: By default, Gradio throttles streaming to `stream_every=0.5` (2 FPS). We set `stream_every=0.04` (a minimum capture spacing, not a delivered rate - see §5), disabled per-frame progress micro-spinners (`show_progress="hidden"`), and enabled WebSocket connection queuing (`demo.queue(default_concurrency_limit=2)`).
   - *Tunnel Security*: Added secure session authentication via `demo.launch(auth=...)` resolving credentials from Colab Secrets, environment variables (`PHARMAGUARD_AUTH_USER`/`PHARMAGUARD_AUTH_PASS`), or ephemeral runtime tokens to ensure unauthorized web crawlers cannot access the live pharmacy surveillance stream. Hardcoded credentials are strictly prohibited.
   - *Pros*: Zero client installation required on the laptop—any browser can open the link, click "Allow Camera", and stream frames to Colab. Includes integrated UI components for telemetry and controls.
   - *Decision*: **Selected for the primary architecture.**

2. **Custom WebRTC (aiortc / Janus Gateway)**
   - *Pros*: Sub-100ms ultra-low latency UDP streaming.
   - *Cons*: High implementation complexity, ICE/STUN/TURN server requirements, fragile firewall traversal in Colab environments.
   - *Decision*: Deferred for future production iterations.

3. **Local-only OpenCV `cv2.VideoCapture(0)`**
   - *Pros*: Lowest latency.
   - *Cons*: Violates the cloud-assisted architecture requirement.
   - *Decision*: Implemented only as an optional local fallback/testing mode, not as the primary delivery vehicle.

---

## 4. Behavioral Sequence Analysis & Zone Representation

### Spatial Modeling
Pharmacy shelf zones are represented as **normalized 2D polygons** `[[x1, y1], [x2, y2], ...]`. Normalized coordinates ensure resolution independence (works identically whether the webcam outputs 480p, 720p, or 1080p).

### Footprint/Centroid Grounding
Rather than using the geometric center of the bounding box (which floats at torso height and can trigger false zone intersections), PharmaGuard evaluates the **bottom-center point** `((x1 + x2)/2, y2)` representing the individual's feet contacting the floor.

### Temporal State Tracking
Each tracked person maintains an accumulated dwell timer. When dwell time exceeds the configured threshold (default: 8.0s), an alert is triggered. To prevent alert storms, an alert cooldown mechanism (default: 10.0s) throttles repeat alerts for the same track ID.

---

## 5. Real-Time Transport Decisions (2026-11-19, measured)

All figures measured on the dev laptop (CPU only, Python 3.13, gradio 6.32.0, ultralytics 8.4.215). Colab GPU and
real-tunnel figures are still **pending a live run**.

### 5.1 How Gradio actually paces webcam streaming
Read from the shipped frontend bundle: the webcam component runs
`setInterval(() => { if (!pending) capture() }, stream_every)`. A new frame is captured **only when the previous
one has completed**, so frames never queue on the server and latency cannot grow without bound. Consequences:
- `stream_every=0.04` is a minimum spacing, **not** "25 FPS". Delivered FPS = 1 / (round-trip time).
- The spacing of frame **arrivals at the server** equals the full browser->server->browser cycle. `pharmaguard/telemetry.py`
  measures end-to-end FPS and latency **on the server** without client clock sync (valid for one streaming client).

### 5.2 Return-frame encoding: JPEG, not Gradio's default WebP
Gradio's `gr.Image.postprocess` cost per annotated frame:

| Format | 1280x720 | 640x360 |
|---|---|---|
| WebP (Gradio default) | 527 ms | 112 ms |
| PNG | 219 ms | 55 ms |
| **JPEG (chosen)** | **13 ms** | **6 ms** |

### 5.3 Capture size: request 640x480 from the browser, downscale on the server
With no constraints, Gradio requests **1920x1440** from `getUserMedia` and uploads each frame as JPEG q0.8. We pass
`WebcamOptions(constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}})` and additionally
downscale anything wider than `stream.max_process_width` (default 640) server-side. Zones are normalised,
so behaviour logic is resolution-independent.

### 5.4 Image-only per-frame event; panels on a 1 Hz timer
The stream event previously returned 8 outputs including two HTML blobs every frame. It now returns only the
annotated image. Telemetry/alert panels refresh from a `gr.Timer(1.0)`, and unchanged HTML is skipped (`gr.skip()`).

### 5.5 Measured effect (localhost, CPU, real photo with 4 people, 1280x720 uploaded frames)

| Configuration | E2E p50 | E2E p95 | Sustained FPS |
|---|---|---|---|
| Previous path (WebP, no downscale) | 553 ms | 650 ms | 1.8 |
| New defaults (JPEG, downscale to 640) | **190 ms** | **215 ms** | **5.3** |

Measured with `tools/e2e_benchmark.py`, which reproduces the browser's four HTTP exchanges per frame
(upload, submit, result stream, download) on one persistent connection. `gradio_client` was **rejected** as a
measurement tool: it opens new connections per call, and each new `httpx` connection costs ~640 ms on the
dev laptop (pooled: 3 ms), which inflated round trips to ~2.6 s that no browser pays.

### 5.6 Privacy defaults
`GRADIO_ANALYTICS_ENABLED=False` is set before Gradio is imported: at startup Gradio was observed contacting
`huggingface.co` and `api.gradio.app`. The `gradio.live` share relay still carries video through third-party
infrastructure, so it is restricted to staged footage of consenting participants.

### 5.7 Dependency pins
`gradio==6.32.0` and `ultralytics==8.4.215` are pinned exactly to the tested set. The previous pins
(`gradio<=5.6.0`, `ultralytics<=8.3.40`) had never been installed or tested together. `supervision`/`lap` were
removed (unused) and `scipy` was added (imported directly). torch is left unpinned so Colab keeps its CUDA build.

---

## 6. Tracker Rewrite: ByteTrack + Kalman + Time-Aware Motion Gate (2026-11-19)

### Decision
`pharmaguard/tracker.py` was rewritten as a faithful ByteTrack, keeping the `TrackedPerson` / `TrackingResult` interface,
so behaviour, annotator and UI did not change.

### Why not Ultralytics' built-in `model.track()`?
- It fuses detection and tracking into one call, which breaks the `BaseDetector` abstraction needed for the Apache-2.0 detector swap.
- Its internals (`ultralytics.trackers.*`) have changed across releases; we pin Ultralytics exactly and still want a tracker we can test and reason about.
- We need two deviations it does not offer: time-scaled prediction and a motion fallback gate for low tunnel FPS.

### Algorithm (deviations from the paper in bold)
1. Constant-velocity Kalman filter on (cx, cy, aspect, h), with DeepSORT/ByteTrack noise weights. **The step is scaled by the real time between frames** (`dt * frame_rate`), because tunnel delivery is irregular (3-25 FPS).
2. Stage 1: tracked and lost tracks against detections with conf >= 0.45, Hungarian assignment on score-fused IoU, IoU >= 0.20.
3. Stage 2: still-unmatched tracked tracks against detections with conf in [0.10, 0.45), IoU >= 0.50. **The detector floor was lowered from 0.40 to 0.10 so this stage receives anything**; the operator slider now governs tracker thresholds instead.
4. Candidates need **2 consecutive detections before an ID is issued**, so one-frame false positives never consume IDs.
5. **Lost tracks expire after 2.0 s of real time** (previously 30 frames, i.e. anything from 1 s to 10 s depending on FPS). **Lost-track velocity decays by 0.85 per nominal frame**, so a person who stops behind an occluder is not extrapolated off-screen. Occluded tracks are drawn as a dashed grey box at the predicted position.
6. **Motion fallback gate** (stages 1 and 3 only): if IoU fails, a match is still accepted when the centre moved <= `2.0 body-heights/s * elapsed + 0.1` and box heights are within 1.35x. Fallback pairs always rank below IoU-qualified pairs. Reason: at 3 FPS a walking person (0.8 h/s) moves ~0.27 h per frame; a 0.4 h-wide box then has IoU ~0.2, so pure IoU gating loses them. This is verified by `test_motion_fallback_disabled_reproduces_pure_bytetrack_limit`.

### Measured on ground-truth scenarios (`tests/scenario_utils.py`), old vs new
"sw" = ID switches, "swap" = identities exchanged between people, "cov" = share of frames where a visible person carried an ID.

| Scenario | Old sw / swap / cov | New sw / swap / cov |
|---|---|---|
| walk 0.8 h/s @ 3 FPS | 11 / 0 / 100% | 0 / 0 / 92% |
| walk 1.2 h/s @ 5 FPS | 19 / 0 / 100% | 0 / 0 / 95% |
| walk 1.2 h/s @ 10 FPS | 0 / 0 / 100% | 0 / 0 / 98% |
| irregular 50-400 ms frame gaps | 0 / 0 / 100% | 0 / 0 / 96% |
| 2 crossing, hidden one at low conf | 2 / 1 / 98% | 0 / 0 / 98% |
| 2 crossing, hidden one undetected | 1 / 0 / 100% | 0 / 0 / 98% |
| side-by-side walkers @ 3 FPS | 16 / 8 / 100% | 0 / 0 / 83% |
| partial occlusion, conf 0.2 for 1.5 s | 1 / 0 / 62% | 0 / 0 / 98% |
| full occlusion 1 s | 1 / 0 / 100% | 0 / 0 / 98% |
| 5-person crowd | 0 / 0 / 100% | 0 / 0 / 98% |
| 1-frame false positive | 1 ID issued | 0 IDs issued |

The new tracker's lower coverage is the confirmation delay: a new person's first frame has no ID (0.33 s at 3 FPS).
Tracker cost: 1.3 ms/frame on the dev CPU with 4 people. Real-detector check: a panned photo of 3-4 people
yields one stable ID per person at a simulated 5 FPS (`tools/track_video.py`,
`test_tracker_with_real_detector_on_panning_photo`).

### Known limits
- No appearance re-identification. Someone hidden for more than 2 s, or who moves far while hidden, gets a new ID. This is acceptable for single-camera dwell and interaction logic; DeepSORT-style ReID was rejected for latency (§2).
- Synthetic scenarios prove the logic, not real-world robustness. Staged webcam recordings of crossings and occlusion must be run through `tools/track_video.py` before Dec 1.
