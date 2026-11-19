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
   - *Implementation status (audit 2026-11-19, corrected)*: the shipped tracker is a **hand-written IoU tracker**, not ByteTrack via `supervision` (never imported) and has **no Kalman filter**. Its low-confidence second stage is unreachable because the detector's 0.40 confidence floor is above the tracker's 0.35 split. It also switches IDs once per-frame motion exceeds ~55 px. Fixing this is Priority 2.

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
