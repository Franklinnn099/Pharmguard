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
   - *Streaming Latency Optimization*: By default, Gradio throttles streaming to `stream_every=0.5` (2 FPS). We explicitly tuned `stream_every=0.04` (~25 FPS), disabled per-frame progress micro-spinners (`show_progress="hidden"`), and enabled WebSocket connection queuing (`demo.queue(default_concurrency_limit=2)`).
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
