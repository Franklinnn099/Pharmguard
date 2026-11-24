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

*Superseded on 2026-11-19 by §7.* The prototype's feet-in-floor-zone rules (dwell >= 8 s, "rapid retreat",
"congregation") were replaced. Measured in the audit (B1-B4): they could not fire for a webcam facing a shelf,
and they alerted on ordinary shoppers.

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

---

## 7. Behaviour Engine: Hand-Based Sequences + Decaying Suspicion Score (2026-11-19)

### Context
Demo setup (decided by the project owner): a laptop webcam facing a mock shelf, with people standing in front of it.
Feet are often out of frame and irrelevant; what matters is **what hands do**. No custom training is allowed.

### Signals (all from pretrained models)
- **YOLOv8n-pose** replaces YOLOv8n as the person model. One pass gives person boxes plus 17 COCO keypoints;
  CPU detect cost was roughly unchanged in the benchmark (pose 92 ms p50 vs 82-94 ms plain). Wrists, hips and shoulders are used.
- **YOLOv8n (COCO)** every 3rd frame for bags (backpack 24, handbag 26, suitcase 28). It costs ~99 ms on CPU when
  it runs and is disabled with `objects.enabled: false`.

### Geometry (`pharmaguard/interaction.py`, stateless)
- **Shelf contact**: wrist inside a shelf polygon **and outside the person's own torso box**. The shelf is behind
  the person in the image, so without this exclusion hands held at chest height "touch" the shelf
  (`test_geometry_hands_at_chest_are_not_shelf_contact_even_though_shelf_is_behind`).
- **Pocket**: wrist within 0.30 torso-lengths of a visible hip keypoint and not above the hip line. If the hips are
  not visible, the pocket test is skipped rather than guessed.
- **Bag**: wrist inside a detected bag box (expanded 10%) that overlaps the person's box.

### Temporal logic (`pharmaguard/behavior.py`)
- Contacts are debounced (0.3 s in and out). A **transfer** requires the SAME hand to reach the pocket (held 0.6 s) or a
  bag (held 0.3 s) within **3 s** of leaving the shelf. A slow transfer after examining is ordinary handling.
- **Return cancels pocket**: if the same hand touches the shelf again within 15 s, earlier pocket evidence is removed
  (item examined at the waist and put back). Bag evidence is not cancelled.
- **Exit**: moving away from the shelf for 1.5 s, or leaving view, within 8 s of a transfer adds evidence.
- **Score**: weighted evidence with a 30 s half-life. Weights: contact 0.10, repeated contacts 0.15, prolonged presence 0.15,
  pocket 0.45, bag 0.70, exit 0.30. **Browsing evidence is capped at 0.35, below REVIEW 0.60**, so browsing alone
  can never trigger anything.
- **Levels and latching**: REVIEW (visual) at 0.60, ALERT (alarm) at 1.00. Each fires once per incident. An incident closes when
  its **concealment** evidence decays below 0.25; browsing is ignored here. Found by test: with the total score, a person
  who kept browsing after an alert stayed latched and a new concealment minutes later was silently swallowed.
- Typical outcomes: shelf->bag->exit = 1.10 (REVIEW at the transfer, ALERT on exit); shelf->pocket->exit = 0.85 (REVIEW only).
- All timing uses frame timestamps, so recorded clips give exactly the live behaviour, and AlertManager cooldowns use frame time too.

### Verified (scripted skeleton scenarios, `tests/test_behavior.py`)
Walk past, browse with 5 contacts, a long browse, pick-up and return via the hip, reaching your own bag, slow transfer,
and hidden hands all produce **no REVIEW/ALERT**. Shelf->bag->leave gives one REVIEW plus one ALERT in the same incident.
Shelf->pocket->leave gives REVIEW only. With two people, only the concealer is flagged. A repeat concealment while latched
raises no new alert; after decay, a new incident raises a new alert. The full pipeline (scripted pose detector -> tracker ->
behaviour -> AlertManager) produces exactly `[WARNING, ALERT]`.

### Limitations (must be stated in the demo)
- Synthetic skeletons validate the logic, not real-world rates. **False-alert and miss rates on real staged footage
  are not yet measured**; see docs/DEMO_STAGING.md for the protocol.
- On a real photo, a relaxed hand resting at the hip is geometrically "at pocket". It is harmless unless it occurs within
  3 s after a shelf contact by the same hand, but it is the weakest signal and `pocket_radius_torso` needs calibration.
- The system cannot see products. It infers from hand trajectories, so palming without visible arm motion is missed.
- Camera geometry matters: the reaching arm must be visible (camera ~45 degrees to the side).

---

## 8. Alarm System (2026-11-19)

### State machine (`pharmaguard/alarm.py`, server-side, shared by all dashboards)
`IDLE -> ACTIVE (sounding) -> ACKNOWLEDGED -> IDLE` via ALERT / Acknowledge / Reset, plus `TEST` for the manual check.
- **Only ALERT sounds.** REVIEW (WARNING) shows an amber, visual-only banner for 20 s.
- **Duplicate prevention, three layers:** (1) the behaviour engine latches one ALERT per incident; (2) AlertManager cooldown
  (frame time); (3) AlarmController keys on `(tracker_id, incident_id)`, so an incident can sound **once, ever**, even after reset.
  A *new* incident during an alarm joins the active list and restarts the siren so the operator hears it.
- **Acknowledge** silences the alarm and records the operator (Gradio auth username) and time. Alerts stay on screen until **Reset**.
  Reset of an unacknowledged alarm is recorded as such.
- **Auto-silence** after 60 s if unacknowledged; the banner stays (state remains ACTIVE, marked "timeout").
- **Test alarm** sounds a distinct tone for 5 s and is never recorded as a security alert. It is refused while a real alarm is active
  (a test must never mask a real alarm), and a real alarm arriving during a test takes over.
- Every transition (raised / acknowledged / reset / test) goes to the event timeline and is appended to the audit JSONL.

### Audio delivery
- The siren is synthesised in the browser with the Web Audio API (no audio asset, no extra transfer).
- The server renders state into `<div id="pg-alarm" data-sounding data-seq data-kind data-volume>`; a page script polls it
  every 250 ms. A dedicated 0.5 s `gr.Timer` refreshes only this banner (skipped when unchanged), so the alarm reaches the
  browser within about 0.5-0.75 s plus network.
- **Browser autoplay policy:** audio is blocked until the user interacts with the page. The page shows
  "Alarm sound: LOCKED ... click anywhere" until the first click/keypress (starting the webcam counts), then "ready".
  The alarm never fails silently without saying so.
- The annotated video frame also gets a flashing red border and an "ALARM - REVIEW NOW" label (drawn after the HUD),
  so the alarm is visible even when the panel is scrolled away or the sound is muted.

### Verified
- 16 tests (`tests/test_alarm.py`): sounding rules, once-per-incident, ack/reset/auto-silence, test-mode rules,
  banner escaping, handlers, and a full pipeline run (scripted shelf->bag->leave) that sounds exactly once, shows the REVIEW
  banner first, draws the border, and persists the acknowledgement.
- **Real browser (Chrome, 2026-11-19, local server):** the banner keeps its `id`/`data-*` attributes after Gradio rendering; the
  script reports the AudioContext "suspended" before interaction and "running" after the click on *Test alarm*; the siren
  played (`playing: true`) while state was TEST and stopped when the server returned to IDLE after 5 s; the action was
  written to the audit log. The real-ALERT browser path uses the same code with `kind=ALERT` (covered by unit tests,
  not separately clicked through).

---

## 9. Event Recording: SQLite Store + Evidence Clips (2026-11-19)

### Store (`pharmaguard/storage.py`)
- **SQLite (stdlib)**, chosen over PostgreSQL/files: zero setup, a single file, transactional, and enough for one camera with
  rare events. Tables `events` (one row per REVIEW/ALERT) and `audit` (every alarm/review action).
- Event fields: `event_uid` (= alert id, UNIQUE, so inserts are idempotent), wall-clock time, **camera_id**, **tracker_id**,
  incident_id, **category** (SUSPICION_ALERT / SUSPICION_REVIEW), severity, zone, **score**, **reasons** (JSON),
  **clip_path / clip_status / clip_codec**, acknowledged_by/at, reset_at, and staff **review_status**
  (open / confirmed / false_alarm / dismissed) with note and reviewer.
  The review fields exist so the measured false-alarm rate can come from real staff outcomes.
- One connection serialised by a lock; written from the frame thread (inserts) and the evidence worker (clip status).

### Google Drive (Colab)
SQLite on a FUSE/Drive mount is a known locking and corruption risk. The database and clips are therefore written to
local disk (`--data-dir`, default `data/`, `/content/pg_data` in Colab). With `--mirror-dir` (Drive), the DB is copied using
**SQLite's online backup API** (consistent while in use; written to `.tmp`, then atomically renamed), and finished clips
are copied too. Mirroring runs on the evidence worker thread, never on the frame path.

### Evidence clips (`pharmaguard/evidence.py`)
- Ring buffer of **annotated** frames (boxes, IDs, wrists, zones are what a reviewer needs), kept by frame timestamp,
  capped at 15 FPS and 5 s (≈52 MB worst case at 640x360). No encoding on the frame path: frames are kept by reference.
- On a recorded alert: pre-event 5 s + post-event 5 s, including the alert's own frame (an ordering bug found while
  designing this: the alert fires before its frame is buffered, so a strict `>` would have dropped it).
- Encoded by one background worker. Codec order, chosen by measurement (2026-11-19, dev laptop):

  | Codec | Writes | 50 frames 640x360 | Browser-playable |
  |---|---|---|---|
  | ffmpeg libx264 (pipe) | yes (ffmpeg present locally and on Colab) | n/a | yes |
  | OpenCV `avc1` | yes on this OpenCV 5.1 build; often absent in Linux pip wheels | 705 ms | yes |
  | OpenCV `VP80` webm | yes | 1017 ms | yes |
  | OpenCV `VP90` webm | yes | 5059 ms (too slow) | yes |
  | OpenCV `mp4v` | yes | 158 ms | **no** (stored as `saved_not_browser_playable`) |

- Variable delivered FPS: the clip frame rate is computed from frame timestamps, so playback runs at real speed.
- Clips record **after** the alarm logic, so they show the overlays the operator saw.

### Privacy and retention
- Clips show faces. `retention_days` (default 7) deletes events, audit rows and clip files at startup.
- `data/`, `*.db` and clip files are git-ignored. Tests use a temp `PHARMAGUARD_DATA_DIR`, and `benchmark.py` and
  `tools/track_video.py` do not record unless asked (`--record-dir`).

### Verified (`tests/test_recording.py`, 13 tests)
Required fields stored; idempotent; acknowledgement keeps the first operator; review statuses validated and audited;
survives restart; retention deletes events and clip files; the mirror copy opens as a valid DB; real clips decode with the
expected frame count via both the ffmpeg and OpenCV paths; the clip window includes pre, post and trigger frames; the ring
buffer is bounded and rate-capped; flush finalises partial clips; a crashing encoder is reported and the worker survives.
In the full pipeline, scripted shelf->bag->leave produces 2 events (REVIEW and ALERT) with camera/track/incident/score/reasons,
2 playable clips (>50 frames), acknowledgement recorded, audit rows, and the DB and clips mirrored.

---

## 10. Dashboard: Live Monitor / Event Review / Zones (2026-11-19)

### Layout
Header and **alarm strip** (banner + Acknowledge / Reset / Test) sit above the tabs, so an alarm is visible from every tab.
- **Live monitor**: camera input, annotated output, stream status, people count, active IDs, delivered/displayed FPS, latency
  breakdown, recent alerts, event timeline, detection/alert thresholds.
- **Event review** (`pharmaguard/ui_review.py`): filterable event table (severity, review status); details (all recorded fields,
  reasons, acknowledgement, staff verdict, non-accusatory framing); evidence clip player; verdict
  (confirmed / false alarm / dismissed / re-open) + note, saved with the operator name and audited.
  The **false-alarm rate is computed only from staff verdicts** (confirmed vs false alarm, "dismissed" excluded), shown as
  "not measured yet" until something is reviewed. That figure is meant for the Dec 1 report.
- **Zones** (`pharmaguard/ui_zones.py`): capture a snapshot of the live (un-annotated) frame, click polygon corners, name the
  zone, Add (working copy) → **Apply** (live, under the pipeline lock, takes effect next frame, audited) → **Save**. Applying an
  empty zone list is refused (monitoring would silently stop).

### Security decisions
- **Zone saves are confined to `configs/*.yaml`.** The dashboard may be reachable through a public share link, so a
  browser-supplied path is reduced to its file name and must end in `.yaml`. Traversal (`../`, `..\\`, absolute paths) cannot
  write outside `configs/`.
- **Clip serving**: Gradio only serves files from the cwd, temp, or `allowed_paths`. Clips live in the data dir (on Colab
  `/content/pg_data`, outside the repo), so `app.py` passes exactly that directory as `allowed_paths`, and nothing else.
- All event/alert text rendered as HTML is escaped (tested with `<script>` in reasons/zone names).

### Verified
- 19 tests (`tests/test_dashboard.py`): zone validation (name, duplicates, <3 points, collinear), normalisation, preview,
  path confinement, YAML save preserving other settings, live `set_zones` effect on the next frame, review table filters, the
  false-alarm-rate maths, escaping, and the review/zone handlers end to end on a pipeline that recorded real events + clips.
- **Real browser (Chrome, 2026-11-19, local server with seeded events):** the review table loaded on tab open; selecting a row
  showed full details; the **evidence clip played** (H.264 via ffmpeg, 10.1 s, 640x480, `readyState 4`, no media error); a
  verdict saved through the UI updated the table and the rate and was persisted with its audit row in SQLite; clicks on the zone
  image registered as numbered corners (`select` works on a non-interactive image in Gradio 6).

---

## 11. CPU Speed Defaults (2026-11-19, after operator report "annotated output is very slow")

Measured on the dev laptop (6-core CPU; its NVIDIA Quadro M2000M is unused because the installed PyTorch is the CPU-only build):
- **Auto inference size**: `detector.imgsz` defaults to None, which means 640 on GPU and **480 on CPU**. Pose detection takes 90 ms at 640 vs **61 ms at 480**
  with the same people found (4.0 vs 4.3 detections/frame on the sample photo). 416 was barely faster and much jumpier (p95 110 ms).
- **On-demand bag detection** (`objects.on_demand`, default True): the bag detector (~100 ms on CPU) runs only while a hand is entering or touching
  a shelf or is inside the 3 s carry window, the only time bag positions can change the score. Trade-off: the informational
  "hand at own bag, no shelf contact" event is no longer produced (it never affected scores or alerts).
- End to end through the real server and HTTP path (`tools/e2e_benchmark.py`, localhost): **236 → 185 ms p50, 389 → 253 ms p95,
  3.8 → 5.2 FPS**.
- The CPU remains the limit (~5 FPS). Smooth real-time needs a GPU: the Colab T4 (intended demo setup). Enabling the laptop's
  Maxwell-generation GPU (compute capability 5.0) needs a CUDA PyTorch build that still supports it; that is unverified and not attempted.

---

## 12. Fast Stream Transport (2026-11-22, after operator report "annotated display is too slow, breaking")

**Problem.** Gradio's streaming webcam keeps one frame in flight and makes 4 HTTP exchanges per frame. Over `gradio.live` each
exchange pays the network round trip, so FPS is set by latency, not by the GPU.

**Decision.** Add `POST /pg/frame` (`pharmaguard/fast_stream.py`) on the same server: body = JPEG, response = annotated JPEG.
The in-page client (Live monitor → "Start camera") keeps up to N requests in flight (default 3, selectable 1-4).
- Same origin, so the Gradio login cookie is sent automatically; requests without a valid session return 401 when auth is on.
- Frames are processed one at a time, strictly in sequence order; a frame that arrives after a newer one was processed is
  skipped (204). Queueing is bounded by the in-flight limit. A random per-camera-start session id restarts numbering.
- An earlier "drop a frame if a newer one was already received" rule raced on fast links and discarded ~50% of uploaded
  frames (measured, 194 skips in 20 s); removed.
- Plain HTTP rather than WebSockets: no new dependency (`websockets` is not installed here) and works through any HTTPS tunnel.
- Browser JPEG encode uses synchronous `canvas.toDataURL`: measured 17 ms per 640x480 frame in Chrome, while the async
  `canvas.toBlob` and `OffscreenCanvas.convertToBlob` took 930-1000 ms.
- The browser reports its own round trip (`X-PG-RTT-Ms`), shown in the dashboard's latency panel as "browser round trip".
- The Gradio webcam stream remains as a fallback in a collapsed accordion. Never run both at once (shared pipeline).
- New config: `stream.fast_jpeg_quality` (default 80).

**Measured (dev laptop, CPU, localhost; `asyncio` client posting the sample photo, network delay simulated in the client):**

| Simulated network RTT | In flight | FPS | round trip p50 |
|---|---|---|---|
| 0 ms | 1 | 8.8 | 98 ms |
| 0 ms | 2 | 9.6 | 201 ms |
| 300 ms | 1 | 2.5 | 407 ms |
| 300 ms | 2 | 4.8 | 406 ms |
| 300 ms | 3 | 7.0 | 420 ms |

Gradio stream path on the same machine, localhost: 185 ms p50, 5.2 FPS. Real Chrome against the local server (synthetic camera
feed, 3 in flight, CPU): 6 FPS displayed, browser round trip p50 508 ms (CPU queueing). **Not yet measured over `gradio.live`
on a Colab GPU**; the share tunnel could not be created from the dev machine.

---

## 13. "Item Not Returned" Trigger (2026-11-22, operator request: taking an item into a pocket/bag should alarm)

**Before:** shelf -> pocket scored 0.55 (nothing) and shelf -> pocket -> walk away 0.85 (REVIEW only). Only the bag path, or
pocketing after several browsing contacts, reached the ALERT at 1.00. Testing in a room, the operator picked up an item,
pocketed it and got nothing.

**Rule added (`behavior.py::_check_kept`):** after a shelf -> pocket/bag transfer, if the same hand does not touch the shelf
again within `keep_confirm_sec` (4.0 s), add `w_item_kept` (0.55, concealment evidence). If the person leaves view inside the
window, it is confirmed immediately. Touching the shelf again with that hand inside the window cancels it (item put back,
RETURNED_TO_SHELF event). New event type `ITEM_NOT_RETURNED`.

**Resulting outcomes (scripted skeleton tests, `tests/test_behavior.py`):**
- shelf -> pocket, stay or leave: **ALERT** 4 s after pocketing (score ~1.05; 0.50 gave 0.997 after decay, hence 0.55).
- shelf -> bag: REVIEW at the transfer (0.80), **ALERT** 4 s later (~1.3).
- pocket or bag, then back to the shelf within 4 s: no ALERT. Browsing, walking past, own-bag reach: unchanged (nothing).
- Examine at the hip for 2 s and put back (S5): still nothing.

**Trade-off (accepted for the demo, must be stated):** a person holding an item at hip height for more than 4 s without
putting it back, or carrying it at the hip towards a till, now raises an ALERT instead of a REVIEW. Real-footage false-alarm
rate remains unmeasured (P3-1). Weights and the window are configurable (`behavior.keep_confirm_sec`, `behavior.w_item_kept`).

---

## 14. Pocket Detection Made Reachable + On-Screen Diagnostics (2026-11-22)

**Operator report (live webcam, Colab):** wrist dot turned orange at the shelf but never magenta at the pocket, so no transfer
was ever recorded. The pocket test needed the wrist keypoint at confidence >= 0.35 inside a 0.30-torso circle around a hip
at >= 0.35. A hand going INTO a pocket is partly hidden, so the wrist confidence drops: the most likely failure.

**Changes (all `BehaviorConfig`):**
- `pocket_wrist_min_conf` 0.10: the pocket/bag test accepts a lower-confidence wrist (`HandObservation.reach_point`).
  Shelf contact still needs `keypoint_min_conf` 0.35 (tested: a 0.2-confidence wrist in the shelf is not a contact).
- `hip_min_conf` 0.25 for placing pocket regions (was the general 0.35).
- `pocket_radius_torso` 0.30 -> 0.42, `pocket_max_above_hip_torso` 0.10 -> 0.15 (also in `configs/demo_shelf.yaml`).
- Annotator: thin magenta circles show the pocket regions; "hips not visible: pocket check off" when hips are missing;
  a hollow marker shows a low-confidence wrist used only for pocket/bag.

**Checked:** scripted scenarios unchanged (browse, examine-at-hip-and-return, relaxed hands still raise nothing); new tests for a
0.2-confidence wrist in the pocket (alarm) and missing hips (pocket off). On the sample photo the new settings flag no hand as
"at pocket" that the old ones did not (one relaxed hand was already flagged: P3-2). **Not yet checked on live pocketing footage.**

---

## 15. Smooth Display: Live Video + Detection Overlay (2026-11-22, operator: "it triggered but the video is still choppy")

**Problem.** Even with frames pipelined (§12), the operator sees a new picture only when an annotated frame comes back,
so the display rate equals the detection rate (CPU ~5-7/s; tunnel round trips add jitter).

**Decision.** The fast stream gains a display mode (`X-PG-Mode: overlay`, default in the UI selector "Smooth"):
- The browser paints its **own camera frames at full rate** (requestAnimationFrame, only when the video has a new frame)
  and draws the latest detections on top: zones + state, bags, person boxes / IDs / score / level, wrist dots (hollow =
  low-confidence wrist), pocket circles, "hips not visible", alarm border, HUD.
- The server returns `PipelineOutput.metadata["overlay"]` as JSON: normalized [0, 1] coordinates, same colours as
  `annotator.py`. Measured on the sample photo: < 20 KB per response (an annotated JPEG is ~40-80 KB).
- The server still annotates every frame (evidence clips and the Gradio fallback need it).
- "Server images" mode keeps the exact §12 behaviour.

**Trade-off (state it if asked):** in smooth mode the boxes and dots lag the live picture by about one round trip (~0.4 s
measured on the local CPU server; tunnel adds more), so they trail a fast-moving person. Detection and alarms are unchanged.

**Measured (local server, CPU, Chrome automation window, synthetic camera):** detections 5-7/s, video painted 8-12 FPS
(the automation window itself only ran requestAnimationFrame at ~7.5-12/s, so this is a lower bound; a foreground window
paints at the display rate). Overlay drawing verified on the sample photo: 3 people, boxes, IDs, wrist and pocket markers,
zone state. **Not yet measured over gradio.live on a GPU.**

---

## 16. Box Lag: Synced Display (2026-11-22, operator: "it's smooth now but the boxes lag behind me")

**Cause.** In §15's smooth mode the live picture is current but each detection describes a frame captured one round trip
earlier, so boxes trail a moving person by speed x lag.

**Decision.** Display mode selector, default **Synced**:
- **Synced (default):** the browser buffers its camera frames (ImageBitmaps, 2.5 s) and shows them delayed by
  `capture->result lag + detection spacing + 30 ms` (smoothed, capped 2 s). Each displayed frame gets the detections
  captured just before and after it, linearly interpolated per person id / hand side, so boxes sit on the person.
  Cost: the whole picture is ~one round trip behind real time, like many CCTV systems.
- **Live:** no video delay; boxes extrapolated from the two newest detections, prediction capped at 0.7 s. Better than no
  correction, but overshoots when a person stops or turns.
- **Server images:** unchanged.
- Server overlay now includes each hand's `side` (needed to match hands between detections).

**Measured (Chrome against local CPU server; fake camera sliding a real photo up to ~250 px/s; box centre compared with
the person's true position in the displayed frame):** Synced median error **3.5 px** (p90 222 px), Live-predicted median
**69 px**; uncorrected would be ~speed x lag (~100 px here). Small sample (9-11 frames per mode: the automation window was
throttled to ~1 frame/s), so treat as indicative. The p90 includes moments when the box matched the wrong reference person.
Not yet measured over gradio.live on a GPU.


---

## 17. Lower Video Delay: Max-Delay Control + Smaller Uploads (2026-11-22, operator: "the video is delayed too much")

**Changes (browser client only):**
- Auto delay is now the median capture->result lag of the last 15 detections (the extra "+ detection spacing + 30 ms"
  margin of §16 is gone; the newest stretch is extrapolated instead of waited for).
- **Max delay** selector (default **300 ms**; Auto / 500 / 300 / 150 / 0): the video is delayed by
  `min(lag, Max delay)`; for the uncovered part of the lag, boxes are extrapolated from the two newest detections
  (capped at 0.7 s). 0 = fully live. The §16 "Live" mode is folded into this (Max delay 0).
- **Upload** selector (default **480 px**, JPEG quality 0.6; 640 / 320 available). Measured on the sample photo
  (OpenCV encoder): 640 px q70 = 70 KB, 480 px q60 = 40 KB (1.8x smaller), 320 px q60 = 19 KB (3.6x smaller).
  Smaller uploads matter when the uplink is slow and several frames are in flight. Pose accuracy at 320 px is unmeasured.

**Measured box error (Node harness running the page's real blend/overlayAt code; person walks ~190 px/s then stops dead;
detections every 150 ms, 600 ms lag):**

| Max delay | mean error | worst (just after the stop) |
|---|---|---|
| 600 ms (= lag, fully synced) | 0.5 px | 17 px |
| 300 ms (default) | 6.2 px | 75 px |
| 150 ms | 11.1 px | 104 px |
| 0 (live) | 17.6 px | 125 px |

Without any correction the error would be speed x lag = 115 px for the whole walk. Chrome check against the local server:
controls apply live (delay capped at 300 then 0), uploads arrive as 480x360, no script errors.

---

## 18. Prediction Jitter Fix; Upload Default Back to 640 px (2026-11-22, operator: "the detection isn't working now")

Reported right after §17. Two changes in §17 could explain it; both are addressed (the operator's screen has not been seen):
- **Prediction amplified jitter (confirmed bug).** Extrapolation took velocity from the two newest detections. With fast GPU
  detections (~30 ms apart), normal keypoint jitter over 30 ms looked like high speed and was projected up to 300 ms ahead.
  Node harness on the page code, person standing still, +-3 px detection jitter: boxes off by mean 21.5 / worst 58 px at
  30 ms spacing (wrist dots jitter more, so worse). Fix: velocity over >= 300 ms of detections (`VELOCITY_BASE_MS`):
  mean 3.0 / worst 7.6 px at 30 ms spacing, 4.5 / 9.5 px at 150 ms. Walk-then-stop at the 300 ms cap: mean 7.9 px,
  worst 75 px (was 6.2 / 75).
- **Upload default 640 px again (JPEG 0.7), 480/320 remain selectable.** Sample photo, pose model at imgsz 640: wrist confidence
  mean 0.63 (640 q70) vs 0.59 (480 q60); a small effect on this photo, but detection was known to work at 640.

---

## 19. Alarm Sound Latency + Duplicate Person Suppression (2026-11-22, operator: "the sound lags and it seems to be tracking 2 people when there's only one")

**Sound lag.** The siren followed the dashboard's alarm panel, refreshed by a 0.5 s Gradio timer whose every tick needs several
round trips over `gradio.live`, on top of the detection lag. Now the overlay JSON of every fast-stream detection carries
`sound: {sounding, seq, kind, volume}`; the page's siren script uses it while fresh (< 1.5 s old) and polls every 100 ms
(was 250 ms), falling back to the panel otherwise (camera stopped, Gradio fallback stream). The siren is not subject to the
display delay of §16-17: it plays when the result arrives. Tested: overlay carries TEST/ALERT sound state; page scripts parse.

**Two people for one.** Close to a webcam the pose model can return two boxes for one body (e.g. upper body inside full body)
that overlap too little for its NMS (IoU 0.45). Boxes cannot distinguish this from two people overlapping; skeletons can.
`detector.suppress_duplicate_poses` (on by default, `detector.dedupe_poses`): two detections sharing >= 3 confidently visible
keypoints with median distance < 0.08 x box height are one person; the one with more visible keypoints (then higher
confidence) is kept, so the full skeleton with hips survives. Tests: same body twice -> 1 (full-body kept); two people one
partly behind the other (boxes overlapping ~60%, skeletons 50 px apart) -> 2; sample photo unchanged (3 people either way).
**Not verified on the operator's footage**; a dashed "(occluded)" second box would instead be a lost-track ghost (shown up
to `tracker.lost_track_seconds` = 2 s) and is not changed here.

---

## 20. Pocket = Bent Elbow; Faster Alarm (2026-11-22, operator: alarm late, dots late, alarms with nothing taken, "pocket when hand at side", some real pocketing missed; runtime CUDA)

**Root cause of the false alarms:** §14 widened the pocket disc, and a relaxed hanging hand fell inside it. Shelf contact, then
simply lowering the arm, read as shelf -> pocket -> not returned -> ALERT. Anthropometrically a relaxed wrist hangs only
~6-8 cm below the hip joint, so position alone cannot separate it from a hand in a pocket.

**Changes (`interaction.py`, `BehaviorConfig`):**
- Pocket region is a box per hip: +-0.40 torso horizontally, 0.15 torso above to 0.25 below the hip keypoint (was a 0.42 disc).
- **Arm shape:** a hand only counts as in a pocket if the shoulder-elbow-wrist angle is < 160 deg (`pocket_straight_arm_deg`).
  A hanging arm is straight. Skipped when shoulder/elbow are not visible.
- Partly hidden wrist accepted from confidence 0.20 (was 0.10: too noisy).
- Faster: `pocket_hold_sec` 0.6 -> 0.4 (quick dips into a pocket were missed), `keep_confirm_sec` 4.0 -> 2.5. Alarm now comes
  ~0.4 + 2.5 s after the hand reaches the pocket, plus network lag.
- Overlay and server annotation draw the pocket boxes.

**Tests:** scripted skeletons now have elbows for two new poses. Shelf contact then a relaxed straight arm with the wrist inside
the band for 8 s: nothing (with the arm check disabled the same script raises an ALERT, i.e. the test reproduces the report).
Shelf -> pocket with a bent elbow: one ALERT within hold + keep window. All earlier scenarios unchanged (155 tests).
**Not verified on real footage**; the 160 deg threshold and band are first estimates for calibration on staged clips.
"Dots change late" is network lag (GPU compute is not the bottleneck); not changed here.

## 21. Phone Cameras over WebSocket (2026-11-24, owner request: "my phone acts as PharmGuard's camera")

**What:** `pharmguard-camera/` — a PWA the phone opens, and a FastAPI server (port 8000, separate from the Gradio dashboard)
that receives JPEG frames on `WS /ws/camera?cam=NAME` and runs each through `PharmaGuardPipeline`. Pairing link
`http://LAN_IP:8000/?cam=NAME&start=1` (QR in the terminal and at `/pair`). Protocol and endpoints: `pharmguard-camera/README.md`.

**Decisions:**
- **One `PharmaGuardPipeline` per camera_id** (tracker, behaviour engine, alerts, alarm, event store/evidence with
  `recording.camera_id = NAME`). **One shared detector** (pose + bag model) behind a lock: memory does not grow per phone,
  and ultralytics `predict()` is not thread-safe. Throughput is therefore split between phones (measured on CPU, dev laptop,
  bus.jpg at 640x853: 1 phone ~2.9 frames/s round trip, 2 phones ~1.5 frames/s each).
- **WebSocket here, unlike §12's plain HTTP**, because the phone is on the LAN (no tunnel) and one socket per phone gives
  ordered frames for free. Exactly one reply per frame (`detections` or `skipped`), so the phone keeps <= 2 in flight; the
  server only ever processes the newest frame, so a slow laptop drops frames rather than building delay.
- **Return format** `{x1,y1,x2,y2,label,conf,alert}` normalized to [0,1] (convention 3), plus `level`/`track_id`/`score`.
  `alert` = person at REVIEW or ALERT level, i.e. "for staff review"; the siren follows ALERT only (convention 5).
- `pipeline._overlay_payload` gained a `conf` key per person (additive).
- Dashboard: "Phone cameras" tab (`pharmaguard/ui_phones.py`) shows `<img src="{server}/view/{id}">` tiles, polls
  `/api/cameras` at 1 Hz, offers Acknowledge/Reset and plays its own siren while a phone camera's alarm sounds. The camera
  server allows the dashboard origin via CORS (`PG_CORS_ORIGINS`). New CLI flag `--phone-cameras URL`.
- Server dependencies live in `pharmguard-camera/server/requirements.txt` (own venv; adds `fastapi`, `uvicorn[standard]`,
  `qrcode`); the main `requirements.txt` is unchanged. `PG_DETECTOR=none` runs without torch (no detection, frames only).

**Risks / not done:** the camera server has **no login**: anyone on the Wi-Fi can view feeds (faces) and acknowledge
alarms. One zone config (`PG_CONFIG`) for all phones. Plain HTTP needs Chrome's insecure-origin flag on each phone.
Not yet tested with a real phone or in a real browser.
