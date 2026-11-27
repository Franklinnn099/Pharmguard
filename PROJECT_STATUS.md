# PharmaGuard — Project Status Report

**Last updated**: 2026-11-27 (demo runs on the laptop + iPhone, Colab = GPU evidence §22; phone cameras §21; fast stream transport §12; "item not returned" alarm trigger §13; pocket detection + diagnostics §14; smooth display mode §15; synced boxes §16; delay control §17; prediction jitter fix §18; siren via fast stream + duplicate-pose suppression §19; bent-elbow pocket rule + faster alarm §20)
**Milestone**: December 1, 2026 live demonstration
**Demo setup (decided 2026-11-27)**: **iPhone (rear camera, landscape) + laptop camera server on the same Wi-Fi/hotspot**,
facing a **mock shelf**. Colab is GPU evidence only (TECHNICAL_DECISIONS §22). Run guide: **`docs/RUN_GUIDE.md`**.
**Test suite**: 184 passed (`python -m pytest tests/ -v`, dev laptop, CPU, Python 3.13, 2026-11-27)

> Every item below is marked by how it was verified. "Unverified" means it was not executed.
> The previous version of this file contained claims that the audit found false. See `docs/AUDIT_2026-11-19.md`.

---

## 1. Verified (executed in this repository)

| Capability | How verified |
|---|---|
| Real HTTP frame round trip (upload → pipeline → JPEG → download) | `tests/test_stream_pipeline.py::test_live_server_http_round_trip_returns_annotated_jpeg` + `tools/e2e_benchmark.py` against a local server |
| Person detection on real imagery | Sample photo: ≥2 people detected and tracked (`test_pipeline_on_real_people_image_downscaled`) |
| Server-side downscale of HD frames to 640 wide | Unit tests (RGB/RGBA/grey 1280x720 → 640x360) |
| End-to-end FPS/latency telemetry and NO SIGNAL detection | Deterministic fake-clock tests; displayed on dashboard |
| Smooth display: live browser video + detection overlay JSON (`X-PG-Mode: overlay`) | `tests/test_fast_stream.py` (real server, JSON < 20 KB, normalized coords); real Chrome: overlay drawn for 3 people on the sample photo |
| Fast stream (`POST /pg/frame`, several frames in flight, login enforced, in-order processing) | `tests/test_fast_stream.py` (incl. real server + YOLO); real Chrome against local server with a synthetic camera feed: 6 FPS on CPU |
| JPEG return + 640x480 capture request served to browser | `test_ui_transport_configuration` inspects the served Gradio config |
| Tests no longer write to `logs/pharmaguard_alerts.jsonl` | Byte-compared before/after a full test run |
| Multi-person tracking: no ID switches or swaps on walking at 3-25 FPS, crossing paths, side-by-side walkers, partial (low-conf) and 1 s full occlusion, 5-person crowd | Ground-truth scenario tests (`tests/test_tracker.py`); old-vs-new table in TECHNICAL_DECISIONS §6 |
| One-frame false positives never get an ID; lost tracks expire after 2 s of real time | Scenario tests at 5 and 30 FPS |
| Real YOLO + tracker: one stable ID per person on a panned photo | `test_tracker_with_real_detector_on_panning_photo`, `tools/track_video.py` at 5 FPS |
| Pose model returns wrist keypoints aligned with person boxes on real imagery | `test_pose_model_returns_aligned_wrist_keypoints_on_real_people` |
| Behaviour engine, mandate §6 scenarios (scripted skeletons): walk past / browse / pick up & return / own-bag reach / slow transfer / hidden hands give **no** review or alert; shelf→bag→leave gives **one REVIEW + one ALERT**; shelf→pocket→leave gives **REVIEW only**; only the concealer is flagged when 2 people are present | `tests/test_behavior.py` (18 tests) |
| One alert per incident; new incident after decay alerts again | `test_one_incident_one_alert_then_new_incident_after_decay` |
| Full pipeline (tracker → behaviour → alerts → timeline) emits exactly `[WARNING, ALERT]` for shelf→bag→leave | `test_end_to_end_shelf_to_bag_sequence_raises_one_review_and_one_alert` |
| Alarm: sounds only on ALERT, once per incident (even after reset); REVIEW visual only; acknowledge (operator recorded) / reset / 60 s auto-silence; test alarm (5 s, refused during a real alarm, never a security record) | `tests/test_alarm.py` (16 tests) |
| Alarm in a real browser: banner state reaches the page; audio locked until first click (shown on screen); Test alarm siren played and stopped after 5 s; action audit-logged | Chrome on local server, 2026-11-19 (TECHNICAL_DECISIONS §8) |
| Event store: REVIEW/ALERT persisted to SQLite with event id, time, camera id, track id, incident, category, score, reasons, clip, acknowledgement, staff review; idempotent; survives restart; 7-day retention deletes clips; Drive-style mirror is a valid DB | `tests/test_recording.py` |
| Evidence clips: 5 s pre + 5 s post annotated frames, encoded off-thread; real files decode (ffmpeg H.264 and OpenCV avc1 paths verified locally) | `tests/test_recording.py` |
| Full pipeline: shelf->bag->leave stores 2 events + 2 playable clips + ack + audit, mirrored | `test_pipeline_records_events_clips_ack_and_mirrors` |
| Dashboard tabs: Live monitor / Event review / Zones; alarm strip visible on all tabs | `tests/test_dashboard.py`, Chrome |
| Event review: table, details, **clip playback in browser**, staff verdict persisted + audited, false-alarm rate from verdicts | 19 tests + Chrome on seeded real recordings (TECHNICAL_DECISIONS §10) |
| Zone editor: click-to-draw on live snapshot, validation, apply live (next frame), save confined to `configs/*.yaml` | Tests + Chrome (corner clicks register) |
| Pinned requirements == tested environment | `pip install --dry-run -r requirements.txt` → nothing to change |
| Gradio analytics disabled | Startup log no longer shows `huggingface.co` / `api.gradio.app` requests |
| Phone cameras (`pharmguard-camera/`): PWA → WebSocket → real PharmaGuard pipeline per phone; QR pairing; `/monitor`, `/view/{id}` MJPEG, `/api/cameras`; dashboard "Phone cameras" tab | `tests/test_phone_cameras.py` (6 tests: protocol order, routes, CORS, MJPEG, real YOLO per-camera state). Live server on the LAN IP: all routes 200; `test_client.py` 12/12 frames acked in order, 3 persons on bus.jpg; 2 fake phones at once: 15/15 each, separate track IDs. Real iPhone (2026-11-24): HTTPS via mkcert profile, PWA streamed over wss, zone drawn and saved in `/zones/phone1`, people boxed on the phone. "Every object" scan (YOLOE prompt-free, background, ~1 refresh/s, display only) added after; checked on sample images only — **not yet on iPhone footage**. Dashboard tab not yet checked in a browser. **Phone alarm (2026-11-27):** WS replies carry `sounding`/`sound_seq`; PWA plays the siren (audio unlocked by the Start tap; vibrates on Android, iOS has no web vibrate) and shows Acknowledge/Reset. Test: scripted shelf->bag over `/ws/camera` sounds exactly once, phone ack silences it. **Siren not yet heard on the iPhone** |

| Rear camera by default, never mirrored (PWA, dashboard fast stream, Gradio fallback webcam); laptop fallback to its webcam; PWA Flip is session-only | Headless Chrome with fake cameras (`pharmguard-camera/tools/pwa_rear_camera_check.js`): `exact`→`ideal` fallback, left/right colour test on uploaded frames, reload and Stop/Start return to rear; config test for the Gradio webcam. **Not yet checked on the iPhone** |
| Camera-server access key (`PG_ACCESS_KEY`): 401 / WS 4001 without it on every route; QR link carries it; PWA stores it; dashboard tab passes it | `tests/test_phone_cameras.py` (denied/allowed/laptop-mode); Chrome: pairing, reload, monitor via cookie, stranger refused; dashboard tab sends it on `/api`, `/view`, reset (`tools/dashboard_phone_tab_key_check.js`) |
| Laptop defaults: object scan off, 1 frame in flight | Test + measured (§2b) |
| Phone fps badge; "too slow for reliable detection" (< 5 fps or stalled) on phone, `/monitor`, dashboard tab; "Rotate to landscape" hint for portrait frames | Tests + Chrome with a throttled fake phone (~2 fps) and portrait/landscape fake cameras (`tools/pwa_slow_fps_check.js`, `tools/pwa_rotate_hint_check.js`) |
| Colab notebook Option 2 (camera server on T4 + Cloudflare tunnel) and Option 1 fixes (branch clone, CUDA-safe install) | `tests/test_colab_notebook.py` (static: compile, order, safety rules) + local simulation of the cells; **owner ran it on a T4**: tunnel cell worked first time, numbers in §2c |

## 2. Measured performance (dev laptop, CPU only)

**Server compute** (`python benchmark.py`, sample photo, 1280x720 in → 640x360 processed, ~3.9 detections/frame):

| Stage | p50 ms | p95 ms |
|---|---|---|
| detect (YOLOv8n, imgsz 640) | 82.4 | 123.5 |
| track (ByteTrack + Kalman, after rewrite) | 1.3 | 1.5 |
| behaviour (hand engine, after P3) | 1.2 | 1.8 |
| annotate | 3.5 | 4.2 |
| JPEG return encode | 5.1 | 8.8 |
| **total** | **93.7** | **137.2** |

The `--imgsz 416 --max-width 480` variant gives a total p50 of 55.8 ms.

**End-to-end round trip on localhost** (`tools/e2e_benchmark.py`, 1280x720 frames uploaded):

| Configuration | p50 | p95 | Sustained FPS |
|---|---|---|---|
| Previous transport (WebP, no downscale) | 553 ms | 650 ms | 1.8 |
| Current defaults | 190 ms | 215 ms | 5.3 |

**Current defaults on CPU** (after the speed fix, TECHNICAL_DECISIONS §11, localhost, pose model + behaviour + recording path):
round trip p50 **185 ms**, p95 253 ms, **5.2 FPS** (was 236 / 389 ms, 3.8 FPS).

Localhost has no network latency. Over the `gradio.live` tunnel, each frame makes 4 HTTP exchanges, and each one adds a network round trip.

**Fast stream** (TECHNICAL_DECISIONS §12, CPU, localhost, 300 ms network RTT simulated): 2.5 FPS with 1 frame in flight,
**7.0 FPS with 3 in flight**, round trip ~410 ms either way. Localhost, no delay, 1 in flight: 98 ms p50, 8.8 FPS.

### 2b. Laptop camera server, CPU (the demo path), 2026-11-27

`test_client.py`, real pipeline, bus.jpg, 100 frames, localhost (no Wi-Fi):

| Object scan | In flight | Delivered | Round trip p50 / p95 | Compute p50 |
|---|---|---|---|---|
| on | 1 | 5.2 fps | 157 / 337 ms | 143 ms |
| on | 2 | 5.2 fps | 363 / 578 ms | 161 ms |
| **off (default)** | **1 (default)** | **9.4 fps** | **93 / 112 ms** | 79 ms |
| off | 2 | 9.9 fps | 179 / 277 ms | 80 ms |

Real iPhone over the laptop's Wi-Fi with these defaults: **not yet measured**. Read the phone's fps badge.

### 2c. Colab T4 (owner's run, 2026-11-27, torch 2.14.1+cu130): GPU evidence

- `benchmark.py` (Option 1 pipeline, 640x360, FP16): pipeline 15.4 ms, total with encode **17.3 ms p50** (56.6 fps ceiling).
- Camera server through the Cloudflare tunnel, from inside Colab (M-1): scan on 21.5 fps (1 in flight, p50 45 ms) /
  30.9 fps (2 in flight, p50 57 ms); scan off 20.8 / 27.6 fps; compute 18-26 ms p50.
- **Real iPhone in Ghana → tunnel → T4: 1.1 fps and 0.5 fps** (640x853 portrait, 0 skipped; compute 19-29 ms). The internet
  round trip is the bottleneck. A staged concealment fell between frames, and **no alarm fired** → the demo runs locally.
- M-2 (laptop → tunnel) not run.

## 3. Awaiting a live Colab run (requires the project owner; cannot be done from the dev machine)

- [x] `benchmark.py` on a T4: recorded in §2c.
- [x] Option 2 (camera server + tunnel) on a T4 with the real iPhone: recorded in §2c (too slow from Ghana).
- [ ] Use **Start camera** (fast stream) through `gradio.live` on a T4: record Displayed FPS and the browser round trip for 2 and 3 frames in flight.
- [ ] Stream the laptop webcam through `gradio.live`, then record the dashboard's Delivered FPS, Displayed FPS, cycle p50/p95, and server ms.
- [ ] Run `python tools/e2e_benchmark.py --url <share-url> --user … --password …` from the laptop and record the output.
- [ ] Confirm the share URL and the generated password are visible in the launch cell output (launch now uses `!python main.py`).
- [ ] Stop and restart the webcam: confirm NO SIGNAL appears within ~2 s and the stream recovers.
- [ ] Record short staged clips on the demo laptop (two people crossing; one walking behind the other; one person briefly leaving view) and run `python tools/track_video.py <clip> --fps 5 --out out.mp4`. Record "IDs issued" against the number of actors.

## 4. Known defects (open)

| Ref | Defect | Planned |
|---|---|---|
| D-1 | **Today on the iPhone + laptop:** rear camera, landscape, fps 5+, zone drawn, browsing gives no alert, staged concealment gives REVIEW then ALARM. None of the 2026-11-27 phone changes have run on the real iPhone yet | Owner, checklist in `docs/RUN_GUIDE.md` §5 |
| D-2 | The HTTPS cert is valid only for `192.168.1.109`; a hotspot changes the IP | Regenerate with mkcert (RUN_GUIDE §1) |
| C-1 | Colab O2-10 (dashboard beside the tunnelled camera server) never run on Colab | Not needed for the demo |
| P3-1 | **False-alert / miss rates on real staged footage are unmeasured.** Logic is verified only on scripted skeletons | Owner records the scenarios in docs/DEMO_STAGING.md |
| P3-4 | Since §13, an item held at hip height > 4 s without being put back raises an ALERT (was REVIEW). Trade-off for a realistic pocket/bag trigger | Check in staged scenario 5 |
| P3-2 | A relaxed hand at the hip registers as "pocket" on real imagery (harmless unless within 3 s after a shelf contact by the same hand); `pocket_radius_torso` uncalibrated | Calibrate on staged clips |
| P3-3 | CPU only: camera server ~9.4 fps per phone (scan off). GPU measured on Colab (§2c) but not usable live from Ghana | Accepted for the demo |
| R8 | One pipeline is shared by all sessions; two simultaneous streams corrupt tracker state | Before demo: single-operator lock |
| P6-1 | Zone editor exercised in the browser only up to corner clicks; Add/Apply/Save verified via handler tests | Use it in the staged run |
| P5-1 | Clip encoding on Colab relies on the ffmpeg binary (present on Colab, unverified there until the live run); fallback codecs are recorded per clip | Colab run |
| P5-2 | Drive mirroring unverified on a real Drive mount (tested against a local directory) | Colab run with Section 7b |
| P4-1 | Real-ALERT siren path verified by unit tests and the shared code path; only the TEST path was clicked through in a browser | Check during staged run (DEMO_STAGING §5) |

## 5. Next priorities

All six engineering priorities are implemented and tested locally. What remains is measurement on the real setup:

1. **Owner — live Colab run** (Section 3, with Drive mounted): first real GPU, tunnel, clip-encoding and Drive numbers.
2. **Owner — staged scenarios** (`docs/DEMO_STAGING.md`): draw the zone in the Zones tab, run all 10 scenarios, review every
   event in the Event review tab. This yields the **measured false-alert and miss rates** the mandate requires.
3. **Nov 29–30:** fix whatever the staged runs expose (likely `pocket_radius_torso` and thresholds), record the fallback video,
   and rehearse with the alarm checks (DEMO_STAGING §5).

## 6. How to run

**Demo: `docs/RUN_GUIDE.md`** (laptop camera server + iPhone).

```powershell
cd pharmguard-camera\server
.\.venv\Scripts\python main.py              # DEMO: phone cameras, https 8443 / http 8000
```

From the repo root:

```powershell
python -m pytest tests/ -v
python main.py --local --port 7860            # http://127.0.0.1:7860
python benchmark.py                           # server compute
python tools/e2e_benchmark.py --url http://127.0.0.1:7860
python tools/track_video.py clip.mp4 --fps 5 --out annotated.mp4
```

Colab (GPU evidence only): `colab/PharmaGuard_Colab.ipynb`, one option per session. See RUN_GUIDE appendix.
