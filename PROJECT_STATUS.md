# PharmaGuard — Project Status Report

**Last updated**: 2026-11-19 (Priorities 1-2 done locally; branch `p1-streaming-pipeline`, not yet merged to `main`)
**Milestone**: December 1, 2026 live demonstration
**Demo setup (decided 2026-11-19)**: laptop webcam facing a **mock shelf**; people stand in front of it.
**Test suite**: 62 passed (`python -m pytest tests/ -v`, dev laptop, CPU, Python 3.13, 2026-11-19)

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
| JPEG return + 640x480 capture request served to browser | `test_ui_transport_configuration` inspects the served Gradio config |
| Tests no longer write to `logs/pharmaguard_alerts.jsonl` | Byte-compared before/after a full test run |
| Multi-person tracking: no ID switches or swaps on walking at 3-25 FPS, crossing paths, side-by-side walkers, partial (low-conf) and 1 s full occlusion, 5-person crowd | Ground-truth scenario tests (`tests/test_tracker.py`); old-vs-new table in TECHNICAL_DECISIONS §6 |
| One-frame false positives never get an ID; lost tracks expire after 2 s of real time | Scenario tests at 5 and 30 FPS |
| Real YOLO + tracker: one stable ID per person on a panned photo | `test_tracker_with_real_detector_on_panning_photo`, `tools/track_video.py` at 5 FPS |
| Pinned requirements == tested environment | `pip install --dry-run -r requirements.txt` → nothing to change |
| Gradio analytics disabled | Startup log no longer shows `huggingface.co` / `api.gradio.app` requests |

## 2. Measured performance (dev laptop, CPU only — NOT the demo hardware)

**Server compute** (`python benchmark.py`, sample photo, 1280x720 in → 640x360 processed, ~3.9 detections/frame):

| Stage | p50 ms | p95 ms |
|---|---|---|
| detect (YOLOv8n, imgsz 640) | 82.4 | 123.5 |
| track (ByteTrack + Kalman, after rewrite) | 1.3 | 1.5 |
| behaviour | 0.1 | 0.1 |
| annotate | 3.5 | 4.2 |
| JPEG return encode | 5.1 | 8.8 |
| **total** | **93.7** | **137.2** |

The `--imgsz 416 --max-width 480` variant gives a total p50 of 55.8 ms.

**End-to-end round trip on localhost** (`tools/e2e_benchmark.py`, 1280x720 frames uploaded):

| Configuration | p50 | p95 | Sustained FPS |
|---|---|---|---|
| Previous transport (WebP, no downscale) | 553 ms | 650 ms | 1.8 |
| Current defaults | 190 ms | 215 ms | 5.3 |

Localhost has no network latency. Over the `gradio.live` tunnel, each frame makes 4 HTTP exchanges, and each one adds a network round trip.

## 3. Awaiting a live Colab run (requires the project owner; cannot be done from the dev machine)

- [ ] Run `colab/PharmaGuard_Colab.ipynb` on a T4 GPU and record the `benchmark.py` table here.
- [ ] Stream the laptop webcam through `gradio.live`, then record the dashboard's Delivered FPS, Displayed FPS, cycle p50/p95, and server ms.
- [ ] Run `python tools/e2e_benchmark.py --url <share-url> --user … --password …` from the laptop and record the output.
- [ ] Confirm the share URL and the generated password are visible in the launch cell output (launch now uses `!python main.py`).
- [ ] Stop and restart the webcam: confirm NO SIGNAL appears within ~2 s and the stream recovers.
- [ ] Record short staged clips on the demo laptop (two people crossing; one walking behind the other; one person briefly leaving view) and run `python tools/track_video.py <clip> --fps 5 --out out.mp4`. Record "IDs issued" against the number of actors.

## 4. Known defects (open)

| Ref | Defect | Planned |
|---|---|---|
| B4 | Zone rule uses feet; a webcam facing a shelf cannot trigger zones | Priority 3 (hand/wrist via pose) |
| B1–B3 | Walking past triggers RAPID RETREAT; one dwell incident re-alerts every 10 s; two shoppers produce 9 alerts in 30 s | Priority 3/4 |
| B5 | With multiple zones, all zones show OCCUPIED when any one is occupied | Priority 3 |
| R8 | One pipeline is shared by all sessions; two simultaneous streams corrupt tracker state | Before demo: single-operator lock |
| — | No audible alarm, acknowledgement, event database, evidence clips, or zone editor | Priorities 4–6 |

## 5. Next priorities

1. **Owner:** live Colab run plus staged tracking clips (Section 3).
2. **Priority 3 — behaviour (Nov 22–25):** add YOLOv8n-pose wrist keypoints; shelf zones judged by hand entry (the webcam faces a mock shelf); bag/pocket regions relative to the person; a per-person incident state machine with a configurable suspicion score; one alert per incident; behaviour and alerts driven by frame timestamps rather than the wall clock.

## 6. How to run

```powershell
python -m pytest tests/ -v
python main.py --local --port 7860            # http://127.0.0.1:7860
python benchmark.py                           # server compute
python tools/e2e_benchmark.py --url http://127.0.0.1:7860
python tools/track_video.py clip.mp4 --fps 5 --out annotated.mp4
```

For Colab, see `colab/PharmaGuard_Colab.ipynb` (Sections 7–9).
