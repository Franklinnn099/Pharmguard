# PharmaGuard — Project Status Report

**Last updated**: 2026-11-19 (Claude Code takeover; branch `p1-streaming-pipeline`)
**Milestone**: December 1, 2026 live demonstration
**Demo setup (decided 2026-11-19)**: laptop webcam facing a **mock shelf**; people stand in front of it.
**Test suite**: 31 passed (`python -m pytest tests/ -v`, dev laptop, CPU, Python 3.13, 2026-11-19)

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
| Pinned requirements == tested environment | `pip install --dry-run -r requirements.txt` → nothing to change |
| Gradio analytics disabled | Startup log no longer shows `huggingface.co` / `api.gradio.app` requests |

## 2. Measured performance (dev laptop, CPU only — NOT the demo hardware)

**Server compute** (`python benchmark.py`, sample photo, 1280x720 in → 640x360 processed, ~3.9 detections/frame):

| Stage | p50 ms | p95 ms |
|---|---|---|
| detect (YOLOv8n, imgsz 640) | 82.4 | 123.5 |
| track | 0.3 | 0.4 |
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

## 4. Known defects (open)

| Ref | Defect | Planned |
|---|---|---|
| F1/F4 | Tracker's low-confidence stage is unreachable; IDs switch above ~55 px/frame motion; no Kalman filter | Priority 2 |
| B4 | Zone rule uses feet; a webcam facing a shelf cannot trigger zones | Priority 3 (hand/wrist via pose) |
| B1–B3 | Walking past triggers RAPID RETREAT; one dwell incident re-alerts every 10 s; two shoppers produce 9 alerts in 30 s | Priority 3/4 |
| B5 | With multiple zones, all zones show OCCUPIED when any one is occupied | Priority 3 |
| R8 | One pipeline is shared by all sessions; two simultaneous streams corrupt tracker state | Before demo: single-operator lock |
| — | No audible alarm, acknowledgement, event database, evidence clips, or zone editor | Priorities 4–6 |

## 5. Next priorities

1. **Owner:** live Colab run (Section 3). This produces the first real GPU and tunnel numbers.
2. **Priority 2 — tracking:** fix the threshold overlap, adopt ByteTrack-standard gating with a Kalman filter, a time-based lost buffer and two-frame activation; add crossing and occlusion tests.
3. **Priority 3 — behaviour:** pose-based hand-in-shelf detection, bag/pocket regions, an incident state machine with scoring, and one alarm per incident.

## 6. How to run

```powershell
python -m pytest tests/ -v
python main.py --local --port 7860            # http://127.0.0.1:7860
python benchmark.py                           # server compute
python tools/e2e_benchmark.py --url http://127.0.0.1:7860
```

For Colab, see `colab/PharmaGuard_Colab.ipynb` (Sections 7–9).
