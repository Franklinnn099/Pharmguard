# pharmguard-camera — use a phone as a PharmaGuard camera

A phone opens a small web app (PWA) served by the laptop, streams its camera to the laptop over a WebSocket, and
draws the detection boxes it gets back. On the laptop, every frame goes through the **real PharmaGuard pipeline**
(`../pharmaguard`: pose detection → ByteTrack → hand/shelf/bag behaviour engine → alerts, alarm, event DB and
evidence clips). Each phone has its own pipeline state, so several phones can stream at once.

```
phone (PWA, Chrome)  ── JPEG frames over ws://LAPTOP:8000/ws/camera?cam=NAME ──►  server/main.py
                     ◄── JSON detections per frame ───────────────────────────    └─ process_frame(camera_id, frame)
                                                                                     └─ PharmaGuardPipeline (one per camera)
laptop browser:  /monitor  ·  /view/{camera_id} (MJPEG)  ·  PharmaGuard dashboard → "Phone cameras" tab
```

## Layout

| Path | What it is |
|---|---|
| `pwa/` | The phone app: `index.html`, `app.js` (camera → WebSocket → overlay), manifest, service worker, icon |
| `server/main.py` | FastAPI server: serves the PWA, `/ws/camera`, `/pair` + terminal QR, `/monitor`, `/view/{id}`, `/api/cameras` |
| `server/pg_bridge.py` | Per-camera `PharmaGuardPipeline`s with one shared, lock-protected YOLO model |
| `server/templates/` | `/pair` and `/monitor` pages |
| `server/test_client.py` | Fake phone: sends JPEG frames and checks one reply per frame comes back |

## Setup (Windows, PowerShell)

```powershell
cd pharmguard-camera\server
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

If torch/ultralytics will not install, install only the base packages and run without detection:

```powershell
.\.venv\Scripts\python -m pip install -r requirements-base.txt
$env:PG_DETECTOR = "none"
```

(macOS/Linux: `python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt`.)

## Run

```powershell
cd pharmguard-camera\server
.\.venv\Scripts\python main.py
```

The terminal prints the laptop's LAN IP, the phone URL, the exact Chrome-flag value and a QR code.
Optional dashboard alongside it (from the repo root, in a second terminal): `python main.py --local`
→ http://127.0.0.1:7860 → **Phone cameras** tab.

| Env var | Default | Meaning |
|---|---|---|
| `PG_PORT` | `8000` | Server port |
| `PG_DETECTOR` | `pharmaguard` | `pharmaguard` = full pipeline; `none` = no inference (frames shown, detections `[]`). Falls back to `none` with an error log if torch/ultralytics are missing |
| `PG_CONFIG` | — | PharmaGuard YAML (zones, thresholds), e.g. `..\..\configs\demo_shelf.yaml`. Applies to every phone |
| `PG_CAM` | `phone1` | Camera name encoded in the terminal QR |
| `PG_NO_RECORDING` | — | `1` = don't write events/clips to `data/` |
| `PG_CORS_ORIGINS` | `http://127.0.0.1:7860,http://localhost:7860` | Dashboard origins allowed to read `/api/cameras` and post ack/reset |

## On the phone (one-time)

Browsers only allow the camera on HTTPS or localhost. For a plain-HTTP LAN address, tell Chrome to trust it:

1. Open `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.
2. Enable it and enter the address the server prints, e.g. `http://192.168.1.20:8000` (scheme + IP + port, no path).
3. Tap **Relaunch**.
4. Phone and laptop on the same Wi-Fi. Scan the QR (or open `/pair?cam=NAME` on the laptop for another name),
   allow camera access. `?start=1` starts streaming at once.

If the laptop's IP changes (different Wi-Fi), update the flag. Windows may ask to allow Python through the
firewall the first time: allow it on **Private** networks, or the phone cannot connect.

## Pairing format

`http://LAN_IP:PORT/?cam=NAME&start=1` — `NAME` is `[A-Za-z0-9_-]{1,32}` and becomes the `camera_id`
(stored with every event). Give each phone its own name; a second device with the same name takes over the
stream and the first one is told to pick another name.

## WebSocket protocol (`/ws/camera?cam=NAME`)

| Direction | Message |
|---|---|
| server → phone | `{"type":"hello","camera_id","detector","model","max_in_flight":2}` once on connect |
| phone → server | binary message = one JPEG frame. Frames are numbered 1, 2, 3… in send order |
| server → phone | `{"type":"detections","frame":n,"detections":[...],"w","h","ms","alarm"}` |
| server → phone | `{"type":"skipped","frame":n}` — a newer frame arrived before frame *n* was processed |
| phone → server | `{"type":"ping"}` (text) → `{"type":"pong","t":...}` |

Exactly one reply per frame, in frame order, so the phone keeps at most `max_in_flight` frames on the wire.
The server always processes the newest frame, so a slow laptop drops frames instead of building up delay.

Each detection: `{x1, y1, x2, y2, label, conf, alert}` plus `level` (`NORMAL`/`REVIEW`/`ALERT`),
`track_id`, `score`, `occluded` for people. Coordinates are **normalized to [0, 1]** of the frame.
`label` is `person #ID` or `bag`. `alert` is `true` when the behaviour engine has raised that person to
REVIEW or ALERT — **for staff review**, not a finding of theft. The alarm (siren) follows ALERT only.

## HTTP endpoints

| Route | |
|---|---|
| `GET /` | The PWA |
| `GET /pair?cam=NAME`, `GET /pair/qr.svg?cam=NAME` | Pairing page / QR image |
| `GET /monitor` | All cameras, live |
| `GET /view/{camera_id}` | Annotated MJPEG (zones, tracks, HUD) — usable as `<img src>` |
| `GET /api/cameras` | Status per camera: connected, fps, frames, skipped, persons, flagged, alarm |
| `POST /api/cameras/{id}/ack`, `/reset` | Acknowledge / reset that camera's alarm (`{"operator": "..."}`) |

## Test without a phone

```powershell
# terminal 1
.\.venv\Scripts\python main.py
# terminal 2
.\.venv\Scripts\python test_client.py --frames 10
```

## Limits (read before the demo)

- **No login on this server.** Anyone on the same Wi-Fi can open `/monitor` and `/view/...` (people's faces)
  and acknowledge alarms. Use a private network. The Gradio dashboard's login does not cover port 8000.
- One zone config (`PG_CONFIG`) for all phones; zones are normalized, so a phone must frame the shelf the
  way the config expects. Per-phone zones are not implemented.
- All phones share one model on one device: total throughput is split between them.
- Plain HTTP on the LAN; the Chrome flag is the workaround. On Colab/`gradio.live` (HTTPS) the dashboard
  tab cannot load an `http://` camera server (mixed content).
