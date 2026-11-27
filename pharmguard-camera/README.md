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

The terminal prints the laptop's LAN IP, the phone URL (https when certificates exist), the phone setup steps and a QR code.
Optional dashboard alongside it (from the repo root, in a second terminal): `python main.py --local`
→ http://127.0.0.1:7860 → **Phone cameras** tab.

| Env var | Default | Meaning |
|---|---|---|
| `PG_PORT` | `8000` | HTTP port (laptop pages, Android with the Chrome flag) |
| `PG_HTTPS_PORT` | `8443` | HTTPS port, active when `certs/cert.pem` + `certs/key.pem` exist (or `PG_SSL_CERT` / `PG_SSL_KEY`) |
| `PG_DETECTOR` | `pharmaguard` | `pharmaguard` = full pipeline; `none` = no inference (frames shown, detections `[]`). Falls back to `none` with an error log if torch/ultralytics are missing |
| `PG_CONFIG` | — | PharmaGuard YAML (zones, thresholds), e.g. `..\..\configs\demo_shelf.yaml`. Applies to every phone |
| `PG_CAM` | `phone1` | Camera name encoded in the terminal QR |
| `PG_ITEMS` | `0` | `1` = turn on the "every object" scan (YOLOE prompt-free, `yoloe-11s-seg-pf.pt`, downloaded on first run): shelves, chairs, phones, bottles…; unsure detections labelled `object`. **Display only** — never affects alerts. Off by default: on the laptop CPU it halves frames/s (9.4 → 5.2) and doubles latency |
| `PG_MAX_IN_FLIGHT` | `1` | Frames each phone keeps on the wire. 1 = lowest latency on the laptop; 2 may help over a slow network |
| `PG_ITEMS_SHOW` | `key` | Default object view: `key` = only shelf, chair, mobile phone, laptop, bottle, box, bag, cup, medicine; `all` = every box incl. `object`; `off`. The phone's **Objects** button switches it per phone |
| `PG_ITEMS_INTERVAL` | `1.0` | Seconds between object scans (it shares the CPU with person tracking). `0` = as fast as possible |
| `PG_ITEMS_IMGSZ` | `640` | Scan resolution; `960`/`1280` finds smaller objects (slower; the phone then sends larger frames) |
| `PG_ITEMS_MODEL` | `yoloe-11s-seg-pf.pt` | Another model, e.g. `yoloe-11m-seg-pf.pt` (more accurate, slower) or `yolov8n.pt` (80 COCO classes, fast) |
| `PG_NO_RECORDING` | — | `1` = don't write events/clips to `data/` |
| `PG_CORS_ORIGINS` | `http://127.0.0.1:7860,http://localhost:7860` | Dashboard origins allowed to read `/api/cameras` and post ack/reset |
| `PG_ACCESS_KEY` | — | Shared secret. Set = every page, `/view`, `/api` and the WebSocket need it (`?key=`, `X-PG-Key` header or the cookie set from `?key=`); the QR/pair link includes it. **Required on any public tunnel** (Colab). Unset = open (LAN only). Also set it for the dashboard process so its Phone cameras tab can connect |
| `PG_ZONES_DIR` | `server/zones` | Where per-camera zone files are kept (Colab: on Google Drive) |

## On the phone (one-time)

Phone browsers only allow the camera on **HTTPS** (or localhost).

### iPhone (any browser: on iOS they all use Safari's engine, and there is no Chrome flag)

The server serves HTTPS on port **8443** when `server/certs/cert.pem` and `key.pem` exist. Create them once on the
laptop with [mkcert](https://github.com/FiloSottile/mkcert) (Windows: `winget install FiloSottile.mkcert`):

```powershell
cd pharmguard-camera\server
mkdir certs
mkcert -cert-file certs\cert.pem -key-file certs\key.pem <LAN_IP> localhost 127.0.0.1
copy "$(mkcert -CAROOT)\rootCA.pem" certs\ca.crt      # PUBLIC CA cert only - never copy rootCA-key.pem
```

Restart the server, then on the iPhone:

1. In **Safari** open `http://LAN_IP:8000/ca.crt` → **Allow** (downloads a configuration profile).
2. **Settings → Profile Downloaded → Install** (enter the passcode).
3. **Settings → General → About → Certificate Trust Settings** → turn on full trust for **mkcert development CA**.
4. Scan the QR the server prints (it points at `https://LAN_IP:8443/?cam=phone1&start=1`) and allow camera access.

`certs/` is git-ignored: `key.pem` is a private key. The certificate is only valid for the IPs it was made for;
if the laptop's IP changes, re-run the `mkcert -cert-file ...` line with the new IP (the iPhone profile stays valid).
To remove the trust later: Settings → General → VPN & Device Management → mkcert profile → Remove.

### Android (Chrome)

Either install the same `ca.crt` (Settings → Security → Install a certificate → CA certificate) and use the https
link, or skip HTTPS: open `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, enable it, enter
`http://LAN_IP:8000` (scheme + IP + port), tap **Relaunch**, and open `http://LAN_IP:8000/?cam=NAME&start=1`.

### Both

Phone and laptop on the same Wi-Fi. Windows may ask to allow Python through the firewall the first time: allow it
on **Private** networks, or the phone cannot connect. `?start=1` starts streaming at once; open `/pair?cam=NAME`
on the laptop for a QR with another camera name.

## Pairing format

`http://LAN_IP:PORT/?cam=NAME&start=1` — `NAME` is `[A-Za-z0-9_-]{1,32}` and becomes the `camera_id`
(stored with every event). Give each phone its own name; a second device with the same name takes over the
stream and the first one is told to pick another name.

## WebSocket protocol (`/ws/camera?cam=NAME`)

| Direction | Message |
|---|---|
| server → phone | `{"type":"hello","camera_id","detector","model","max_in_flight":1}` once on connect |
| phone → server | binary message = one JPEG frame. Frames are numbered 1, 2, 3… in send order |
| server → phone | `{"type":"detections","frame":n,"detections":[...],"w","h","ms","alarm"}` |
| server → phone | `{"type":"skipped","frame":n}` — a newer frame arrived before frame *n* was processed |
| phone → server | `{"type":"ping"}` (text) → `{"type":"pong","t":...}` |

Exactly one reply per frame, in frame order, so the phone keeps at most `max_in_flight` frames on the wire.
The server always processes the newest frame, so a slow laptop drops frames instead of building up delay.

Each detection: `{x1, y1, x2, y2, label, conf, alert}` plus `level` (`NORMAL`/`REVIEW`/`ALERT`),
`track_id`, `score`, `occluded` for people. Coordinates are **normalized to [0, 1]** of the frame.
`label` is `person #ID`, `bag`, or for scanned objects (`kind: "item"`) a name such as `shelf`, `chair`, `mobile phone`, or `object` when the model is unsure (`name` holds its raw guess). Object boxes refresh about once a second. `alert` is `true` when the behaviour engine has raised that person to
REVIEW or ALERT — **for staff review**, not a finding of theft. The alarm (siren) follows ALERT only.

## HTTP endpoints

| Route | |
|---|---|
| `GET /` | The PWA |
| `GET /ca.crt` | The local CA's public certificate, for installing on an iPhone |
| `GET /pair?cam=NAME`, `GET /pair/qr.svg?cam=NAME` | Pairing page / QR image |
| `GET /monitor` | All cameras, live |
| `GET /zones/{camera_id}` | Zone editor: tap the shelf corners on the live picture; saved per camera in `server/zones/` |
| `GET/PUT/DELETE /api/cameras/{id}/zones`, `GET /api/cameras/{id}/snapshot.jpg` | Zone API and the un-annotated frame it draws on |
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

- **No login unless `PG_ACCESS_KEY` is set.** Without it, anyone on the same Wi-Fi can open `/monitor` and
  `/view/...` (people's faces) and acknowledge alarms: use a private network. With it, anyone holding the QR/link
  has full access, so treat the link like a password. The Gradio dashboard's login does not cover this server.
- Colab (notebook section "Option 2"): frames go through a Cloudflare quick tunnel; the notebook always sets a key.
- Zones: each phone gets its own via `/zones/{camera_id}`; until one is saved it uses `PG_CONFIG`'s zones.
- Object names come from a general-purpose vocabulary; below 45% confidence they are shown as `object`. Specific products (your medicine packs) are not recognised by name.
- All phones share one model on one device: total throughput is split between them.
- iPhones need the HTTPS port and a trusted local CA (above); Android can use plain HTTP with the Chrome flag. On Colab/`gradio.live` (HTTPS) the dashboard
  tab cannot load an `http://` camera server (mixed content).
