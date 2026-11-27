"""
PharmGuard phone-camera server.

A phone opens the PWA (served from here), streams JPEG frames over a WebSocket, and gets PharmaGuard
detections back. Every frame goes through ``process_frame(camera_id, frame)``, which runs the real PharmaGuard
pipeline (pose detection -> tracking -> hand/shelf/bag behaviour -> alerts) with separate state per phone.

Routes
    GET  /                      PWA (open on the phone; ?cam=NAME&start=1 names the camera and starts it)
    WS   /ws/camera?cam=NAME    binary JPEG frames in, JSON results out (protocol below)
    GET  /pair                  pairing page: QR code for a camera name
    GET  /pair/qr.svg?cam=NAME  the QR code itself
    GET  /monitor               live grid of every camera
    GET  /view/{camera_id}      annotated MJPEG stream for one camera (usable as <img src>)
    GET  /api/cameras           camera status JSON (connection, FPS, persons, flagged, alarm)
    POST /api/cameras/{id}/ack  acknowledge that camera's alarm      (body: {"operator": "..."})
    POST /api/cameras/{id}/reset  reset that camera's alarm

WebSocket protocol (/ws/camera)
    server -> {"type": "hello", "camera_id", "detector", "max_in_flight"}       once, on connect
    client -> binary message = one JPEG frame (frames are numbered 1, 2, 3... in send order)
    server -> {"type": "detections", "frame": n, "detections": [...], "w", "h", "ms", "alarm",
                                                          "sounding", "sound_seq"}   siren while sounding
              {"type": "skipped", "frame": n}     a newer frame arrived before this one was processed
    Exactly one reply per frame, in frame order, so the client can count frames in flight.
    client -> {"type": "ping"}  (text)   server -> {"type": "pong", "t": ...}
    Each detection: {x1, y1, x2, y2, label, conf, alert, level, ...}, coordinates normalized to [0, 1].

Run:  python main.py            (env: PG_PORT=8000, PG_DETECTOR=pharmaguard|none, PG_CONFIG=../../configs/x.yaml)
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import socket
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from pg_bridge import CameraStats, make_bridge

logger = logging.getLogger("pharmguard-camera")

HERE = Path(__file__).resolve().parent
PWA_DIR = HERE.parent / "pwa"
PORT = int(os.environ.get("PG_PORT", "8000"))
# HTTPS (needed by iPhones: iOS only allows the camera on https). Enabled when a certificate exists:
# certs/cert.pem + certs/key.pem (see README, "iPhone"), or PG_SSL_CERT / PG_SSL_KEY.
CERT_DIR = HERE / "certs"
SSL_CERT = os.environ.get("PG_SSL_CERT") or str(CERT_DIR / "cert.pem")
SSL_KEY = os.environ.get("PG_SSL_KEY") or str(CERT_DIR / "key.pem")
TLS = os.path.exists(SSL_CERT) and os.path.exists(SSL_KEY)
HTTPS_PORT = int(os.environ.get("PG_HTTPS_PORT", "8443"))
CA_CERT = CERT_DIR / "ca.crt"     # PUBLIC root certificate only; the CA's private key never leaves the laptop
# Optional shared secret (needed on a public tunnel, e.g. Colab + cloudflared). Unset = no check (laptop/LAN mode).
# Accepted as ?key=, header X-PG-Key or the pg_key cookie, which is set from a valid ?key= so pages, <img> streams
# and fetches on this origin keep working after one QR scan.
ACCESS_KEY = os.environ.get("PG_ACCESS_KEY", "").strip()
# Static PWA shell: no camera data, and the manifest is fetched without cookies, so it stays public.
PUBLIC_PATHS = {"/", "/index.html", "/app.js", "/sw.js", "/manifest.webmanifest", "/icon.svg", "/ca.crt"}
MAX_FRAME_BYTES = 4 * 1024 * 1024
MAX_IN_FLIGHT = 2
VIEW_JPEG_QUALITY = 75
CAM_RE = re.compile(r"[^A-Za-z0-9_-]")


def sanitize_cam(name: Optional[str]) -> str:
    cam = CAM_RE.sub("", (name or "").strip())[:32]
    return cam or "phone"


def lan_ip() -> str:
    """The address phones on the same Wi-Fi should use. A UDP connect sends no packets; it only asks the OS
    which interface would route outward."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def base_url() -> str:
    return f"http://{lan_ip()}:{PORT}"


def phone_base_url() -> str:
    """What phones open: https when a certificate is configured (works on iPhone and Android), else http."""
    return f"https://{lan_ip()}:{HTTPS_PORT}" if TLS else base_url()


def pair_url(cam: str) -> str:
    return f"{phone_base_url()}/?cam={cam}&start=1" + key_query("&")


def key_query(sep: str = "?") -> str:
    """'' without a key, else '?key=...' (or '&key=...') to append to a link."""
    from urllib.parse import quote
    return f"{sep}key={quote(ACCESS_KEY)}" if ACCESS_KEY else ""


def key_ok(*candidates: Optional[str]) -> bool:
    import hmac
    return not ACCESS_KEY or any(c and hmac.compare_digest(c.encode(), ACCESS_KEY.encode()) for c in candidates)


def phone_setup_html() -> str:
    """One-time phone setup steps for the /pair page."""
    if TLS:
        return (f"<li><b>iPhone, one-time:</b> in Safari open <code>{base_url()}/ca.crt</code> and tap Allow. "
                "Then Settings &rarr; <i>Profile Downloaded</i> &rarr; Install. Then Settings &rarr; General &rarr; About "
                "&rarr; Certificate Trust Settings &rarr; turn on full trust for <i>mkcert</i>.</li>"
                "<li><b>Android:</b> either install the same <code>ca.crt</code> (Settings &rarr; Security &rarr; "
                "Install a certificate &rarr; CA certificate), or skip HTTPS: enable Chrome's flag "
                f"<code>chrome://flags/#unsafely-treat-insecure-origin-as-secure</code> for <code>{base_url()}</code> "
                f"and open <code>{base_url()}/?cam=NAME&amp;start=1</code>.</li>")
    return ("<li>One-time (Android Chrome): open <code>chrome://flags/#unsafely-treat-insecure-origin-as-secure</code>, "
            f"enable it, enter <code>{base_url()}</code>, then Relaunch. iPhones need HTTPS: see the README.</li>")


# --------------------------------------------------------------------------- per-camera state
class CameraState:
    def __init__(self, camera_id: str):
        self.stats = CameraStats(camera_id=camera_id)
        self.jpeg: Optional[bytes] = None      # latest annotated frame for /view
        self.version = 0
        self.ws: Optional[WebSocket] = None
        self._fps_times: List[float] = []

    def mark_frame(self) -> None:
        now = time.time()
        self.stats.last_frame_at = now
        self._fps_times.append(now)
        cutoff = now - 3.0
        while self._fps_times and self._fps_times[0] < cutoff:
            self._fps_times.pop(0)

    @property
    def fps(self) -> float:
        t = self._fps_times
        if len(t) < 2:
            return 0.0
        return round((len(t) - 1) / max(t[-1] - t[0], 1e-6), 1)


cameras: Dict[str, CameraState] = {}
bridge = make_bridge(config_path=os.environ.get("PG_CONFIG") or None)


def camera(cam: str) -> CameraState:
    st = cameras.get(cam)
    if st is None:
        st = cameras[cam] = CameraState(cam)
    return st


def process_frame(camera_id: str, frame: np.ndarray) -> Dict[str, Any]:
    """Run one RGB frame from ``camera_id`` through the PharmaGuard pipeline.

    Returns {"detections": [{x1, y1, x2, y2, label, conf, alert, ...}], "jpeg": annotated JPEG, ...}.
    ``alert`` is True when the behaviour engine has raised that person to REVIEW or ALERT (for staff review).
    Runs in a worker thread; per-camera state lives in the bridge, keyed by camera_id.
    """
    res = bridge.process(camera_id, frame)
    jpeg = None
    if res.annotated_rgb is not None:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(res.annotated_rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, VIEW_JPEG_QUALITY])
        jpeg = buf.tobytes() if ok else None
    snap = bridge.alarm_snapshot(camera_id) or {}
    return {"detections": res.detections, "jpeg": jpeg, "w": res.width, "h": res.height,
            "ms": round(res.processing_ms, 1), "alarm": res.alarm,
            "sounding": bool(snap.get("sounding")), "sound_seq": int(snap.get("sound_seq", 0))}


def _placeholder_jpeg(text: str) -> bytes:
    img = np.full((360, 640, 3), 24, dtype=np.uint8)
    cv2.putText(img, text, (40, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    return cv2.imencode(".jpg", img)[1].tobytes()


# --------------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    bridge.close()   # finish evidence clips, close event DBs


app = FastAPI(title="PharmGuard camera server", lifespan=lifespan)
# The PharmaGuard dashboard (Gradio, port 7860) reads /api/cameras and posts ack/reset from the browser.
_origins = [o.strip() for o in os.environ.get(
    "PG_CORS_ORIGINS", "http://127.0.0.1:7860,http://localhost:7860").split(",") if o.strip()]
@app.middleware("http")
async def require_key(request: Request, call_next):
    """With PG_ACCESS_KEY set, every route except the static PWA shell needs the key (WebSocket: see ws_camera)."""
    q = request.query_params.get("key")
    if not key_ok(q, request.headers.get("x-pg-key"), request.cookies.get("pg_key")) \
            and request.url.path not in PUBLIC_PATHS and request.method != "OPTIONS":
        return Response("Access key required: open the link from the server's QR code.", status_code=401)
    response = await call_next(request)
    if ACCESS_KEY and q and key_ok(q):   # remember it for this origin's pages, <img> streams and fetches
        https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
        response.set_cookie("pg_key", ACCESS_KEY, max_age=86400 * 30, httponly=True, samesite="lax", secure=https)
    return response


# Added after require_key so CORS wraps it: a 401 still carries CORS headers and preflights pass.
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST", "PUT", "DELETE"],
                   allow_headers=["Content-Type", "X-PG-Key"])


@app.websocket("/ws/camera")
async def ws_camera(ws: WebSocket) -> None:
    cam = sanitize_cam(ws.query_params.get("cam"))
    await ws.accept()
    if not key_ok(ws.query_params.get("key"), ws.cookies.get("pg_key")):
        await ws.close(code=4001, reason="access key required")
        return
    st = camera(cam)
    if st.ws is not None:                      # same name reconnecting (or a second phone): newest wins
        old = st.ws
        st.ws = None
        try:
            await old.close(code=4000, reason="replaced by a newer connection")
        except Exception:  # noqa: BLE001
            pass
    st.ws = ws
    st.stats.connected = True
    st.stats.connected_at = time.time()
    logger.info("Camera '%s' connected from %s", cam, ws.client.host if ws.client else "?")

    send_lock = asyncio.Lock()
    pending: Dict[str, Any] = {"frame": None}
    wake = asyncio.Event()
    closed = asyncio.Event()

    async def send(msg: Dict[str, Any]) -> None:
        async with send_lock:
            await ws.send_text(json.dumps(msg, separators=(",", ":")))

    async def receiver() -> None:
        seq = 0
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    seq += 1
                    data = msg["bytes"]
                    if len(data) > MAX_FRAME_BYTES:
                        st.stats.skipped += 1
                        await send({"type": "skipped", "frame": seq, "reason": "too large"})
                        continue
                    prev = pending["frame"]
                    if prev is not None:      # never processed: a newer frame replaces it
                        st.stats.skipped += 1
                        await send({"type": "skipped", "frame": prev[0]})
                    pending["frame"] = (seq, data)
                    wake.set()
                elif msg.get("text"):
                    try:
                        if json.loads(msg["text"]).get("type") == "ping":
                            await send({"type": "pong", "t": time.time()})
                    except (ValueError, AttributeError):
                        pass
        finally:
            closed.set()
            wake.set()

    async def processor() -> None:
        while not closed.is_set():
            await wake.wait()
            wake.clear()
            item, pending["frame"] = pending["frame"], None
            if item is None:
                continue
            seq, data = item
            frame_bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                st.stats.skipped += 1
                await send({"type": "skipped", "frame": seq, "reason": "not a JPEG"})
                continue
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            try:
                out = await run_in_threadpool(process_frame, cam, rgb)
            except Exception:  # noqa: BLE001 - one bad frame must not kill the stream
                logger.exception("process_frame failed for camera '%s'", cam)
                st.stats.skipped += 1
                await send({"type": "skipped", "frame": seq, "reason": "processing error"})
                continue
            st.jpeg = out["jpeg"] or data       # "none" mode: show the phone's own frame
            st.version += 1
            st.mark_frame()
            s = st.stats
            s.frames += 1
            s.processing_ms, s.width, s.height, s.alarm = out["ms"], out["w"], out["h"], out["alarm"]
            persons = [d for d in out["detections"] if d["label"].startswith("person")]
            s.persons, s.flagged = len(persons), sum(1 for d in persons if d["alert"])
            if not closed.is_set():
                await send({"type": "detections", "frame": seq, "detections": out["detections"],
                            "w": out["w"], "h": out["h"], "ms": out["ms"], "alarm": out["alarm"],
                            "sounding": out.get("sounding", False), "sound_seq": out.get("sound_seq", 0)})

    tasks: List[asyncio.Task] = []
    try:
        await send({"type": "hello", "camera_id": cam, "detector": bridge.mode, "model": bridge.model_name,
                    "objects": bridge.items_model, "max_in_flight": MAX_IN_FLIGHT, "send_width": bridge.send_width,
                    "objects_show": bridge.items_show})
        tasks = [asyncio.create_task(receiver()), asyncio.create_task(processor())]
        await asyncio.gather(*tasks)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        for t in tasks:
            t.cancel()
        if st.ws is ws:
            st.ws = None
            st.stats.connected = False
        logger.info("Camera '%s' disconnected (%d frames processed)", cam, st.stats.frames)


@app.get("/view/{camera_id}")
async def view(camera_id: str) -> StreamingResponse:
    cam = sanitize_cam(camera_id)

    async def frames():
        last = -1
        waiting = _placeholder_jpeg(f"Waiting for camera '{cam}'...")
        sent_placeholder = 0.0
        while True:
            st = cameras.get(cam)
            if st is not None and st.jpeg is not None and st.version != last:
                last, jpg = st.version, st.jpeg
            elif (st is None or st.jpeg is None) and time.time() - sent_placeholder > 1.0:
                jpg, sent_placeholder = waiting, time.time()
            else:
                await asyncio.sleep(0.03)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode()
                   + b"\r\n\r\n" + jpg + b"\r\n")

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


@app.get("/api/cameras")
async def api_cameras() -> JSONResponse:
    now = time.time()
    out = []
    for cam, st in sorted(cameras.items()):
        s = st.stats
        out.append({
            "camera_id": cam, "connected": s.connected, "fps": st.fps if s.connected else 0.0,
            "frames": s.frames, "skipped": s.skipped, "processing_ms": s.processing_ms,
            "resolution": f"{s.width}x{s.height}" if s.width else "",
            "last_frame_age_sec": round(now - s.last_frame_at, 1) if s.last_frame_at else None,
            "persons": s.persons, "flagged": s.flagged, "alarm": bridge.alarm_snapshot(cam) or {"state": s.alarm},
            "view_url": f"/view/{cam}",
        })
    return JSONResponse({"detector": bridge.mode, "model": bridge.model_name, "server": base_url(),
                         "cameras": out}, headers={"Cache-Control": "no-store"})


async def _operator(request: Request) -> str:
    try:
        body = await request.json()
        return str(body.get("operator") or "operator")[:64]
    except Exception:  # noqa: BLE001
        return "operator"


@app.post("/api/cameras/{camera_id}/ack")
async def api_ack(camera_id: str, request: Request) -> JSONResponse:
    ok = bridge.acknowledge(sanitize_cam(camera_id), await _operator(request))
    return JSONResponse({"ok": ok})


@app.post("/api/cameras/{camera_id}/reset")
async def api_reset(camera_id: str, request: Request) -> JSONResponse:
    ok = bridge.reset_alarm(sanitize_cam(camera_id), await _operator(request))
    return JSONResponse({"ok": ok})


@app.get("/api/cameras/{camera_id}/zones")
async def api_get_zones(camera_id: str) -> JSONResponse:
    return JSONResponse(await run_in_threadpool(bridge.get_zones, sanitize_cam(camera_id)),
                        headers={"Cache-Control": "no-store"})


@app.put("/api/cameras/{camera_id}/zones")
async def api_put_zones(camera_id: str, request: Request) -> JSONResponse:
    """Body {"zones": [{"name", "polygon_normalized": [[x, y], ...]}]}; applied on the camera's next frame."""
    from pg_bridge import validate_zones
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({"ok": False, "error": "body must be JSON"}, status_code=400)
    zones, err = validate_zones(body.get("zones") if isinstance(body, dict) else None)
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    cam = sanitize_cam(camera_id)
    await run_in_threadpool(bridge.set_zones, cam, zones)
    return JSONResponse({"ok": True, **(await run_in_threadpool(bridge.get_zones, cam))})


@app.delete("/api/cameras/{camera_id}/zones")
async def api_reset_zones(camera_id: str) -> JSONResponse:
    cam = sanitize_cam(camera_id)
    await run_in_threadpool(bridge.reset_zones, cam)
    return JSONResponse({"ok": True, **(await run_in_threadpool(bridge.get_zones, cam))})


@app.get("/api/cameras/{camera_id}/snapshot.jpg")
async def api_snapshot(camera_id: str) -> Response:
    """Latest frame WITHOUT annotations, for drawing zones on."""
    cam = sanitize_cam(camera_id)
    rgb = bridge.raw_frame(cam)
    if rgb is not None:
        ok, buf = await run_in_threadpool(cv2.imencode, ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        data = buf.tobytes() if ok else None
    else:
        st = cameras.get(cam)
        data = st.jpeg if (st is not None and bridge.mode != "pharmaguard") else None   # "none" mode: raw frame
    if data is None:
        return Response("No frame from this camera yet - start streaming first.", status_code=404)
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/zones/{camera_id}", response_class=HTMLResponse)
async def zones_page(camera_id: str) -> HTMLResponse:
    cam = sanitize_cam(camera_id)
    return HTMLResponse((HERE / "templates" / "zones.html").read_text(encoding="utf-8").replace("{{CAM}}", cam))


@app.get("/pair/qr.svg")
async def pair_qr(cam: str = "phone1") -> Response:
    import qrcode
    import qrcode.image.svg
    img = qrcode.make(pair_url(sanitize_cam(cam)), image_factory=qrcode.image.svg.SvgPathFillImage, box_size=12)
    buf = io.BytesIO()
    img.save(buf)
    return Response(buf.getvalue(), media_type="image/svg+xml", headers={"Cache-Control": "no-store"})


@app.get("/pair", response_class=HTMLResponse)
async def pair(cam: str = "phone1") -> HTMLResponse:
    cam = sanitize_cam(cam)
    return HTMLResponse((HERE / "templates" / "pair.html").read_text(encoding="utf-8")
                        .replace("{{CAM}}", cam).replace("{{URL}}", pair_url(cam))
                        .replace("{{SETUP}}", phone_setup_html()))


@app.get("/ca.crt")
async def ca_cert() -> Response:
    """The local CA's PUBLIC certificate, for installing on an iPhone (Safari turns it into a profile)."""
    if not CA_CERT.exists():
        return Response("No CA certificate configured (see README, iPhone).", status_code=404)
    return Response(CA_CERT.read_bytes(), media_type="application/x-x509-ca-cert",
                    headers={"Content-Disposition": 'attachment; filename="pharmguard-ca.crt"'})


@app.get("/monitor", response_class=HTMLResponse)
async def monitor() -> HTMLResponse:
    return HTMLResponse((HERE / "templates" / "monitor.html").read_text(encoding="utf-8"))


# PWA last, so it never shadows the routes above.
app.mount("/", StaticFiles(directory=str(PWA_DIR), html=True), name="pwa")


def print_banner(cam: str = "phone1") -> None:
    url = pair_url(cam)
    print("\n" + "=" * 64)
    print(" PharmGuard camera server")
    print(f"   Detector:   {bridge.mode}  ({bridge.model_name})")
    print(f"   Objects:    {bridge.items_model}  (display only; PG_ITEMS=0 off, PG_ITEMS_IMGSZ=960 smaller objects)")
    print(f"   LAN IP:     {lan_ip()}")
    print(f"   Phone URL:  {url}")
    print(f"   Pair page:  {base_url()}/pair{key_query()}     Monitor: {base_url()}/monitor{key_query()}")
    if ACCESS_KEY:
        print("   Access key: required on every page (PG_ACCESS_KEY); the links above and the QR include it")
    if TLS:
        print(f"   HTTPS:      {phone_base_url()}  (phones; iPhone needs this)")
        print(f"   iPhone, one-time: Safari -> {base_url()}/ca.crt -> Allow, then Settings -> Profile Downloaded")
        print("               -> Install, then General -> About -> Certificate Trust Settings -> enable mkcert")
    else:
        print(f"   Android Chrome flag (chrome://flags/#unsafely-treat-insecure-origin-as-secure): {base_url()}")
        print("   iPhone: needs HTTPS - create certs/cert.pem + key.pem (README, 'iPhone') and restart")
    print("=" * 64)
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(out=sys.stdout, invert=True)
    except Exception as exc:  # noqa: BLE001 - the URL above still works
        print(f"(QR code unavailable: {exc})")
    print(f"Scan the QR with the phone (camera name '{cam}'). Other names: {base_url()}/pair?cam=NAME\n",
          flush=True)


if __name__ == "__main__":
    import uvicorn
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print_banner(os.environ.get("PG_CAM", "phone1"))
    host = os.environ.get("PG_HOST", "0.0.0.0")
    servers = [uvicorn.Server(uvicorn.Config(app, host=host, port=PORT, log_level="info"))]
    if TLS:   # same app on a second port with TLS; lifespan (pipeline shutdown) runs once, on the HTTP server
        servers.append(uvicorn.Server(uvicorn.Config(app, host=host, port=HTTPS_PORT, log_level="info",
                                                     ssl_certfile=SSL_CERT, ssl_keyfile=SSL_KEY, lifespan="off")))

    async def serve_all() -> None:
        await asyncio.gather(*(srv.serve() for srv in servers))

    asyncio.run(serve_all())
