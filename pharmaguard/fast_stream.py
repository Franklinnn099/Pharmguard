"""
Low-latency frame transport for PharmaGuard ("fast stream").

Why this exists (measured problem, see TECHNICAL_DECISIONS §12):
Gradio's streaming webcam sends one frame, waits for the full round trip, and only then captures the
next. Each frame costs several HTTP exchanges, so over a ``gradio.live`` tunnel the frame rate is
bounded by network latency, not by the GPU.

This module adds one plain HTTP endpoint on the same server (same origin, same login cookie):

    POST /pg/frame   body = JPEG bytes, header X-PG-Seq = client frame number
                     -> 200 image/jpeg (annotated frame)   or   204 (frame skipped, a newer one is queued)

The browser client (``FAST_STREAM_HEAD``) keeps up to N requests in flight, so throughput is
``in_flight / round_trip`` up to the compute limit instead of ``1 / round_trip``.

Ordering: the tracker needs frames in time order. Frames are processed one at a time under a lock, and
a frame whose sequence number is not newer than the last processed frame (it arrived late, overtaken
by a newer one) is skipped, so processing order is always increasing. Queueing is bounded by the
client's in-flight limit (default 3), so a frame never waits behind more than one other frame.
Measured: an earlier "drop if anything newer was received" rule raced on fast links and discarded
about half the uploaded frames, so it was removed. Plain HTTP was chosen over WebSockets because it needs no extra
dependency and passes through any HTTPS tunnel.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

import cv2
import numpy as np
from fastapi import Request
from fastapi.responses import Response
from fastapi.routing import APIRoute
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

FRAME_PATH = "/pg/frame"
MAX_BODY_BYTES = 4 * 1024 * 1024  # a 640x480 JPEG is ~40-80 KB


@dataclass
class FrameResult:
    status: int                  # 200 processed, 204 skipped, 400 bad frame
    body: bytes = b""
    processing_ms: float = 0.0


class FastFrameService:
    """Decode -> pipeline.process_frame -> JPEG encode, strictly in sequence order."""

    def __init__(self, process_rgb: Callable[[np.ndarray], np.ndarray], jpeg_quality: int = 80,
                 on_client_rtt: Optional[Callable[[float], None]] = None):
        self._process_rgb = process_rgb
        self._quality = int(jpeg_quality)
        self._on_client_rtt = on_client_rtt
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._last_processed = -1
        self._session = ""
        self.processed = 0
        self.skipped = 0

    def handle(self, jpeg: bytes, seq: int, client_rtt_ms: Optional[float] = None,
               session: str = "") -> FrameResult:
        """``session`` is a random id per camera start; a new id restarts sequence numbering."""
        if client_rtt_ms is not None and self._on_client_rtt is not None and 0 < client_rtt_ms < 60000:
            self._on_client_rtt(client_rtt_ms)
        with self._state_lock:
            if session != self._session:      # page reload / camera restarted: numbering starts again
                self._session = session
                self._last_processed = -1
        frame_bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR) if jpeg else None
        if frame_bgr is None:
            return FrameResult(status=400)
        with self._lock:
            with self._state_lock:
                stale = seq <= self._last_processed
                if not stale:
                    self._last_processed = seq
            if stale:
                self.skipped += 1
                return FrameResult(status=204)
            annotated_rgb = self._process_rgb(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, self._quality])
            self.processed += 1
        if not ok:
            return FrameResult(status=500)
        return FrameResult(status=200, body=buf.tobytes())


def install_fast_stream_route(fastapi_app: Any, service: FastFrameService, require_login: bool) -> None:
    """Register POST /pg/frame on Gradio's FastAPI app, ahead of Gradio's own routes.

    When the dashboard has a login, the request must carry Gradio's session cookie (the browser sends it
    automatically because the endpoint is same-origin); otherwise 401.
    """
    async def pg_frame(request: Request) -> Response:
        if require_login:
            user = await fastapi_app.get_current_user(request)
            if not user:
                return Response(status_code=401)
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            return Response(status_code=413)
        try:
            seq = int(request.headers.get("x-pg-seq", "0"))
        except ValueError:
            seq = 0
        rtt: Optional[float]
        try:
            rtt = float(request.headers["x-pg-rtt-ms"]) if "x-pg-rtt-ms" in request.headers else None
        except ValueError:
            rtt = None
        session = request.headers.get("x-pg-session", "")[:64]
        result = await run_in_threadpool(service.handle, body, seq, rtt, session)
        if result.status != 200:
            return Response(status_code=result.status)
        return Response(content=result.body, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    route = APIRoute(FRAME_PATH, pg_frame, methods=["POST"], include_in_schema=False)
    fastapi_app.router.routes.insert(0, route)


# Browser client. Wires up the elements rendered by FAST_STREAM_HTML (event delegation, because
# Gradio renders HTML blocks after the head scripts run).
FAST_STREAM_HEAD = """
<script>
(() => {
  const S = {session:'', running:false, stream:null, seq:0, inflight:0, maxInflight:3, lastPainted:-1,
             painted:0, rtts:[], lastRtt:null, errors:0, timer:null, cap:null, capCtx:null};
  const $ = (id) => document.getElementById(id);
  const say = (msg, color) => { const el = $('pg-fast-msg'); if (el) { el.textContent = msg; el.style.color = color || '#94a3b8'; } };
  const median = (a) => { if (!a.length) return 0; const s = [...a].sort((x,y)=>x-y); return s[Math.floor(s.length/2)]; };
  const p95 = (a) => { if (!a.length) return 0; const s = [...a].sort((x,y)=>x-y); return s[Math.min(s.length-1, Math.floor(s.length*0.95))]; };
  const base = () => { let p = window.location.pathname; if (!p.endsWith('/')) p = p.replace(/[^/]*$/, ''); return p; };

  async function sendOne() {
    if (!S.running || S.inflight >= S.maxInflight) return;
    const video = $('pg-fast-video');
    if (!video || video.readyState < 2 || !video.videoWidth) return;
    if (!S.cap) { S.cap = document.createElement('canvas'); S.capCtx = S.cap.getContext('2d'); }
    S.cap.width = video.videoWidth; S.cap.height = video.videoHeight;
    S.capCtx.drawImage(video, 0, 0);
    const seq = S.seq++;
    S.inflight++;
    const t0 = performance.now();
    try {
      // Synchronous toDataURL: measured 17 ms per 640x480 frame in Chrome, while the async
      // canvas.toBlob / OffscreenCanvas.convertToBlob encoders took ~930-1000 ms (scheduled at low priority).
      const b64 = S.cap.toDataURL('image/jpeg', 0.75).split(',')[1];
      const bin = atob(b64); const bytes = new Uint8Array(bin.length);
      for (let k = 0; k < bin.length; k++) bytes[k] = bin.charCodeAt(k);
      const blob = new Blob([bytes], {type: 'image/jpeg'});
      const headers = {'Content-Type': 'image/jpeg', 'X-PG-Seq': String(seq), 'X-PG-Session': S.session};
      if (S.lastRtt !== null) headers['X-PG-RTT-Ms'] = S.lastRtt.toFixed(1);
      const res = await fetch(base() + 'pg/frame', {method:'POST', body: blob, headers, credentials:'same-origin', cache:'no-store'});
      const rtt = performance.now() - t0;
      if (res.status === 200) {
        S.lastRtt = rtt; S.rtts.push(rtt); if (S.rtts.length > 60) S.rtts.shift();
        const img = await res.blob();
        if (seq > S.lastPainted) {
          S.lastPainted = seq;
          const bmp = await createImageBitmap(img);
          const out = $('pg-fast-out');
          if (out) { out.width = bmp.width; out.height = bmp.height; out.getContext('2d').drawImage(bmp, 0, 0); S.painted++; window.__pgFastPainted = (window.__pgFastPainted || 0) + 1; }
          bmp.close && bmp.close();
        }
        S.errors = 0;
      } else if (res.status === 401) {
        say('Not logged in - reload the page and log in again.', '#f87171'); stop();
      } else if (res.status !== 204) {
        S.errors++;
      }
    } catch (e) {
      S.errors++;
      if (S.errors > 20) say('Connection problem: ' + e, '#f87171');
    } finally {
      S.inflight--;
    }
  }

  async function start() {
    if (S.running) return;
    try {
      S.stream = await navigator.mediaDevices.getUserMedia({video: {width: {ideal: 640}, height: {ideal: 480}}, audio: false});
    } catch (e) { say('Camera blocked: ' + e.message + ' (allow camera access, use the https link)', '#f87171'); return; }
    const video = $('pg-fast-video');
    video.srcObject = S.stream; await video.play().catch(() => {});
    const sel = $('pg-fast-inflight'); S.maxInflight = sel ? parseInt(sel.value, 10) || 3 : 3;
    S.session = Math.random().toString(36).slice(2) + Date.now().toString(36);
    S.running = true; S.seq = 0; S.lastPainted = -1; S.rtts = []; S.lastRtt = null; S.errors = 0;
    S.timer = setInterval(sendOne, 15);
    say('Streaming - detection running on the server', '#34d399');
  }

  function stop() {
    S.running = false;
    if (S.timer) clearInterval(S.timer); S.timer = null;
    if (S.stream) S.stream.getTracks().forEach(t => t.stop()); S.stream = null;
    say('Stopped', '#94a3b8');
  }

  document.addEventListener('click', (ev) => {
    const t = ev.target.closest && ev.target.closest('button');
    if (!t) return;
    if (t.id === 'pg-fast-start') start();
    if (t.id === 'pg-fast-stop') stop();
  });
  document.addEventListener('change', (ev) => {
    if (ev.target && ev.target.id === 'pg-fast-inflight') S.maxInflight = parseInt(ev.target.value, 10) || 3;
  });
  setInterval(() => {
    const el = $('pg-fast-stats');
    if (el && S.running) el.textContent = `Displayed ${S.painted} FPS | browser round trip p50 ${median(S.rtts).toFixed(0)} / p95 ${p95(S.rtts).toFixed(0)} ms | ${S.maxInflight} in flight`;
    S.painted = 0;
    window.__pgFast = {running: S.running, seq: S.seq, rttP50: median(S.rtts)};  // automated checks
  }, 1000);
})();
</script>
"""

FAST_STREAM_HTML = """
<div style='border:1px solid #334155;border-radius:8px;padding:10px;background:#0f172a;'>
  <div style='display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px;'>
    <button id='pg-fast-start' style='background:#0284c7;color:white;border:none;border-radius:6px;padding:6px 14px;font-weight:600;cursor:pointer;'>Start camera</button>
    <button id='pg-fast-stop' style='background:#334155;color:white;border:none;border-radius:6px;padding:6px 14px;cursor:pointer;'>Stop</button>
    <label style='font-size:12px;color:#94a3b8;'>Frames in flight
      <select id='pg-fast-inflight' style='background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:4px;'>
        <option value='1'>1</option><option value='2'>2</option><option value='3' selected>3</option><option value='4'>4</option>
      </select></label>
    <span id='pg-fast-msg' style='font-size:12px;color:#94a3b8;'>Click Start camera and allow camera access</span>
  </div>
  <canvas id='pg-fast-out' width='640' height='480' style='width:100%;max-width:960px;background:#000;border-radius:6px;display:block;'></canvas>
  <div id='pg-fast-stats' style='font-size:12px;color:#94a3b8;margin-top:6px;font-family:monospace;'>-</div>
  <video id='pg-fast-video' muted playsinline style='display:none;'></video>
</div>
"""
