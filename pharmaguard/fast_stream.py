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

import json
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
    media_type: str = "image/jpeg"
    processing_ms: float = 0.0


class FastFrameService:
    """Decode -> pipeline.process_frame -> JPEG encode, strictly in sequence order."""

    def __init__(self, process_rgb: Callable[[np.ndarray], Any], jpeg_quality: int = 80,
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
               session: str = "", mode: str = "image") -> FrameResult:
        """``session`` is a random id per camera start; a new id restarts sequence numbering.
        ``mode`` "image" returns the annotated JPEG; "overlay" returns only the drawing data as JSON
        (the browser draws it over its own live video, so nothing large is downloaded).

        ``process_rgb`` may return the annotated RGB array or a PipelineOutput (``annotated_frame_rgb`` +
        ``metadata["overlay"]``)."""
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
            result = self._process_rgb(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            self.processed += 1
        annotated_rgb = getattr(result, "annotated_frame_rgb", result)
        if mode == "overlay":
            overlay = (getattr(result, "metadata", None) or {}).get("overlay")
            if overlay is not None:
                return FrameResult(status=200, body=json.dumps(overlay, separators=(",", ":")).encode(),
                                   media_type="application/json")
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, self._quality])
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
        mode = "overlay" if request.headers.get("x-pg-mode", "") == "overlay" else "image"
        result = await run_in_threadpool(service.handle, body, seq, rtt, session, mode)
        if result.status != 200:
            return Response(status_code=result.status)
        return Response(content=result.body, media_type=result.media_type,
                        headers={"Cache-Control": "no-store"})

    route = APIRoute(FRAME_PATH, pg_frame, methods=["POST"], include_in_schema=False)
    fastapi_app.router.routes.insert(0, route)


# Browser client. Wires up the elements rendered by FAST_STREAM_HTML (event delegation, because
# Gradio renders HTML blocks after the head scripts run). Default display mode 'overlay' paints the
# browser's own camera frames at full rate and draws the latest detections (JSON) on top; 'image'
# shows the server-annotated JPEGs.
FAST_STREAM_HEAD = """
<script>
(() => {
  const S = {session:'', running:false, stream:null, seq:0, inflight:0, maxInflight:3, lastPainted:-1,
             painted:0, updates:0, rtts:[], lastRtt:null, errors:0, timer:null, cap:null, capCtx:null,
             mode:'overlay', overlay:null, overlayAt:0, lastVideoTime:-1, raf:null};
  const $ = (id) => document.getElementById(id);
  const say = (msg, color) => { const el = $('pg-fast-msg'); if (el) { el.textContent = msg; el.style.color = color || '#94a3b8'; } };
  const median = (a) => { if (!a.length) return 0; const s = [...a].sort((x,y)=>x-y); return s[Math.floor(s.length/2)]; };
  const p95 = (a) => { if (!a.length) return 0; const s = [...a].sort((x,y)=>x-y); return s[Math.min(s.length-1, Math.floor(s.length*0.95))]; };
  const base = () => { let p = window.location.pathname; if (!p.endsWith('/')) p = p.replace(/[^/]*$/, ''); return p; };
  const markPainted = () => { S.painted++; window.__pgFastPainted = (window.__pgFastPainted || 0) + 1; };

  // ---- overlay drawing (mirrors pharmaguard/annotator.py colours)
  const ZONE = {CONTACT: ['rgba(180,20,20,0.25)', 'rgb(255,50,50)', 'HAND CONTACT'],
                PRESENCE: ['rgba(180,120,20,0.25)', 'rgb(255,180,40)', 'PERSON AT SHELF'],
                IDLE: ['rgba(20,130,180,0.25)', 'rgb(50,200,255)', 'CLEAR']};
  const LEVEL = {0: 'rgb(40,180,255)', 1: 'rgb(255,190,0)', 2: 'rgb(255,0,0)'};
  const HAND = {free: 'rgb(80,220,80)', shelf: 'rgb(255,140,0)', pocket: 'rgb(255,0,255)'};
  function label(g, text, x, y, color, bg) {
    g.font = '12px sans-serif'; const w = g.measureText(text).width;
    g.fillStyle = bg || 'rgba(30,30,30,0.85)'; g.fillRect(x - 3, y - 13, w + 6, 17);
    g.fillStyle = color; g.fillText(text, x, y);
  }
  function drawOverlay(g, W, H, o, stale) {
    g.save(); if (stale) g.globalAlpha = 0.45;
    for (const z of o.zones || []) {
      const st = ZONE[z.state] || ZONE.IDLE;
      g.beginPath();
      z.polygon.forEach((p, i) => i ? g.lineTo(p[0]*W, p[1]*H) : g.moveTo(p[0]*W, p[1]*H));
      g.closePath();
      g.fillStyle = st[0]; g.fill(); g.strokeStyle = st[1]; g.lineWidth = 2; g.stroke();
      label(g, 'ZONE: ' + z.name + ' [' + st[2] + ']', Math.max(8, z.polygon[0][0]*W), Math.max(40, z.polygon[0][1]*H - 8), st[1]);
    }
    for (const b of o.bags || []) {
      g.strokeStyle = 'rgb(200,60,200)'; g.lineWidth = 1; g.strokeRect(b[0]*W, b[1]*H, (b[2]-b[0])*W, (b[3]-b[1])*H);
      label(g, 'bag', b[0]*W + 3, b[1]*H - 4, 'rgb(200,60,200)');
    }
    for (const p of o.persons || []) {
      const x1 = p.box[0]*W, y1 = p.box[1]*H, x2 = p.box[2]*W, y2 = p.box[3]*H;
      if (p.occluded) {
        g.setLineDash([8, 8]); g.strokeStyle = 'rgb(150,150,150)'; g.lineWidth = 1;
        g.strokeRect(x1, y1, x2-x1, y2-y1); g.setLineDash([]);
        label(g, 'ID #' + p.id + ' (occluded)', x1 + 4, y1 + 16, 'rgb(150,150,150)');
        continue;
      }
      const lvl = p.level || 0;
      const col = (lvl === 0 && p.in_shelf) ? 'rgb(255,140,0)' : LEVEL[lvl];
      g.strokeStyle = col; g.lineWidth = lvl === 2 ? 4 : 2; g.strokeRect(x1, y1, x2-x1, y2-y1);
      for (const c of p.pockets || []) {
        g.beginPath(); g.arc(c.x*W, c.y*H, Math.max(4, c.r*W), 0, 2*Math.PI);
        g.strokeStyle = 'rgb(200,0,200)'; g.lineWidth = 1; g.stroke();
      }
      if (p.hips_visible === false) label(g, 'hips not visible: pocket check off', x1 + 4, y2 - 6, 'rgb(255,200,0)');
      for (const h of p.hands || []) {
        g.beginPath(); g.arc(h.x*W, h.y*H, 6, 0, 2*Math.PI);
        if (h.weak) { g.strokeStyle = HAND[h.state]; g.lineWidth = 2; g.stroke(); }
        else { g.fillStyle = HAND[h.state]; g.fill(); g.strokeStyle = 'white'; g.lineWidth = 1; g.stroke(); }
      }
      const tag = ({1: ' REVIEW', 2: ' ALERT'})[lvl] || '';
      label(g, 'ID #' + p.id + ' | ' + (p.score || 0).toFixed(2) + tag, x1 + 4, Math.max(40, y1 - 4), 'white', lvl ? col : undefined);
    }
    g.restore();
    const flash = Math.floor(performance.now() / 400) % 2 === 0;
    const border = ({ACTIVE: flash ? 'rgb(255,0,0)' : null, ACKNOWLEDGED: 'rgb(160,0,0)',
                     TEST: flash ? 'rgb(200,60,220)' : null})[o.alarm];
    if (border) { g.strokeStyle = border; g.lineWidth = 12; g.strokeRect(6, 6, W - 12, H - 12); }
    g.fillStyle = 'rgba(15,23,42,0.8)'; g.fillRect(0, 0, W, 24);
    g.font = '12px sans-serif'; g.fillStyle = stale ? 'rgb(255,200,0)' : 'rgb(52,211,153)';
    const n = (o.persons || []).filter(p => !p.occluded).length;
    g.fillText(stale ? 'PHARMAGUARD [WAITING FOR DETECTIONS]'
                     : 'PHARMAGUARD [LIVE] | TRACKED: ' + n + ' | ALERTS: ' + (o.alerts || 0), 10, 16);
  }

  // Smooth mode: paint every new camera frame immediately, with the latest detections on top.
  function render() {
    if (!S.running || S.mode !== 'overlay') { S.raf = null; return; }
    const video = $('pg-fast-video'), out = $('pg-fast-out');
    if (video && out && video.readyState >= 2 && video.videoWidth && video.currentTime !== S.lastVideoTime) {
      S.lastVideoTime = video.currentTime;
      if (out.width !== video.videoWidth || out.height !== video.videoHeight) {
        out.width = video.videoWidth; out.height = video.videoHeight;
      }
      const g = out.getContext('2d');
      g.drawImage(video, 0, 0, out.width, out.height);
      if (S.overlay) drawOverlay(g, out.width, out.height, S.overlay, performance.now() - S.overlayAt > 1500);
      markPainted();
    }
    S.raf = requestAnimationFrame(render);
  }

  // ---- frame upload (several requests in flight)
  async function sendOne() {
    if (!S.running || S.inflight >= S.maxInflight) return;
    const video = $('pg-fast-video');
    if (!video || video.readyState < 2 || !video.videoWidth) return;
    if (!S.cap) { S.cap = document.createElement('canvas'); S.capCtx = S.cap.getContext('2d'); }
    S.cap.width = video.videoWidth; S.cap.height = video.videoHeight;
    S.capCtx.drawImage(video, 0, 0);
    const seq = S.seq++;
    const mode = S.mode;
    S.inflight++;
    const t0 = performance.now();
    try {
      // Synchronous toDataURL: measured 17 ms per 640x480 frame in Chrome, while the async
      // canvas.toBlob / OffscreenCanvas.convertToBlob encoders took ~930-1000 ms (scheduled at low priority).
      const b64 = S.cap.toDataURL('image/jpeg', 0.7).split(',')[1];
      const bin = atob(b64); const bytes = new Uint8Array(bin.length);
      for (let k = 0; k < bin.length; k++) bytes[k] = bin.charCodeAt(k);
      const blob = new Blob([bytes], {type: 'image/jpeg'});
      const headers = {'Content-Type': 'image/jpeg', 'X-PG-Seq': String(seq), 'X-PG-Session': S.session, 'X-PG-Mode': mode};
      if (S.lastRtt !== null) headers['X-PG-RTT-Ms'] = S.lastRtt.toFixed(1);
      const res = await fetch(base() + 'pg/frame', {method:'POST', body: blob, headers, credentials:'same-origin', cache:'no-store'});
      const rtt = performance.now() - t0;
      if (res.status === 200) {
        S.lastRtt = rtt; S.rtts.push(rtt); if (S.rtts.length > 60) S.rtts.shift();
        if ((res.headers.get('content-type') || '').includes('json')) {
          const o = await res.json();
          if (seq > S.lastPainted) { S.lastPainted = seq; S.overlay = o; S.overlayAt = performance.now(); S.updates++; }
        } else {
          const img = await res.blob();
          if (seq > S.lastPainted && S.mode === 'image') {
            S.lastPainted = seq;
            const bmp = await createImageBitmap(img);
            const out = $('pg-fast-out');
            if (out) {
              out.width = bmp.width; out.height = bmp.height; out.getContext('2d').drawImage(bmp, 0, 0);
              markPainted(); S.updates++;
            }
            if (bmp.close) bmp.close();
          }
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
    const md = $('pg-fast-mode'); S.mode = md ? md.value : 'overlay';
    S.session = Math.random().toString(36).slice(2) + Date.now().toString(36);
    S.running = true; S.seq = 0; S.lastPainted = -1; S.rtts = []; S.lastRtt = null; S.errors = 0; S.overlay = null;
    S.timer = setInterval(sendOne, 15);
    if (S.mode === 'overlay' && !S.raf) S.raf = requestAnimationFrame(render);
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
    if (!ev.target) return;
    if (ev.target.id === 'pg-fast-inflight') S.maxInflight = parseInt(ev.target.value, 10) || 3;
    if (ev.target.id === 'pg-fast-mode') {
      S.mode = ev.target.value; S.overlay = null;
      if (S.running && S.mode === 'overlay' && !S.raf) S.raf = requestAnimationFrame(render);
    }
  });
  setInterval(() => {
    const el = $('pg-fast-stats');
    const what = S.mode === 'overlay' ? 'Video ' + S.painted + ' FPS | detections ' + S.updates + '/s'
                                      : 'Displayed ' + S.painted + ' FPS';
    if (el && S.running) el.textContent = what + ' | browser round trip p50 ' + median(S.rtts).toFixed(0) +
        ' / p95 ' + p95(S.rtts).toFixed(0) + ' ms | ' + S.maxInflight + ' in flight';
    window.__pgFast = {running: S.running, seq: S.seq, rttP50: median(S.rtts), mode: S.mode,
                       painted: S.painted, updates: S.updates};  // automated checks
    S.painted = 0; S.updates = 0;
  }, 1000);
})();
</script>
"""

FAST_STREAM_HTML = """
<div style='border:1px solid #334155;border-radius:8px;padding:10px;background:#0f172a;'>
  <div style='display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px;'>
    <button id='pg-fast-start' style='background:#0284c7;color:white;border:none;border-radius:6px;padding:6px 14px;font-weight:600;cursor:pointer;'>Start camera</button>
    <button id='pg-fast-stop' style='background:#334155;color:white;border:none;border-radius:6px;padding:6px 14px;cursor:pointer;'>Stop</button>
    <label style='font-size:12px;color:#94a3b8;'>Display
      <select id='pg-fast-mode' style='background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:4px;'>
        <option value='overlay' selected>Smooth (live video + detections)</option>
        <option value='image'>Server images (exact, choppier)</option>
      </select></label>
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
