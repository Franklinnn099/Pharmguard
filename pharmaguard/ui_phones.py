"""
"Phone cameras" dashboard tab.

Phones stream to the separate phone-camera server (``pharmguard-camera/server``, default port 8000), which runs
one PharmaGuard pipeline per phone. This tab shows those feeds inside the dashboard:

* each camera is an ``<img src="{server}/view/{camera_id}">`` (annotated MJPEG from that camera's pipeline);
* status comes from ``{server}/api/cameras`` once a second (connected, FPS, persons, flagged, alarm state);
* Acknowledge / Reset buttons POST to ``{server}/api/cameras/{id}/ack|reset``;
* a siren plays while any phone camera's alarm is sounding (same once-per-incident rule as the main alarm).

Everything runs in the browser (Gradio's ``gr.HTML`` does not execute scripts, so the logic is in
``PHONE_CAMERAS_HEAD``, added to the page head). The camera server must allow the dashboard's origin
(``PG_CORS_ORIGINS``, default http://127.0.0.1:7860 and http://localhost:7860).
"""

from html import escape
from typing import Optional

import gradio as gr

PHONE_CAMERAS_HEAD = """
<script>
(() => {
  let actx = null, osc = null, gain = null, flip = null, lastKey = '';
  const unlock = () => {
    if (!actx) { const AC = window.AudioContext || window.webkitAudioContext; if (AC) actx = new AC(); }
    if (actx && actx.state === 'suspended') actx.resume();
  };
  ['pointerdown', 'keydown', 'touchstart'].forEach(ev => document.addEventListener(ev, unlock, {capture: true}));
  const sirenStop = () => {
    if (flip) clearInterval(flip); flip = null;
    try { if (osc) osc.stop(); } catch (e) {} osc = null;
    if (gain) gain.disconnect(); gain = null;
  };
  const sirenStart = () => {
    unlock(); if (!actx || osc) return;
    gain = actx.createGain(); gain.gain.value = 0.25; gain.connect(actx.destination);
    osc = actx.createOscillator(); osc.type = 'square'; osc.connect(gain); osc.start();
    let hi = true; osc.frequency.value = 960;
    flip = setInterval(() => { hi = !hi; if (osc) osc.frequency.setValueAtTime(hi ? 960 : 640, actx.currentTime); }, 400);
  };
  const esc = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const serverOf = (root) => (root.dataset.server || (location.protocol + '//' + location.hostname + ':8000')).replace(/\\/$/, '');
  const tiles = {};

  window.pgPhoneAction = async (cam, action) => {
    const root = document.getElementById('pg-phones'); if (!root) return;
    try {
      await fetch(serverOf(root) + '/api/cameras/' + encodeURIComponent(cam) + '/' + action,
        {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({operator: 'dashboard'})});
    } catch (e) {}
  };

  async function poll() {
    const root = document.getElementById('pg-phones');
    if (!root) return;                         // tab not rendered yet
    const grid = root.querySelector('.pg-phone-grid'), info = root.querySelector('.pg-phone-info');
    const server = serverOf(root);
    let j;
    try { j = await (await fetch(server + '/api/cameras', {cache: 'no-store'})).json(); }
    catch (e) {
      info.innerHTML = 'Camera server not reachable at <code>' + esc(server) + '</code>. Start it with ' +
        '<code>pharmguard-camera\\\\server\\\\.venv\\\\Scripts\\\\python main.py</code>.';
      sirenStop(); return;
    }
    info.innerHTML = 'Server <code>' + esc(server) + '</code> · detector: ' + esc(j.detector) + ' (' + esc(j.model) +
      ') · ' + j.cameras.length + ' camera(s) · pair a phone at <a target="_blank" href="' + esc(j.server) + '/pair">' +
      esc(j.server) + '/pair</a>';
    let sounding = false, key = '';
    const seen = new Set();
    for (const c of j.cameras) {
      seen.add(c.camera_id);
      let t = tiles[c.camera_id];
      if (!t || !t.isConnected) {
        t = document.createElement('div'); t.className = 'pg-phone-tile';
        t.innerHTML = '<img alt="Camera ' + esc(c.camera_id) + '" src="' + server + '/view/' + encodeURIComponent(c.camera_id) + '">' +
          '<div class="pg-phone-meta"></div>';
        grid.appendChild(t); tiles[c.camera_id] = t;
      }
      const a = c.alarm || {}, st = a.state || 'IDLE';
      if (a.sounding) { sounding = true; key += c.camera_id + ':' + a.sound_seq + ';'; }
      t.classList.toggle('hot', c.flagged > 0 || st === 'ACTIVE');
      const btns = (st === 'ACTIVE' || st === 'ACKNOWLEDGED')
        ? ' <button onclick="pgPhoneAction(\\'' + esc(c.camera_id) + '\\',\\'' + (st === 'ACTIVE' ? 'ack' : 'reset') + '\\')">' +
          (st === 'ACTIVE' ? 'Acknowledge' : 'Reset') + '</button>' : '';
      t.querySelector('.pg-phone-meta').innerHTML =
        '<span><b style="color:' + (c.connected ? '#22c55e' : '#64748b') + '">●</b> <b>' + esc(c.camera_id) + '</b> ' +
        (c.connected ? 'live' : 'offline') + '</span><span>' + c.fps + ' fps · ' + c.persons + ' person(s)' +
        (c.flagged ? ' · <span class="pg-badge rv">' + c.flagged + ' for staff review</span>' : '') +
        (st !== 'IDLE' ? ' · <span class="pg-badge ' + (st === 'ACTIVE' ? 'al' : 'rv') + '">' + esc(st) + '</span>' : '') +
        btns + '</span>';
    }
    for (const k of Object.keys(tiles)) if (!seen.has(k)) { tiles[k].remove(); delete tiles[k]; }
    root.querySelector('.pg-phone-empty').style.display = j.cameras.length ? 'none' : 'block';
    if (sounding && key !== lastKey) { sirenStop(); sirenStart(); }
    if (!sounding) sirenStop();
    lastKey = sounding ? key : '';
  }
  setInterval(poll, 1000);
})();
</script>
<style>
#pg-phones .pg-phone-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,380px),1fr));gap:12px;margin-top:10px}
#pg-phones .pg-phone-tile{background:#111827;border:1px solid #334155;border-radius:8px;overflow:hidden}
#pg-phones .pg-phone-tile.hot{border-color:#ef4444;box-shadow:0 0 0 2px #ef4444}
#pg-phones img{display:block;width:100%;aspect-ratio:4/3;object-fit:contain;background:#000}
#pg-phones .pg-phone-meta{display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px;padding:8px 10px;font-size:13px;color:#e2e8f0}
#pg-phones .pg-phone-info{font-size:13px;color:#94a3b8}
#pg-phones .pg-phone-empty{padding:24px 0;color:#94a3b8}
#pg-phones .pg-badge{padding:1px 6px;border-radius:4px;font-weight:600}
#pg-phones .pg-badge.rv{background:#f59e0b;color:#111}
#pg-phones .pg-badge.al{background:#ef4444;color:#fff}
#pg-phones button{margin-left:6px;background:#0ea5e9;color:#04111c;border:0;border-radius:4px;padding:2px 8px;font-weight:600;cursor:pointer}
#pg-phones code{background:#1e293b;padding:1px 5px;border-radius:4px}
#pg-phones a{color:#38bdf8}
</style>
"""


def phone_cameras_html(server_url: Optional[str] = None) -> str:
    """Container the head script fills in. ``server_url`` None = same host as the dashboard, port 8000."""
    attr = f' data-server="{escape(server_url.rstrip("/"))}"' if server_url else ""
    return (f'<div id="pg-phones"{attr}><div class="pg-phone-info">Connecting to the phone camera server…</div>'
            '<div class="pg-phone-empty">No phone cameras yet. Start the camera server and scan its QR code '
            'with a phone.</div><div class="pg-phone-grid"></div></div>')


def build_phones_tab(server_url: Optional[str] = None) -> None:
    gr.HTML(phone_cameras_html(server_url))
