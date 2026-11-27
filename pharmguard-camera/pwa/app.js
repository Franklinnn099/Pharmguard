// PharmGuard phone camera: stream JPEG frames over WebSocket, draw returned detections.
// Protocol: see server/main.py. One binary message = one JPEG; exactly one JSON reply per frame.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const video = $('v'), ov = $('ov'), ctx = ov.getContext('2d');
  const camInput = $('cam'), btn = $('go'), statusEl = $('status'), flag = $('flag'), errBox = $('err');
  const JPEG_QUALITY = 0.7;
  let sendWidth = 640;   // the server raises this when its object scan runs at a higher resolution
  const params = new URLSearchParams(location.search);
  const clean = (s) => (s || '').replace(/[^A-Za-z0-9_-]/g, '').slice(0, 32);
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
  };
  camInput.value = clean(params.get('cam')) || clean(store.get('pg-cam')) || 'phone1';
  // Access key from the QR link (servers with PG_ACCESS_KEY); kept so a reload or home-screen launch still connects.
  if (params.get('key')) store.set('pg-key', params.get('key'));
  const accessKey = store.get('pg-key') || '';
  const keyQ = accessKey ? '&key=' + encodeURIComponent(accessKey) : '';
  // Zone editor for this camera (draws on the last frame the server received from it).
  const zonesLink = $('zones');
  const syncZonesLink = () => {
    zonesLink.href = '/zones/' + encodeURIComponent(clean(camInput.value) || 'phone1') + keyQ.replace('&', '?');
  };
  syncZonesLink();
  camInput.addEventListener('input', syncZonesLink);

  let stream = null, ws = null, running = false, inFlight = 0, maxInFlight = 1;
  // Object boxes: 'key' = people + key objects only (default, uncluttered), 'all' = every box, 'off' = people only.
  const OBJ_MODES = ['key', 'all', 'off'], OBJ_TEXT = { key: 'Objects: Key', all: 'Objects: All', off: 'Objects: Off' };
  const objBtn = $('objs');
  let objMode = OBJ_MODES.includes(store.get('pg-objects')) ? store.get('pg-objects') : null;
  const setObjMode = (m) => { objMode = m; objBtn.textContent = OBJ_TEXT[m]; };
  setObjMode(objMode || 'key');
  objBtn.onclick = () => { const m = OBJ_MODES[(OBJ_MODES.indexOf(objMode) + 1) % 3]; setObjMode(m); store.set('pg-objects', m); };
  const shown = (d) => d.kind !== 'item' || (objMode === 'all') || (objMode === 'key' && d.key);
  let dets = [], fpsTimes = [], lastMs = 0, detector = '', wakeLock = null, retry = 0;
  const grab = document.createElement('canvas');
  const rotateHint = $('rotate');

  const setStatus = (html) => { statusEl.innerHTML = html; };
  const showError = (html) => { errBox.innerHTML = html; errBox.style.display = html ? 'block' : 'none'; };

  // Alarm: siren while the server says this camera's alarm is sounding (once per incident, auto-silences
  // server-side); Acknowledge / Reset post to the same routes the dashboard uses. Same tone as the dashboard.
  const alarmBtn = $('alarm');
  let actx = null, osc = null, flip = null, lastSeq = 0, alarmState = 'IDLE';
  const audioUnlock = () => {   // iOS: audio only starts inside a tap, so call this from the Start tap
    if (!actx) { const AC = window.AudioContext || window.webkitAudioContext; if (AC) actx = new AC(); }
    if (actx && actx.state === 'suspended') actx.resume();
  };
  const sirenStop = () => {
    if (flip) clearInterval(flip); flip = null;
    try { if (osc) osc.stop(); } catch { /* already stopped */ } osc = null;
    try { if (navigator.vibrate) navigator.vibrate(0); } catch { /* unsupported */ }
  };
  const sirenStart = () => {
    if (!actx || osc) return;
    const gain = actx.createGain(); gain.gain.value = 0.25; gain.connect(actx.destination);
    osc = actx.createOscillator(); osc.type = 'square'; osc.frequency.value = 960; osc.connect(gain); osc.start();
    let hi = true;
    flip = setInterval(() => {
      hi = !hi; if (osc) osc.frequency.setValueAtTime(hi ? 960 : 640, actx.currentTime);
      try { if (navigator.vibrate) navigator.vibrate(300); } catch { /* iOS has no vibrate */ }
    }, 400);
  };
  const onAlarm = (state, sounding, seq) => {
    alarmState = state;
    if (sounding && seq !== lastSeq) { sirenStop(); sirenStart(); }
    if (!sounding) sirenStop();
    lastSeq = sounding ? seq : 0;
    alarmBtn.style.display = state === 'ACTIVE' || state === 'ACKNOWLEDGED' ? 'inline-block' : 'none';
    alarmBtn.textContent = state === 'ACTIVE' ? 'Acknowledge' : 'Reset alarm';
  };
  alarmBtn.onclick = async () => {
    sirenStop();
    const cam = encodeURIComponent(clean(camInput.value) || 'phone1');
    try {
      await fetch('/api/cameras/' + cam + '/' + (alarmState === 'ACTIVE' ? 'ack' : 'reset'), { method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-PG-Key': accessKey }, body: JSON.stringify({ operator: 'phone ' + cam }) });
    } catch { /* next reply shows the real state */ }
  };

  // Rear camera by default. 'facing' is session-only and never saved, so every Start (QR scan, reload)
  // opens the rear camera; the Flip button switches for the current session only.
  let facing = 'environment';
  const flipBtn = $('flip');
  async function openCamera(want) {
    const cons = (f) => ({ facingMode: f, width: { ideal: 1280 }, height: { ideal: 960 } });
    try {
      return await navigator.mediaDevices.getUserMedia({ audio: false, video: cons({ exact: want }) });
    } catch (e) {   // no camera facing that way (e.g. a laptop): take whatever is closest
      if (e.name !== 'OverconstrainedError' && e.name !== 'NotFoundError') throw e;
      return navigator.mediaDevices.getUserMedia({ audio: false, video: cons({ ideal: want }) });
    }
  }
  flipBtn.onclick = async () => {
    if (!running) return;
    const want = facing === 'environment' ? 'user' : 'environment';
    try {
      const s = await openCamera(want);
      stream.getTracks().forEach((t) => t.stop());
      stream = s; facing = want; video.srcObject = s; await video.play().catch(() => {});
    } catch (e) { showError('Could not switch camera (' + e.name + ').'); }
  };

  async function requestWakeLock() {
    try { if (navigator.wakeLock) wakeLock = await navigator.wakeLock.request('screen'); } catch { wakeLock = null; }
  }

  async function start() {
    audioUnlock();
    showError('');
    const cam = clean(camInput.value) || 'phone1';
    camInput.value = cam; store.set('pg-cam', cam);
    if (!window.isSecureContext || !navigator.mediaDevices) {
      showError('Camera access is blocked because this page is not a secure origin.<br>' +
        'In Chrome open <code>chrome://flags/#unsafely-treat-insecure-origin-as-secure</code>, ' +
        'enable it, add <code>' + location.origin + '</code>, then tap Relaunch.');
      return;
    }
    try {
      facing = 'environment';
      stream = await openCamera(facing);
      // Captured larger than sent: each frame is shrunk to the server's send_width before upload.
    } catch (e) {
      showError('Could not open the camera (' + e.name + '). Allow camera access for this site.');
      return;
    }
    video.srcObject = stream;
    await video.play().catch(() => {});
    running = true; btn.textContent = 'Stop'; btn.classList.add('stop'); camInput.disabled = true;
    flipBtn.style.display = 'inline-block';
    await requestWakeLock();
    connect(cam);
    requestAnimationFrame(draw);
  }

  function stop() {
    running = false;
    if (ws) { ws.onclose = null; ws.close(); ws = null; }
    if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
    try { if (wakeLock) wakeLock.release(); } catch { /* already released */ }
    wakeLock = null; dets = []; inFlight = 0; flag.style.display = 'none'; rotateHint.style.display = 'none';
    onAlarm('IDLE', false, 0);
    btn.textContent = 'Start'; btn.classList.remove('stop'); camInput.disabled = false;
    flipBtn.style.display = 'none';
    ctx.clearRect(0, 0, ov.width, ov.height); setStatus('Stopped');
  }

  function connect(cam) {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    setStatus('Connecting…');
    ws = new WebSocket(proto + '://' + location.host + '/ws/camera?cam=' + encodeURIComponent(cam) + keyQ);
    ws.binaryType = 'arraybuffer';
    inFlight = 0;
    ws.onopen = () => { retry = 0; };
    ws.onmessage = (ev) => {
      let m;
      try { m = JSON.parse(ev.data); } catch { return; }
      if (m.type === 'hello') { detector = m.detector; maxInFlight = m.max_in_flight || 1; sendWidth = m.send_width || 640;
        if (!store.get('pg-objects') && OBJ_MODES.includes(m.objects_show)) setObjMode(m.objects_show);
        pump(); return; }
      if (m.type === 'detections' || m.type === 'skipped') {
        inFlight = Math.max(0, inFlight - 1);
        if (m.type === 'detections') {
          dets = m.detections || []; lastMs = m.ms || 0;
          const now = performance.now();
          fpsTimes.push(now);
          while (fpsTimes.length && now - fpsTimes[0] > 3000) fpsTimes.shift();
          onAlarm(m.alarm || 'IDLE', !!m.sounding, m.sound_seq || 0);
          flag.textContent = m.alarm === 'ACTIVE' ? 'ALARM · for staff review' : 'For staff review';
          flag.style.display = m.alarm === 'ACTIVE' || dets.some((d) => d.alert) ? 'block' : 'none';
        }
        pump();
      }
    };
    ws.onclose = (ev) => {
      if (!running) return;
      dets = []; inFlight = 0;
      if (ev.code === 4000) {   // another phone took this camera name
        stop(); showError('Another device connected as "' + cam + '". Pick a different camera name.');
        return;
      }
      if (ev.code === 4001) {   // server has PG_ACCESS_KEY and ours is missing or wrong
        stop(); showError('Access key missing or wrong. Scan the QR code from the camera server again.');
        return;
      }
      const delay = Math.min(5000, 500 * 2 ** retry++);
      setStatus('Disconnected, retrying in ' + (delay / 1000).toFixed(1) + ' s…');
      setTimeout(() => { if (running) connect(cam); }, delay);
    };
  }

  // Keep up to maxInFlight frames on the wire; each reply (detections or skipped) frees a slot.
  function pump() {
    if (!running) return;
    if (!ws || ws.readyState !== 1 || !video.videoWidth) { setTimeout(pump, 100); return; }
    while (inFlight < maxInFlight) {
      inFlight++;
      const scale = Math.min(1, sendWidth / video.videoWidth);
      grab.width = Math.round(video.videoWidth * scale);
      grab.height = Math.round(video.videoHeight * scale);
      grab.getContext('2d').drawImage(video, 0, 0, grab.width, grab.height);
      grab.toBlob((b) => {
        if (b && ws && ws.readyState === 1) ws.send(b);
        else { inFlight = Math.max(0, inFlight - 1); setTimeout(pump, 100); }
      }, 'image/jpeg', JPEG_QUALITY);
    }
  }

  function draw() {
    if (!running) return;
    const dpr = window.devicePixelRatio || 1, W = ov.clientWidth, H = ov.clientHeight;
    if (ov.width !== Math.round(W * dpr) || ov.height !== Math.round(H * dpr)) {
      ov.width = Math.round(W * dpr); ov.height = Math.round(H * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const vw = video.videoWidth, vh = video.videoHeight;
    // Portrait frames (phone upright) cut the shelf's width; hint only, streaming continues.
    rotateHint.style.display = vw && vh > vw ? 'block' : 'none';
    if (vw && vh) {
      // Boxes are normalized [0,1]; map them onto the letterboxed (object-fit: contain) video.
      const s = Math.min(W / vw, H / vh), dw = vw * s, dh = vh * s, ox = (W - dw) / 2, oy = (H - dh) / 2;
      ctx.lineWidth = 3; ctx.font = '600 14px system-ui, sans-serif';
      for (const d of dets.filter(shown)) {
        const x = ox + d.x1 * dw, y = oy + d.y1 * dh, w = (d.x2 - d.x1) * dw, h = (d.y2 - d.y1) * dh;
        const color = d.alert ? '#ef4444' : d.label === 'bag' ? '#a78bfa' : d.kind === 'item' ? '#38bdf8' : '#22c55e';
        ctx.strokeStyle = color; ctx.strokeRect(x, y, w, h);
        const text = d.label + (d.alert ? ' · ' + (d.level || 'REVIEW') : '') + ' ' + Math.round((d.conf || 0) * 100) + '%';
        const tw = ctx.measureText(text).width + 10;
        ctx.fillStyle = color; ctx.fillRect(x, Math.max(0, y - 20), tw, 20);
        ctx.fillStyle = d.alert ? '#fff' : '#04111c'; ctx.fillText(text, x + 5, Math.max(15, y - 5));
      }
    }
    const n = fpsTimes.length;
    const fps = n > 1 ? ((n - 1) * 1000 / (fpsTimes[n - 1] - fpsTimes[0])).toFixed(1) : '0.0';
    if (ws && ws.readyState === 1) {
      setStatus('<b>' + camInput.value + '</b> live · ' + fps + ' fps · ' + lastMs.toFixed(0) + ' ms · ' + detector);
    }
    requestAnimationFrame(draw);
  }

  btn.onclick = () => (running ? stop() : start());
  document.addEventListener('visibilitychange', () => {
    if (running && document.visibilityState === 'visible') requestWakeLock();
  });
  if ('serviceWorker' in navigator && window.isSecureContext) navigator.serviceWorker.register('/sw.js').catch(() => {});
  if (params.get('start') === '1') start();
})();
