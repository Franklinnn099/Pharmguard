// PharmGuard phone camera: stream JPEG frames over WebSocket, draw returned detections.
// Protocol: see server/main.py. One binary message = one JPEG; exactly one JSON reply per frame.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const video = $('v'), ov = $('ov'), ctx = ov.getContext('2d');
  const camInput = $('cam'), btn = $('go'), statusEl = $('status'), flag = $('flag'), errBox = $('err');
  const SEND_WIDTH = 640, JPEG_QUALITY = 0.7;
  const params = new URLSearchParams(location.search);
  const clean = (s) => (s || '').replace(/[^A-Za-z0-9_-]/g, '').slice(0, 32);
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
  };
  camInput.value = clean(params.get('cam')) || clean(store.get('pg-cam')) || 'phone1';

  let stream = null, ws = null, running = false, inFlight = 0, maxInFlight = 2;
  let dets = [], fpsTimes = [], lastMs = 0, detector = '', wakeLock = null, retry = 0;
  const grab = document.createElement('canvas');

  const setStatus = (html) => { statusEl.innerHTML = html; };
  const showError = (html) => { errBox.innerHTML = html; errBox.style.display = html ? 'block' : 'none'; };

  async function requestWakeLock() {
    try { if (navigator.wakeLock) wakeLock = await navigator.wakeLock.request('screen'); } catch { wakeLock = null; }
  }

  async function start() {
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
      stream = await navigator.mediaDevices.getUserMedia({ audio: false,
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 640 }, height: { ideal: 480 } } });
    } catch (e) {
      showError('Could not open the camera (' + e.name + '). Allow camera access for this site.');
      return;
    }
    video.srcObject = stream;
    await video.play().catch(() => {});
    running = true; btn.textContent = 'Stop'; btn.classList.add('stop'); camInput.disabled = true;
    await requestWakeLock();
    connect(cam);
    requestAnimationFrame(draw);
  }

  function stop() {
    running = false;
    if (ws) { ws.onclose = null; ws.close(); ws = null; }
    if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
    try { if (wakeLock) wakeLock.release(); } catch { /* already released */ }
    wakeLock = null; dets = []; inFlight = 0; flag.style.display = 'none';
    btn.textContent = 'Start'; btn.classList.remove('stop'); camInput.disabled = false;
    ctx.clearRect(0, 0, ov.width, ov.height); setStatus('Stopped');
  }

  function connect(cam) {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    setStatus('Connecting…');
    ws = new WebSocket(proto + '://' + location.host + '/ws/camera?cam=' + encodeURIComponent(cam));
    ws.binaryType = 'arraybuffer';
    inFlight = 0;
    ws.onopen = () => { retry = 0; };
    ws.onmessage = (ev) => {
      let m;
      try { m = JSON.parse(ev.data); } catch { return; }
      if (m.type === 'hello') { detector = m.detector; maxInFlight = m.max_in_flight || 2; pump(); return; }
      if (m.type === 'detections' || m.type === 'skipped') {
        inFlight = Math.max(0, inFlight - 1);
        if (m.type === 'detections') {
          dets = m.detections || []; lastMs = m.ms || 0;
          const now = performance.now();
          fpsTimes.push(now);
          while (fpsTimes.length && now - fpsTimes[0] > 3000) fpsTimes.shift();
          flag.style.display = dets.some((d) => d.alert) ? 'block' : 'none';
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
      const scale = Math.min(1, SEND_WIDTH / video.videoWidth);
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
    if (vw && vh) {
      // Boxes are normalized [0,1]; map them onto the letterboxed (object-fit: contain) video.
      const s = Math.min(W / vw, H / vh), dw = vw * s, dh = vh * s, ox = (W - dw) / 2, oy = (H - dh) / 2;
      ctx.lineWidth = 3; ctx.font = '600 14px system-ui, sans-serif';
      for (const d of dets) {
        const x = ox + d.x1 * dw, y = oy + d.y1 * dh, w = (d.x2 - d.x1) * dw, h = (d.y2 - d.y1) * dh;
        const color = d.alert ? '#ef4444' : (d.label === 'bag' ? '#a78bfa' : '#22c55e');
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
