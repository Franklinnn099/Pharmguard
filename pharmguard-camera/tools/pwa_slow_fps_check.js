// Manual browser check (not part of pytest). Needs Chrome + npm install playwright-core@1.49.1.
// Start the server with PG_DETECTOR=none PG_PORT=8100, then: node pharmguard-camera/tools/pwa_slow_fps_check.js http://127.0.0.1:8100
// PWA shows live fps; a throttled phone (~2 fps) gets the "too slow" warning on the phone AND on /monitor.
const { chromium } = require('playwright-core');
const BASE = process.argv[2];
const STUB = (delayMs) => `navigator.mediaDevices.getUserMedia=async()=>{const cv=document.createElement('canvas');cv.width=640;cv.height=480;
  const g=cv.getContext('2d');setInterval(()=>{g.fillStyle='#'+((Math.random()*0xffffff)|0).toString(16).padStart(6,'0');g.fillRect(0,0,640,480)},50);return cv.captureStream(15)};
  const s=WebSocket.prototype.send;WebSocket.prototype.send=function(d){if(d instanceof Blob&&${delayMs}){setTimeout(()=>s.call(this,d),${delayMs});return}return s.call(this,d)};`;
(async () => {
  const b = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
  let ok = true; const say = (c, m) => { console.log((c ? 'PASS ' : 'FAIL ') + m); ok = ok && c; };
  const open = async (cam, delay) => { const c = await b.newContext(); await c.addInitScript(STUB(delay)); const p = await c.newPage();
    await p.goto(BASE + '/?cam=' + cam + '&start=1'); return p; };
  const fast = await open('fastcam', 0), slow = await open('slowcam', 450);
  await fast.waitForTimeout(9000);   // slow phone needs 10 replies (~5 s at 2 fps) before it is judged
  const fb = await fast.textContent('#fps'), sb = await slow.textContent('#fps');
  say(/^\d+\.\d fps$/.test(fb) && !(await fast.evaluate(() => document.getElementById('fps').classList.contains('slow'))), 'fast phone shows plain fps: "' + fb + '"');
  say(await slow.evaluate(() => document.getElementById('fps').classList.contains('slow')) && /too slow for reliable detection/.test(sb), 'slow phone warned: "' + sb + '"');
  const mon = await (await b.newContext()).newPage(); await mon.goto(BASE + '/monitor'); await mon.waitForTimeout(2500);
  const rows = await mon.evaluate(() => [...document.querySelectorAll('.tile')].map((t) => ({
    cam: t.querySelector('strong').textContent, slow: !t.querySelector('.slow').hidden, text: t.querySelector('.slow').textContent })));
  const r = Object.fromEntries(rows.map((x) => [x.cam, x]));
  say(r.slowcam && r.slowcam.slow, 'monitor warns for slowcam: "' + (r.slowcam && r.slowcam.text) + '"');
  say(r.fastcam && !r.fastcam.slow, 'monitor shows no warning for fastcam');
  console.log(ok ? 'ALL PASSED' : 'FAILED'); await b.close(); process.exit(ok ? 0 : 1);
})().catch((e) => { console.error(e); process.exit(2); });
