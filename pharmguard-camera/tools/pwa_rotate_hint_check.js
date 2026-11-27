// Manual browser check (not part of pytest). Needs Chrome + npm install playwright-core@1.49.1.
// Start the server with PG_DETECTOR=none PG_PORT=8100, then: node pharmguard-camera/tools/pwa_rotate_hint_check.js http://127.0.0.1:8100
// PWA: "Rotate to landscape" hint shows for portrait frames, hides for landscape, never blocks streaming.
const { chromium } = require('playwright-core');
const BASE = process.argv[2];
const STUB = (w, h) => `window.__sent=0;navigator.mediaDevices.getUserMedia=async()=>{const cv=document.createElement('canvas');
  cv.width=${w};cv.height=${h};const g=cv.getContext('2d');setInterval(()=>{g.fillStyle='#'+((Math.random()*0xffffff)|0).toString(16).padStart(6,'0');g.fillRect(0,0,${w},${h})},50);return cv.captureStream(15)};
  const s=WebSocket.prototype.send;WebSocket.prototype.send=function(d){if(d instanceof Blob)window.__sent++;return s.call(this,d)};`;
(async () => {
  const b = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
  let ok = true; const say = (c, m) => { console.log((c ? 'PASS ' : 'FAIL ') + m); ok = ok && c; };
  for (const [w, h, want] of [[480, 640, true], [640, 480, false]]) {
    const ctx = await b.newContext(); await ctx.addInitScript(STUB(w, h)); const p = await ctx.newPage();
    await p.goto(BASE + '/?cam=rot' + w + '&start=1');
    await p.waitForFunction(() => window.__sent >= 5, null, { timeout: 15000 });
    await p.waitForTimeout(500);
    const vis = await p.isVisible('#rotate');
    const pe = await p.evaluate(() => getComputedStyle(document.getElementById('rotate')).pointerEvents);
    say(vis === want, `${w}x${h}: hint ${vis ? 'shown' : 'hidden'} (want ${want ? 'shown' : 'hidden'}), frames sent ${await p.evaluate(() => window.__sent)}`);
    if (want) {
      say(pe === 'none', 'hint does not take taps (pointer-events ' + pe + ')');
      await p.click('#go');                                    // Stop still works with the hint up
      say(!(await p.isVisible('#rotate')) && (await p.textContent('#go')) === 'Start', 'Stop works and hides the hint');
    }
    await ctx.close();
  }
  console.log(ok ? 'ALL PASSED' : 'FAILED'); await b.close(); process.exit(ok ? 0 : 1);
})().catch((e) => { console.error(e); process.exit(2); });
