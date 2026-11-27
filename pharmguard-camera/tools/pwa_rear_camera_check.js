// Manual browser check (not part of pytest). Needs Chrome and: npm install playwright-core@1.49.1
// Start the server with PG_DETECTOR=none, then: node pharmguard-camera/tools/pwa_rear_camera_check.js http://127.0.0.1:8000
const { chromium } = require('playwright-core');
const BASE = process.argv[2];
const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
let fail = 0;
const check = (ok, msg) => { console.log((ok ? 'PASS ' : 'FAIL ') + msg); if (!ok) fail++; };

// Records every getUserMedia constraint; optional "phone" mode serves a canvas stream (left red / right blue)
// for any facingMode, so 'exact' succeeds like on a real phone. Also captures uploaded JPEG frames.
const INIT = (phone) => `
  window.__calls = []; window.__sent = [];
  const real = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async (c) => {
    window.__calls.push(JSON.stringify(c.video.facingMode));
    if (!${phone}) return real(c);
    const cv = document.createElement('canvas'); cv.width = 640; cv.height = 480;
    const g = cv.getContext('2d');
    const paint = () => { g.fillStyle = '#f00'; g.fillRect(0, 0, 320, 480); g.fillStyle = '#00f'; g.fillRect(320, 0, 320, 480); };
    paint(); setInterval(paint, 50);
    return cv.captureStream(15);
  };
  const send = WebSocket.prototype.send;
  WebSocket.prototype.send = function (d) { if (d instanceof Blob && window.__sent.length < 3) window.__sent.push(d); return send.call(this, d); };
`;

async function firstPixels(page) {   // decode the first uploaded frame, sample left and right
  return page.evaluate(async () => {
    const bmp = await createImageBitmap(window.__sent[0]);
    const c = document.createElement('canvas'); c.width = bmp.width; c.height = bmp.height;
    const g = c.getContext('2d'); g.drawImage(bmp, 0, 0);
    const px = (x) => Array.from(g.getImageData(x, bmp.height >> 1, 1, 1).data.slice(0, 3));
    return { left: px(10), right: px(bmp.width - 10) };
  });
}

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true,
    args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'] });

  // A) Laptop-like fake camera (no rear camera): exact must fail, ideal fallback must open the camera.
  let ctx = await browser.newContext({ permissions: ['camera'] });
  await ctx.addInitScript(INIT(false));
  let page = await ctx.newPage();
  await page.goto(BASE + '/?cam=t1&start=1');
  await page.waitForFunction(() => window.__sent.length > 0, null, { timeout: 15000 });
  let calls = await page.evaluate(() => window.__calls);
  check(calls[0] === '{"exact":"environment"}', 'laptop: first request is exact environment: ' + calls[0]);
  check(calls[1] === '{"ideal":"environment"}', 'laptop: falls back to ideal environment: ' + calls[1]);
  check(await page.evaluate(() => !!document.getElementById('v').srcObject), 'laptop: camera opened and frames uploaded');
  await ctx.close();

  // B) Phone-like: exact environment succeeds, no fallback; frames not mirrored; flip; reload resets to rear.
  ctx = await browser.newContext({ permissions: ['camera'] });
  await ctx.addInitScript(INIT(true));
  page = await ctx.newPage();
  await page.goto(BASE + '/?cam=t2&start=1');
  await page.waitForFunction(() => window.__sent.length > 0, null, { timeout: 15000 });
  calls = await page.evaluate(() => window.__calls);
  check(calls.length === 1 && calls[0] === '{"exact":"environment"}', 'phone: rear camera via exact, no fallback: ' + calls);
  const px = await firstPixels(page);
  check(px.left[0] > 200 && px.left[2] < 60 && px.right[2] > 200 && px.right[0] < 60,
    'phone: uploaded frame not mirrored (left red, right blue): ' + JSON.stringify(px));
  const tf = await page.evaluate(() => getComputedStyle(document.getElementById('v')).transform);
  check(tf === 'none', 'phone: preview not mirrored (transform=' + tf + ')');
  check(await page.isVisible('#flip'), 'phone: Flip button visible while running');
  await page.click('#flip');
  await page.waitForFunction(() => window.__calls.length >= 2);
  calls = await page.evaluate(() => window.__calls);
  check(calls[1] === '{"exact":"user"}', 'phone: Flip asks for the front camera: ' + calls[1]);
  const saved = await page.evaluate(() => JSON.stringify(localStorage));
  check(!/user|facing|device/i.test(saved), 'phone: camera choice not saved: ' + saved);
  await page.reload();
  await page.waitForFunction(() => window.__calls.length >= 1, null, { timeout: 15000 });
  calls = await page.evaluate(() => window.__calls);
  check(calls[0] === '{"exact":"environment"}', 'phone: after reload starts on rear again: ' + calls[0]);
  // Stop/Start (manual restart) also returns to rear after a flip.
  await page.click('#flip'); await page.waitForFunction(() => window.__calls.length >= 2);
  await page.click('#go'); await page.click('#go');
  await page.waitForFunction(() => window.__calls.length >= 3);
  calls = await page.evaluate(() => window.__calls);
  check(calls[2] === '{"exact":"environment"}', 'phone: Stop then Start returns to rear: ' + calls[2]);
  await ctx.close();
  await browser.close();
  console.log(fail ? fail + ' FAILED' : 'ALL PASSED');
  process.exit(fail ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(2); });
