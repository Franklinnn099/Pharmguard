// Manual browser check (not part of pytest). Needs Chrome + npm install playwright-core@1.49.1.
// Start the server on port 8100 with PG_ACCESS_KEY=k3y-test PG_DETECTOR=none, then: node pharmguard-camera/tools/pwa_access_key_check.js
const { chromium } = require('playwright-core');
(async () => {
  const b = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true,
    args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'] });
  let ok = true; const say = (c, m) => { console.log((c ? 'PASS ' : 'FAIL ') + m); ok = ok && c; };
  const ctx = await b.newContext({ permissions: ['camera'] }); const page = await ctx.newPage();
  // 1) no key: clear error, no streaming
  await page.goto('http://127.0.0.1:8100/?cam=k1&start=1'); await page.waitForTimeout(3000);
  say((await page.textContent('#err')).includes('Access key'), 'no key: PWA shows access-key error');
  // 2) QR link with key: streams; then reload WITHOUT key in URL still connects (stored key)
  await page.goto('http://127.0.0.1:8100/?cam=k1&start=1&key=k3y-test');
  await page.waitForFunction(() => /live/.test(document.getElementById('status').textContent), null, { timeout: 15000 });
  say(true, 'QR link with key: streaming live');
  await page.goto('http://127.0.0.1:8100/?cam=k1&start=1');
  await page.waitForFunction(() => /live/.test(document.getElementById('status').textContent), null, { timeout: 15000 });
  say(true, 'reload without key in URL: still live (stored key)');
  // 3) monitor + zones pages work via cookie; status API shows the camera
  const r = await page.goto('http://127.0.0.1:8100/monitor'); say(r.status() === 200, 'monitor via cookie: ' + r.status());
  const z = await page.goto('http://127.0.0.1:8100/zones/k1'); say(z.status() === 200, 'zones via cookie: ' + z.status());
  // 4) fresh browser without cookie/key is refused
  const c2 = await b.newContext(); const p2 = await c2.newPage();
  const m = await p2.goto('http://127.0.0.1:8100/monitor'); say(m.status() === 401, 'stranger on /monitor: ' + m.status());
  console.log(ok ? 'ALL PASSED' : 'FAILED'); await b.close(); process.exit(ok ? 0 : 1);
})().catch((e) => { console.error(e); process.exit(2); });
