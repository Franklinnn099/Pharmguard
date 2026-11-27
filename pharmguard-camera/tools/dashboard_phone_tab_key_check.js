// Manual browser check (not part of pytest). Needs Chrome + npm install playwright-core@1.49.1.
// Camera server: PG_ACCESS_KEY=dash-k3y PG_DETECTOR=none PG_PORT=8100 PG_CORS_ORIGINS=http://127.0.0.1:7861
// Dashboard:     PG_ACCESS_KEY=dash-k3y python main.py --local --port 7861 --no-recording --phone-cameras http://127.0.0.1:8100
const { chromium } = require('playwright-core');
(async () => {
  const b = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true,
    args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'] });
  let ok = true; const say = (c, m) => { console.log((c ? 'PASS ' : 'FAIL ') + m); ok = ok && c; };
  // 1) a phone streams to the keyed camera server
  const pc = await b.newContext({ permissions: ['camera'] }); const phone = await pc.newPage();
  await phone.goto('http://127.0.0.1:8100/?cam=dashcam&start=1&key=dash-k3y');
  await phone.waitForFunction(() => /live/.test(document.getElementById('status').textContent), null, { timeout: 20000 });
  // 2) the dashboard (separate origin, no cookie for :8100) shows it in the Phone cameras tab
  const dc = await b.newContext(); const dash = await dc.newPage();
  const reqs = [];
  dash.on('request', (r) => { if (r.url().includes(':8100/')) reqs.push(r.url()); });
  const statuses = {};
  dash.on('response', (r) => { if (r.url().includes(':8100/')) statuses[r.url().split('?')[0]] = r.status(); });
  await dash.goto('http://127.0.0.1:7861/');
  await dash.getByRole('tab', { name: /Phone cameras/i }).click();
  await dash.waitForFunction(() => document.querySelectorAll('#pg-phones img').length > 0, null, { timeout: 20000 });
  await dash.waitForTimeout(3000);
  const img = await dash.evaluate(() => { const i = document.querySelector('#pg-phones img'); return { src: i.src, w: i.naturalWidth }; });
  const info = await dash.textContent('#pg-phones .pg-phone-info');
  say(reqs.some((u) => u.includes('/api/cameras?key=dash-k3y')), '/api/cameras sent with the key');
  say(statuses['http://127.0.0.1:8100/api/cameras'] === 200, '/api/cameras answered 200: ' + statuses['http://127.0.0.1:8100/api/cameras']);
  say(img.src.endsWith('/view/dashcam?key=dash-k3y') && img.w > 0, '/view/<cam> loads with the key: ' + img.src + ' width ' + img.w);
  say(!/not reachable/.test(info), 'tab shows server info: ' + info.slice(0, 90));
  // 3) acknowledge/reset go out with the key
  await dash.evaluate(() => window.pgPhoneAction('dashcam', 'reset')); await dash.waitForTimeout(800);
  say(reqs.some((u) => u.includes('/api/cameras/dashcam/reset?key=dash-k3y')) &&
      statuses['http://127.0.0.1:8100/api/cameras/dashcam/reset'] === 200, 'reset POST sent with the key -> ' + statuses['http://127.0.0.1:8100/api/cameras/dashcam/reset']);
  console.log(ok ? 'ALL PASSED' : 'FAILED'); await b.close(); process.exit(ok ? 0 : 1);
})().catch((e) => { console.error(e); process.exit(2); });
