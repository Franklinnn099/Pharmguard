# PharmaGuard — Run Guide (1 Dec demo)

**Main path: laptop camera server + iPhone on the same Wi-Fi or hotspot.** No internet is used for frames.
Why not Colab live: TECHNICAL_DECISIONS §22 (the iPhone reached 0.5-1.1 fps through the tunnel from Ghana; no alarm fired).

Run **only the camera server** on the laptop (not the Gradio dashboard as well): the laptop ran out of memory with both.

---

## 1. Before you start (once per network)

1. Connect the laptop and the iPhone to the **same** Wi-Fi or the same phone hotspot.
2. Find the laptop's IP on that network: start the server once (step 2). The banner prints `LAN IP: ...`.
3. **The HTTPS certificate is only valid for the IP it was made for** (it was made for `192.168.1.109`). On a new network
   or hotspot the IP changes. Regenerate it, then restart the server:
   ```powershell
   cd C:\Users\USER\Desktop\pharmguard\pharmguard-camera\server
   mkcert -cert-file certs\cert.pem -key-file certs\key.pem <NEW_LAN_IP> localhost 127.0.0.1
   ```
   The iPhone's mkcert profile stays valid. You do not need to install it again.
4. Windows firewall: if it asks, allow Python on **Private** networks. Mark a hotspot as Private, or the phone
   cannot connect.

## 2. Start the laptop demo

```powershell
cd C:\Users\USER\Desktop\pharmguard\pharmguard-camera\server
.\.venv\Scripts\python main.py
```

The banner shows: `Detector: pharmaguard (yolov8n-pose.pt on cpu)`, `Objects: off`, `In flight: 1`, the **Phone URL**
(`https://<LAN_IP>:8443/?cam=phone1&start=1`), the Monitor link and a QR code.

Optional:
- `$env:PG_ITEMS = "1"` before starting turns on the "every object" boxes. Display only, and they halve the fps on the
  laptop, so leave them off for the demo.
- `$env:PG_ACCESS_KEY = "<something long>"` locks every page behind the QR link. Use it if other people share the network.

## 3. Pair the iPhone

1. Scan the QR code in the terminal with the iPhone camera. The link opens in Safari, the camera starts, and Safari asks
   for camera access: tap **Allow**.
2. The **rear** camera must open, not mirrored: hold up printed text and it must read normally. If the front camera
   opens, tap **Stop**, then **Start**. **Flip camera** switches for this session only.
3. Hold the phone **landscape**. While it is upright, a yellow "Rotate to landscape" hint shows.
4. Check the **fps badge** (top right) is **5 or more**. A yellow "too slow for reliable detection" warning means
   weak Wi-Fi: move closer to the router or hotspot.
5. iPhone: **Settings → Display & Brightness → Auto-Lock → Never**. Keep Safari in the foreground and the phone plugged in.
6. Laptop: open the **Monitor** link from the banner (`http://<LAN_IP>:8000/monitor`) to watch the annotated feed live.

If the server's certificate is new, the iPhone needs the one-time profile: `pharmguard-camera/README.md` → "iPhone".

## 4. Draw the shelf zone

1. On the phone tap **Zones** (or on the laptop open the tile's **Zones** link on the Monitor page). The zone editor shows
   the last frame from that camera.
2. Tap the corners of the **front of the shelf** where hands reach in (3-24 corners), type a name, tap **Add zone**,
   then **Save & apply**. It applies on the next frame and is stored in `pharmguard-camera/server/zones/phone1.json`.
3. Camera placement: about 45° to the side of the shelf, so the reaching arm and the person's hips are visible
   (`docs/DEMO_STAGING.md`).

## 5. Checks before the audience arrives (Section 9 checks, phone version)

| Test | Action | Expected |
|---|---|---|
| 1. Stream up | Phone streaming | Monitor tile `live`, fps 5+ (no yellow warning) |
| 2. People + IDs | One, then two people walk in | A box per person with a stable ID; IDs survive crossing |
| 3. Wrists | Raise a hand | Wrist markers on the Monitor feed |
| 4. Shelf contact | Reach an arm into the zone | The zone on the Monitor feed changes to its CONTACT highlight |
| 5. Browsing | Reach in several times, look at items at chest height, put them back | **No** review, no alarm |
| 6. Staged concealment | Reach in, then within 1-2 s put the hand into a handbag/pocket and walk away | Amber **For staff review**, then red **ALARM** + siren on the phone; **Acknowledge** silences it, **Reset alarm** clears it |
| 7. Restart | Tap **Stop** on the phone, wait 5 s, tap **Start** | Tile shows offline, then live again; still the rear camera |

Evidence clips and the event database for phone cameras are saved in `pharmguard-camera/server/data/`. They contain faces:
staged, consenting participants only, and never commit `data/`.

## 6. If something goes wrong

| Symptom | Fix |
|---|---|
| iPhone says the connection is not private | Certificate made for another IP: redo step 1.3 and restart |
| Phone cannot reach the laptop | Same network? Windows firewall (Private)? Hotspot set to Private? |
| fps badge yellow (< 5) | Move closer to the router/hotspot; close other apps on the laptop; keep `PG_ITEMS` off |
| "Access key missing or wrong" | Scan the current QR again (a new `PG_ACCESS_KEY` invalidates old links) |
| Laptop runs out of memory | Run only the camera server; close the Gradio dashboard |

---

## Appendix: Colab, GPU evidence only (not the live demo)

`colab/PharmaGuard_Colab.ipynb`. Run **only one option per Colab session** (T4 GPU runtime).

- **Option 1 (Sections 2-9):** the dashboard through a `gradio.live` link. Section 7 `benchmark.py` measured on a T4:
  pipeline 15.4 ms, 17.3 ms p50 with encoding (56.6 fps compute ceiling).
- **Option 2 (cells O2-1 … O2-12):** camera server on the GPU behind a Cloudflare quick tunnel, the iPhone pairs from the
  O2-9 QR code, and the access key is always set. Measured from inside Colab (M-1): 21-31 fps, round trip p50 45-70 ms. From
  the iPhone in Ghana: **0.5-1.1 fps**. That is why it is evidence, not the demo. O2-10 (dashboard alongside) has not
  been run on Colab.

Full numbers: TECHNICAL_DECISIONS §22. Laptop CPU numbers for comparison: 9.4 fps, round trip p50 93 ms (scan off, 1 in flight).
