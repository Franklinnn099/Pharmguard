"""
Pretend to be a phone: send JPEG frames to /ws/camera and check that one reply comes back per frame.

    python test_client.py                          # 10 frames to ws://127.0.0.1:8000, camera "testcam"
    python test_client.py --url ws://192.168.1.20:8000 --frames 30 --cam bench
    python test_client.py --url wss://192.168.1.20:8443 --cafile certs/ca.crt     # HTTPS port (iPhone path)

Frames are Ultralytics' bus.jpg sample (real people, so the pipeline has something to detect) when
ultralytics is installed, otherwise a synthetic image. Exits non-zero if any frame got no reply.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

import cv2
import numpy as np
import websockets


def sample_jpeg() -> bytes:
    img = None
    try:
        import ultralytics
        img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
    except ImportError:
        pass
    if img is None:
        img = np.full((480, 640, 3), 90, dtype=np.uint8)
        cv2.rectangle(img, (250, 100), (390, 440), (40, 160, 220), -1)
    h, w = img.shape[:2]
    if w > 640:
        img = cv2.resize(img, (640, int(h * 640 / w)))
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()


async def run(url: str, cam: str, frames: int, in_flight: int, cafile: str | None = None) -> int:
    jpeg = sample_jpeg()
    replies: dict[int, dict] = {}
    ssl_ctx = None
    if url.startswith("wss://"):
        import ssl
        ssl_ctx = ssl.create_default_context(cafile=cafile)
    async with websockets.connect(f"{url}/ws/camera?cam={cam}", max_size=8 * 1024 * 1024, ssl=ssl_ctx) as ws:
        hello = json.loads(await ws.recv())
        print(f"hello: {hello}")
        assert hello["type"] == "hello", hello
        in_flight = min(in_flight, int(hello.get("max_in_flight", in_flight)))
        sent, t0 = 0, time.perf_counter()
        while len(replies) < frames:
            while sent < frames and sent - len(replies) < in_flight:
                await ws.send(jpeg)
                sent += 1
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
            if msg["type"] in ("detections", "skipped"):
                replies[msg["frame"]] = msg
                if msg["type"] == "detections":
                    d = msg["detections"]
                    flagged = sum(1 for x in d if x.get("alert"))
                    print(f"  frame {msg['frame']:>3}: {len(d)} detection(s), {flagged} flagged, "
                          f"{msg['ms']:.0f} ms server, {msg['w']}x{msg['h']}, alarm {msg['alarm']}")
                else:
                    print(f"  frame {msg['frame']:>3}: skipped ({msg.get('reason', 'newer frame queued')})")
        elapsed = time.perf_counter() - t0
    processed = sum(1 for m in replies.values() if m["type"] == "detections")
    missing = [i for i in range(1, frames + 1) if i not in replies]
    order_ok = list(replies) == sorted(replies)
    print(f"\n{frames} sent, {len(replies)} acked ({processed} processed, {len(replies) - processed} skipped), "
          f"{frames / elapsed:.1f} frames/s round trip, replies in order: {order_ok}")
    last = next((m for m in reversed(list(replies.values())) if m["type"] == "detections"), None)
    if last and last["detections"]:
        print("sample detection:", json.dumps(last["detections"][0]))
    if missing or not order_ok:
        print(f"FAIL: missing replies for frames {missing}" if missing else "FAIL: replies out of order")
        return 1
    print("OK")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="ws://127.0.0.1:8000")
    ap.add_argument("--cam", default="testcam")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--in-flight", type=int, default=2)
    ap.add_argument("--cafile", default=None, help="CA certificate to trust for wss:// (e.g. certs/ca.crt)")
    a = ap.parse_args()
    sys.exit(asyncio.run(run(a.url.rstrip("/"), a.cam, a.frames, a.in_flight, a.cafile)))


if __name__ == "__main__":
    main()
