"""
End-to-end transport benchmark for a running PharmaGuard server.

Run this ON THE LAPTOP against the Colab share URL to measure what the browser experiences:

    python tools/e2e_benchmark.py --url https://xxxx.gradio.live --user security_operator --password <pw>

Each iteration performs the same network round trip as the browser's webcam stream
(upload JPEG frame -> queue -> pipeline -> JPEG encode -> download annotated frame), sequentially,
exactly like Gradio's webcam client, which never sends a new frame while one is pending.
All requests share ONE persistent HTTP connection, as a browser tab does.

What it does NOT include: browser camera capture and on-screen paint (both typically < 1 frame).
Do not stream from a browser at the same time - frames would interleave in the shared tracker.
"""

import argparse
import json
import os
import statistics
import sys
import time

import cv2
import numpy as np


def load_frames(video: str | None, width: int, height: int, count: int) -> list[np.ndarray]:
    """Frames from a video file, or a real photo containing people (Ultralytics' bus.jpg)."""
    frames: list[np.ndarray] = []
    if video:
        cap = cv2.VideoCapture(video)
        while len(frames) < count:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.resize(f, (width, height)))
        cap.release()
        if not frames:
            sys.exit(f"Could not read frames from {video}")
    else:
        import ultralytics
        img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
        base = cv2.resize(img, (width, height))
        # Slight horizontal pan so the tracker sees motion rather than a frozen frame.
        for i in range(count):
            shift = int(8 * np.sin(i / 6.0))
            frames.append(np.roll(base, shift, axis=1))
    return frames


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


class GradioHTTP:
    """Minimal Gradio HTTP-API client on ONE persistent connection, like a browser tab.

    gradio_client was rejected for measurement: it opens new HTTP connections per call, and on the
    dev laptop each new httpx connection costs ~640 ms of client-side setup (measured 2026-11-19).
    That overhead would be reported as "latency" although no browser pays it.

    Protocol (Gradio documented HTTP API): POST /gradio_api/upload -> POST /gradio_api/call/<api>
    -> GET SSE /gradio_api/call/<api>/<event_id> -> GET annotated frame URL.
    """

    def __init__(self, url: str, user: str | None, password: str | None, api_name: str = "process_frame"):
        import httpx
        self.base = url.rstrip("/")
        self.api = api_name
        self.c = httpx.Client(timeout=30.0, follow_redirects=True)
        if user and password:
            r = self.c.post(f"{self.base}/login", data={"username": user, "password": password})
            if r.status_code >= 400:
                sys.exit(f"Login failed ({r.status_code}). Check --user/--password.")

    def round_trip(self, jpeg: bytes) -> dict[str, float]:
        t: dict[str, float] = {}
        t0 = time.perf_counter()
        r = self.c.post(f"{self.base}/gradio_api/upload", files={"files": ("frame.jpg", jpeg, "image/jpeg")})
        r.raise_for_status()
        server_path = r.json()[0]
        t1 = time.perf_counter()
        t["upload"] = (t1 - t0) * 1000

        body = {"data": [{"path": server_path, "meta": {"_type": "gradio.FileData"}}]}
        r = self.c.post(f"{self.base}/gradio_api/call/{self.api}", json=body)
        r.raise_for_status()
        event_id = r.json()["event_id"]
        t2 = time.perf_counter()
        t["submit"] = (t2 - t1) * 1000

        result = None
        with self.c.stream("GET", f"{self.base}/gradio_api/call/{self.api}/{event_id}") as resp:
            event = None
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:") and event in ("complete", "error"):
                    if event == "error":
                        raise RuntimeError(f"server error: {line[5:].strip()}")
                    result = json.loads(line[5:])
                    break
        t3 = time.perf_counter()
        t["wait_result"] = (t3 - t2) * 1000  # queue + PharmaGuard pipeline + JPEG encode
        if not result or not result[0] or not result[0].get("url"):
            raise RuntimeError(f"unexpected result payload: {result!r}")

        img = self.c.get(result[0]["url"])
        img.raise_for_status()
        t["download"] = (time.perf_counter() - t3) * 1000
        return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="Server URL, e.g. https://xxxx.gradio.live or http://127.0.0.1:7860")
    ap.add_argument("--user", default=os.environ.get("PHARMAGUARD_AUTH_USER"))
    ap.add_argument("--password", default=os.environ.get("PHARMAGUARD_AUTH_PASS"))
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--video", default=None, help="Optional video file to use instead of the sample photo")
    ap.add_argument("--jpeg-quality", type=int, default=80, help="Upload quality (browser uses 0.8)")
    args = ap.parse_args()

    http = GradioHTTP(args.url, args.user, args.password)
    frames = load_frames(args.video, args.width, args.height, args.frames + args.warmup)
    payloads = []
    for f in frames:
        _, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        payloads.append(buf.tobytes())

    print(f"Target: {args.url}  frames={args.frames} (+{args.warmup} warm-up)  "
          f"upload={args.width}x{args.height} JPEG q{args.jpeg_quality} "
          f"(~{statistics.mean(len(p) for p in payloads) / 1024:.0f} KB)")

    rtts: list[float] = []
    stages: dict[str, list[float]] = {k: [] for k in ("upload", "submit", "wait_result", "download")}
    failures = 0
    t_start = None
    for i, data in enumerate(payloads):
        if i == args.warmup:
            t_start = time.perf_counter()
        t0 = time.perf_counter()
        try:
            st = http.round_trip(data)
        except Exception as exc:  # network errors are data here, not crashes
            failures += 1
            print(f"  frame {i}: FAILED ({type(exc).__name__}: {exc})")
            continue
        if i >= args.warmup:
            rtts.append((time.perf_counter() - t0) * 1000.0)
            for k, v in st.items():
                stages[k].append(v)
    wall = time.perf_counter() - t_start if t_start else 0.0

    print("\n=== End-to-end round trip (upload + queue + process + encode + download) ===")
    if rtts:
        print(f"  frames OK        : {len(rtts)} / {args.frames}   failures: {failures}")
        print(f"  RTT p50 / p95    : {pct(rtts, 50):.0f} / {pct(rtts, 95):.0f} ms   (min {min(rtts):.0f}, max {max(rtts):.0f})")
        print(f"  sustained FPS    : {len(rtts) / wall:.1f}")
        for k, v in stages.items():
            print(f"    {k:<13}: p50 {pct(v, 50):6.0f} ms   p95 {pct(v, 95):6.0f} ms")
    else:
        print("  no successful frames")
    print("Note: browser streaming uses 4 HTTP exchanges per frame like this tool; "
          "over a tunnel each adds one network round trip.")


if __name__ == "__main__":
    main()
