"""
Evidence clips for PharmaGuard.

``EvidenceRecorder`` keeps a time-bounded ring buffer of annotated frames (at most ``max_clip_fps``, so memory
is bounded: 5 s x 15 fps x 640x360x3 ~= 52 MB worst case). When an alert is recorded, a clip job collects the
pre-event frames plus ``post_event_sec`` of following frames, then a single background worker encodes it.
Nothing is encoded on the real-time path.

``ClipWriter`` produces browser-playable video where possible, trying in order:
  1. ffmpeg binary (libx264, yuv420p, faststart)  - available on Colab
  2. OpenCV 'avc1' (H.264 .mp4)                   - availability depends on the OpenCV build
  3. OpenCV 'VP80' (.webm)
  4. OpenCV 'mp4v' (.mp4)                         - saved, but NOT playable in browsers (status says so)
All timing uses frame timestamps, so recorded video produces the same clips as live input.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional, Tuple
import logging
import os
import queue
import re
import shutil
import subprocess
import threading

import cv2
import numpy as np

from pharmaguard.config import RecordingConfig

logger = logging.getLogger(__name__)

ClipDone = Callable[[str, str, Optional[str], Optional[str]], None]   # (event_uid, status, path, codec)


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


class ClipWriter:
    """Encodes a list of RGB frames to the most browser-compatible format available."""

    def __init__(self, prefer_ffmpeg: bool = True):
        self.ffmpeg = shutil.which("ffmpeg") if prefer_ffmpeg else None

    def write(self, frames: List[np.ndarray], fps: float, out_base: str) -> Tuple[Optional[str], str, str]:
        """Returns (path, status, codec). status: saved | saved_not_browser_playable | failed."""
        if not frames:
            return None, "failed", ""
        h, w = frames[0].shape[:2]
        h2, w2 = h - h % 2, w - w % 2                          # yuv420p needs even dimensions
        frames = [f[:h2, :w2] if f.shape[:2] != (h2, w2) else f for f in frames]
        fps = float(max(1.0, min(fps, 30.0)))

        if self.ffmpeg:
            path = out_base + ".mp4"
            try:
                proc = subprocess.Popen(
                    [self.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                     "-s", f"{w2}x{h2}", "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                     "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                for f in frames:
                    proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
                _, err = proc.communicate(timeout=120)
                if proc.returncode == 0 and os.path.getsize(path) > 0:
                    return path, "saved", "h264(ffmpeg)"
                logger.warning("ffmpeg clip encode failed (%s); falling back to OpenCV", err.decode(errors="ignore")[:200])
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                logger.warning("ffmpeg clip encode error: %s; falling back to OpenCV", exc)

        for fourcc, ext, playable in (("avc1", ".mp4", True), ("VP80", ".webm", True), ("mp4v", ".mp4", False)):
            path = out_base + ext
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, (w2, h2))
            if not writer.isOpened():
                writer.release()
                continue
            for f in frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return path, ("saved" if playable else "saved_not_browser_playable"), fourcc
        return None, "failed", ""


@dataclass
class _ClipJob:
    event_uid: str
    trigger_ts: float
    end_ts: float
    frames: List[Tuple[float, np.ndarray]] = field(default_factory=list)


class EvidenceRecorder:
    """Ring buffer + clip jobs + one background encoder thread."""

    def __init__(self, config: RecordingConfig, clips_dir: str, on_clip_done: Optional[ClipDone] = None,
                 writer: Optional[ClipWriter] = None):
        self.config = config
        self.clips_dir = clips_dir
        os.makedirs(clips_dir, exist_ok=True)
        self.on_clip_done = on_clip_done
        self.writer = writer or ClipWriter()
        self._buffer: Deque[Tuple[float, np.ndarray]] = deque()
        self._last_kept_ts: Optional[float] = None
        self._jobs: List[_ClipJob] = []
        self._lock = threading.Lock()
        self._queue: "queue.Queue[Optional[Callable[[], None]]]" = queue.Queue()
        self._worker = threading.Thread(target=self._run, name="pharmaguard-evidence", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------ real-time path (cheap)
    def add_frame(self, frame_rgb: np.ndarray, ts: float) -> None:
        cfg = self.config
        with self._lock:
            if self._last_kept_ts is None or ts - self._last_kept_ts >= 1.0 / cfg.max_clip_fps - 1e-6:
                self._last_kept_ts = ts
                self._buffer.append((ts, frame_rgb))
                for job in self._jobs:
                    # >= : the alert's own frame is annotated/buffered AFTER trigger() - include it.
                    if job.trigger_ts <= ts <= job.end_ts:
                        job.frames.append((ts, frame_rgb))
            while self._buffer and ts - self._buffer[0][0] > cfg.pre_event_sec:
                self._buffer.popleft()
            done = [j for j in self._jobs if ts >= j.end_ts]
            self._jobs = [j for j in self._jobs if ts < j.end_ts]
        for job in done:
            self._submit_job(job)

    def trigger(self, event_uid: str, ts: float) -> None:
        with self._lock:
            pre = [(t, f) for t, f in self._buffer if t >= ts - self.config.pre_event_sec]
            self._jobs.append(_ClipJob(event_uid, ts, ts + self.config.post_event_sec, frames=pre))

    # ------------------------------------------------------------------ background work
    def submit(self, fn: Callable[[], None]) -> None:
        """Run arbitrary slow work (e.g. database mirroring) on the evidence worker thread."""
        self._queue.put(fn)

    def _submit_job(self, job: _ClipJob) -> None:
        self._queue.put(lambda: self._encode(job))

    def _encode(self, job: _ClipJob) -> None:
        frames = [f for _, f in job.frames]
        if not frames:
            self._done(job.event_uid, "unavailable", None, None)
            return
        span = job.frames[-1][0] - job.frames[0][0]
        fps = (len(frames) - 1) / span if span > 0 else self.config.max_clip_fps
        base = os.path.join(self.clips_dir, _safe_name(job.event_uid))
        try:
            path, status, codec = self.writer.write(frames, fps, base)
        except Exception as exc:  # never let a codec problem kill the worker
            logger.exception("Evidence clip for %s failed: %s", job.event_uid, exc)
            path, status, codec = None, "failed", None
        self._done(job.event_uid, status, path, codec)

    def _done(self, uid: str, status: str, path: Optional[str], codec: Optional[str]) -> None:
        if self.on_clip_done is not None:
            try:
                self.on_clip_done(uid, status, path, codec)
            except Exception:
                logger.exception("on_clip_done callback failed for %s", uid)

    def _run(self) -> None:
        while True:
            fn = self._queue.get()
            if fn is None:
                self._queue.task_done()
                return
            try:
                fn()
            except Exception:
                logger.exception("Evidence worker task failed")
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------ lifecycle
    def flush(self) -> None:
        """Finalise in-progress clips with whatever frames they have (shutdown / end of video)."""
        with self._lock:
            jobs, self._jobs = self._jobs, []
        for job in jobs:
            self._submit_job(job)

    def wait_idle(self, timeout: Optional[float] = None) -> None:
        """Block until queued encodes/mirrors have finished (tests, shutdown)."""
        if timeout is None:
            self._queue.join()
            return
        done = threading.Event()
        threading.Thread(target=lambda: (self._queue.join(), done.set()), daemon=True).start()
        done.wait(timeout)

    @property
    def pending_jobs(self) -> int:
        with self._lock:
            return len(self._jobs)

    def close(self) -> None:
        self.flush()
        self._queue.put(None)
        self._worker.join(timeout=120)
