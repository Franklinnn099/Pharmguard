"""
Stream telemetry for PharmaGuard.

Gradio's webcam client only captures a new frame once the previous request is no longer pending
(frontend: ``setInterval(() => { if (!pending) capture() }, stream_every)``). The server therefore sees
a closed loop:

    arrival[n] -> processing -> return[n] -> (encode + download + browser + next capture + upload) -> arrival[n+1]

From timestamps taken purely on the server we can measure:

* ``cycle_ms``       = arrival[n+1] - arrival[n]   -> the true end-to-end frame period (1 / delivered FPS).
* ``processing_ms``  = return[n] - arrival[n]      -> time spent inside our pipeline.
* ``outside_ms``     = arrival[n+1] - return[n]    -> Gradio encoding + network + browser + capture wait.

``cycle_ms`` is an upper bound on a frame's capture-to-display latency (it additionally includes at most
one ``stream_every`` capture wait). These figures are only meaningful with a single streaming client.
"""

from dataclasses import dataclass
from collections import deque
from typing import Callable, Deque, Optional
import time

import numpy as np


@dataclass
class TelemetrySnapshot:
    """Point-in-time view of stream health, safe to hand to the UI thread."""
    signal: str                  # "LIVE", "NO_SIGNAL", "STANDBY"
    frames_total: int
    delivered_fps: float         # 1000 / median cycle
    cycle_p50_ms: float
    cycle_p95_ms: float
    processing_p50_ms: float
    processing_p95_ms: float
    outside_p50_ms: float        # encode + network + browser share of the cycle
    seconds_since_frame: Optional[float]
    interruptions: int           # count of gaps longer than the signal timeout
    client_rtt_p50_ms: float = 0.0   # browser-measured POST round trip (fast stream only; 0 = not reported)
    client_rtt_p95_ms: float = 0.0


def _pct(values: Deque[float], q: float) -> float:
    return float(np.percentile(np.fromiter(values, dtype=np.float64), q)) if values else 0.0


class StreamTelemetry:
    """Rolling-window stream statistics derived from server-side frame timestamps."""

    def __init__(self, window: int = 60, signal_timeout_sec: float = 2.0,
                 clock: Callable[[], float] = time.perf_counter):
        self.window = window
        self.signal_timeout_sec = signal_timeout_sec
        self._clock = clock
        self.reset()

    def reset(self) -> None:
        self._cycle_ms: Deque[float] = deque(maxlen=self.window)
        self._processing_ms: Deque[float] = deque(maxlen=self.window)
        self._outside_ms: Deque[float] = deque(maxlen=self.window)
        self._client_rtt_ms: Deque[float] = deque(maxlen=self.window)
        self._last_arrival: Optional[float] = None
        self._last_return: Optional[float] = None
        self.frames_total = 0
        self.interruptions = 0

    def mark_arrival(self) -> float:
        """Call when a frame enters the pipeline. Returns the arrival timestamp."""
        now = self._clock()
        if self._last_arrival is not None:
            gap = now - self._last_arrival
            if gap > self.signal_timeout_sec:
                # Stream was interrupted: do not let the outage pollute latency statistics.
                self.interruptions += 1
            else:
                self._cycle_ms.append(gap * 1000.0)
                if self._last_return is not None:
                    self._outside_ms.append(max(0.0, (now - self._last_return) * 1000.0))
        self._last_arrival = now
        self.frames_total += 1
        return now

    def mark_return(self, arrival_ts: float) -> float:
        """Call when the processed frame is handed back to the transport. Returns processing ms."""
        now = self._clock()
        processing = (now - arrival_ts) * 1000.0
        self._processing_ms.append(processing)
        self._last_return = now
        return processing

    def record_client_rtt(self, rtt_ms: float) -> None:
        """Round trip measured in the browser (capture-encode excluded, upload+process+download included)."""
        self._client_rtt_ms.append(float(rtt_ms))

    def snapshot(self) -> TelemetrySnapshot:
        now = self._clock()
        since = None if self._last_arrival is None else now - self._last_arrival
        if since is None:
            signal = "STANDBY"
        elif since > self.signal_timeout_sec:
            signal = "NO_SIGNAL"
        else:
            signal = "LIVE"
        cycle_p50 = _pct(self._cycle_ms, 50)
        return TelemetrySnapshot(
            signal=signal,
            frames_total=self.frames_total,
            delivered_fps=round(1000.0 / cycle_p50, 1) if cycle_p50 > 0 else 0.0,
            cycle_p50_ms=round(cycle_p50, 1),
            cycle_p95_ms=round(_pct(self._cycle_ms, 95), 1),
            processing_p50_ms=round(_pct(self._processing_ms, 50), 1),
            processing_p95_ms=round(_pct(self._processing_ms, 95), 1),
            outside_p50_ms=round(_pct(self._outside_ms, 50), 1),
            seconds_since_frame=None if since is None else round(since, 2),
            interruptions=self.interruptions,
            client_rtt_p50_ms=round(_pct(self._client_rtt_ms, 50), 1),
            client_rtt_p95_ms=round(_pct(self._client_rtt_ms, 95), 1),
        )
