"""
Performance and latency benchmark script for PharmaGuard.
Evaluates end-to-end processing throughput on synthetic video frames.
"""

import time
import numpy as np
import cv2
from pharmaguard.config import PharmaGuardConfig
from pharmaguard.pipeline import PharmaGuardPipeline


def run_benchmark(num_frames: int = 50, width: int = 640, height: int = 480):
    print("=" * 60)
    print("🛡️  PHARMAGUARD PIPELINE PERFORMANCE BENCHMARK")
    print(f"Frames: {num_frames} | Resolution: {width}x{height}")
    print("=" * 60)

    config = PharmaGuardConfig()
    pipeline = PharmaGuardPipeline(config)

    # Generate synthetic frames with moving shapes
    latencies = []

    print("\nWarmup run...")
    dummy = np.zeros((height, width, 3), dtype=np.uint8)
    pipeline.process_frame(dummy)

    print(f"Benchmarking {num_frames} frames...")
    for i in range(num_frames):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        # Draw a moving rectangle
        cx = int(100 + (i * 8) % (width - 200))
        cy = int(100 + (i * 4) % (height - 200))
        cv2.rectangle(frame, (cx, cy), (cx + 80, cy + 160), (200, 200, 200), -1)

        t0 = time.perf_counter()
        output = pipeline.process_frame(frame)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000.0)

    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0

    print("-" * 60)
    print(f"Mean Latency:        {avg_latency:.2f} ms")
    print(f"P95 Latency:         {p95_latency:.2f} ms")
    print(f"Effective Pipeline FPS: {fps:.1f} FPS")
    print(f"Active Tracks at end:   {output.active_tracks_count}")
    print(f"Device:                 {pipeline.detector.get_model_info().get('device')}")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
