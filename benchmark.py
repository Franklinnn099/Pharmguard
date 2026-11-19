"""
Server-side pipeline benchmark for PharmaGuard (compute only - no network).

Measures every per-frame stage the server performs, INCLUDING the return-frame encoding that Gradio
does after our handler returns (previously unmeasured, and the dominant cost with WebP):

    downscale -> detect -> track -> behaviour -> annotate -> Gradio output encode

Input is a real photo containing people (Ultralytics' bus.jpg) or any video file via --video, so
tracking/behaviour timings reflect actual detections rather than empty frames.

For the full browser-equivalent round trip (upload + network + download) use tools/e2e_benchmark.py.

    python benchmark.py                       # sample photo, 1280x720 input
    python benchmark.py --video staged.mp4    # recorded staged scenario
    python benchmark.py --imgsz 480 --max-width 480
"""

import argparse
import os
import statistics
import time

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import cv2
import numpy as np
import torch

from pharmaguard.config import PharmaGuardConfig
from pharmaguard.pipeline import PharmaGuardPipeline


def load_frames(video: str | None, width: int, height: int, count: int) -> list[np.ndarray]:
    """RGB frames from a video, or the sample photo with a slight pan."""
    frames: list[np.ndarray] = []
    if video:
        cap = cv2.VideoCapture(video)
        while len(frames) < count:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(cv2.resize(f, (width, height)), cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise SystemExit(f"Could not read frames from {video}")
    else:
        import ultralytics
        img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
        base = cv2.cvtColor(cv2.resize(img, (width, height)), cv2.COLOR_BGR2RGB)
        frames = [np.roll(base, int(8 * np.sin(i / 6.0)), axis=1) for i in range(count)]
    return frames


def p(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def run_benchmark(num_frames: int = 100, width: int = 1280, height: int = 720, video: str | None = None,
                  imgsz: int | None = None, max_width: int | None = None, output_format: str | None = None,
                  model: str | None = None, bags: bool = True) -> dict:
    import gradio as gr

    config = PharmaGuardConfig()
    config.alert.alert_log_file = None  # benchmarking must not write audit records
    config.recording.enabled = False    # ...nor events/clips into the operational database
    if imgsz:
        config.detector.imgsz = imgsz
    if max_width:
        config.stream.max_process_width = max_width
    if output_format:
        config.stream.output_format = output_format
    if model:
        config.detector.model_name = model
    config.objects.enabled = bags

    pipeline = PharmaGuardPipeline(config)
    encoder = gr.Image(type="numpy", format=config.stream.output_format)  # exactly what the UI uses
    device = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else f"CPU (torch {torch.__version__})"

    frames = load_frames(video, width, height, num_frames + 5)
    for f in frames[:5]:  # warm-up (CUDA kernels, allocator, encoder)
        encoder.postprocess(pipeline.process_frame(f).annotated_frame_rgb)

    stages: dict[str, list[float]] = {k: [] for k in ("detect", "objects", "track", "behavior", "annotate", "pipeline", "encode", "total")}
    detections: list[int] = []
    for f in frames[5:]:
        t0 = time.perf_counter()
        out = pipeline.process_frame(f)
        t1 = time.perf_counter()
        encoder.postprocess(out.annotated_frame_rgb)
        t2 = time.perf_counter()
        st = out.metadata["stage_timings"]
        stages["detect"].append(st["detect_ms"])
        stages["track"].append(st["track_ms"])
        stages["objects"].append(st.get("objects_ms", 0.0))
        stages["behavior"].append(st["behavior_ms"])
        stages["annotate"].append(st["annotate_ms"])
        stages["pipeline"].append((t1 - t0) * 1000)
        stages["encode"].append((t2 - t1) * 1000)
        stages["total"].append((t2 - t0) * 1000)
        detections.append(out.metadata["detections_count"])

    n = len(stages["total"])
    print("=" * 72)
    print("PHARMAGUARD SERVER-SIDE BENCHMARK (compute only; network NOT included)")
    print("=" * 72)
    print(f"Device            : {device}")
    print(f"Model / imgsz     : {pipeline.model_name} / {pipeline.detector.imgsz}"
          f"{' (FP16)' if getattr(pipeline.detector, 'half', False) else ''}")
    print(f"Input -> processed: {width}x{height} -> {out.metadata['resolution']}  "
          f"(source: {'video ' + video if video else 'sample photo'})")
    print(f"Return encoding   : {config.stream.output_format}")
    print(f"Bag detector      : {'every ' + str(config.objects.frame_stride) + ' frames' if config.objects.enabled else 'disabled'}"
          f"  ('objects' row averages in the frames where it does not run)")
    print(f"Frames measured   : {n}   mean detections/frame: {statistics.mean(detections):.1f}")
    print("-" * 72)
    print(f"{'stage':<12}{'p50 ms':>10}{'p95 ms':>10}{'mean ms':>10}")
    for k, v in stages.items():
        print(f"{k:<12}{p(v, 50):>10.1f}{p(v, 95):>10.1f}{statistics.mean(v):>10.1f}")
    print("-" * 72)
    cap_fps = 1000.0 / statistics.mean(stages["total"])
    print(f"Server compute ceiling: {cap_fps:.1f} FPS  (delivered FPS will be lower: add network RTTs)")
    print("=" * 72)

    return {"device": device, "frames": n, **{f"{k}_p50_ms": p(v, 50) for k, v in stages.items()},
            "server_ceiling_fps": cap_fps}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--video", default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--max-width", type=int, default=None)
    ap.add_argument("--format", dest="output_format", default=None, help="jpeg (default) | webp | png")
    ap.add_argument("--model", default=None, help="Person model, e.g. yolov8n.pt (no hand analysis)")
    ap.add_argument("--no-bags", action="store_true", help="Disable the bag detector")
    a = ap.parse_args()
    run_benchmark(a.frames, a.width, a.height, a.video, a.imgsz, a.max_width, a.output_format,
                  model=a.model, bags=not a.no_bags)
