"""
Run the PharmaGuard pipeline over a recorded video and report tracking quality.

Uses the VIDEO's timestamps (not wall-clock), so results are identical however fast the machine is.
Typical uses:
  * Evaluate staged scenarios (crossing paths, occlusion, multiple people) recorded on the demo laptop.
  * Produce an annotated fallback video for the presentation if the Colab tunnel fails.

    python tools/track_video.py staged_crossing.mp4 --out crossing_annotated.mp4
    python tools/track_video.py staged.mp4 --fps 5        # simulate a 5 FPS tunnel by frame skipping

Report: IDs issued, max simultaneous people, and per-ID lifetime / occlusion counts. For a staged clip
with N actors, "IDs issued" > N means ID switches or false tracks - inspect the annotated output.

Behaviour analysis also runs on video time, so the REVIEW / ALERT events listed at the end are exactly
what the live system would raise for the same footage. Use this to count false alerts and missed events
on staged scenarios (docs/DEMO_STAGING.md).
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from pharmaguard.config import PharmaGuardConfig  # noqa: E402
from pharmaguard.pipeline import PharmaGuardPipeline  # noqa: E402


@dataclass
class IdStats:
    first_t: float
    last_t: float
    frames_visible: int = 0
    frames_occluded: int = 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", default=None, help="Write annotated MP4 here")
    ap.add_argument("--fps", type=float, default=None, help="Process at this rate by skipping frames (simulate tunnel FPS)")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--config", default=None, help="YAML config (zones, thresholds)")
    ap.add_argument("--record-dir", default=None,
                    help="Also write events + evidence clips to this directory (off by default so offline "
                         "evaluation never touches the operational event database)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    config = PharmaGuardConfig.load_from_yaml(args.config) if args.config else PharmaGuardConfig()
    config.alert.alert_log_file = None  # offline evaluation must not write operational audit records
    config.recording.enabled = bool(args.record_dir)
    if args.record_dir:
        config.recording.data_dir = args.record_dir
        config.recording.camera_id = f"video:{os.path.basename(args.video)}"
    pipeline = PharmaGuardPipeline(config)

    writer = None
    stats: dict[int, IdStats] = {}
    max_simultaneous = 0
    flagged = []
    processed = 0
    next_t = 0.0
    idx = -1
    t0 = time.perf_counter()
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        idx += 1
        t = idx / src_fps
        if args.max_seconds and t > args.max_seconds:
            break
        if args.fps and t + 1e-9 < next_t:
            continue
        next_t = t + (1.0 / args.fps if args.fps else 0.0)

        out = pipeline.process_frame(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), timestamp=t)
        processed += 1
        for a in out.metadata.get("new_alerts", []):
            flagged.append((t, a))
        persons = pipeline.tracker.tracked_registry
        visible = [tid for tid, p in persons.items() if p.frames_lost == 0]
        max_simultaneous = max(max_simultaneous, len(visible))
        for tid, p in persons.items():
            s = stats.setdefault(tid, IdStats(first_t=t, last_t=t))
            if p.frames_lost == 0:
                s.frames_visible += 1
                s.last_t = t
            else:
                s.frames_occluded += 1

        if args.out:
            annotated = cv2.cvtColor(out.annotated_frame_rgb, cv2.COLOR_RGB2BGR)
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps or src_fps, (w, h))
            writer.write(annotated)
    pipeline.close()                    # flush evidence clips
    wall = time.perf_counter() - t0
    cap.release()
    if writer is not None:
        writer.release()

    print("=" * 66)
    print(f"Video            : {args.video}  ({src_fps:.1f} FPS source)")
    print(f"Processed frames : {processed}  (at {args.fps or src_fps:.1f} FPS of video time)")
    print(f"Compute speed    : {processed / wall:.1f} frames/s on {pipeline.device.upper()}")
    print(f"IDs issued       : {len(stats)}")
    print(f"Max simultaneous : {max_simultaneous}")
    print("-" * 66)
    print(f"{'ID':>4} {'first s':>8} {'last s':>8} {'visible fr':>11} {'occluded fr':>12}")
    for tid in sorted(stats):
        s = stats[tid]
        print(f"{tid:>4} {s.first_t:>8.2f} {s.last_t:>8.2f} {s.frames_visible:>11} {s.frames_occluded:>12}")
    print("=" * 66)
    print(f"Staff-review events: {len(flagged)}")
    for t, a in flagged:
        print(f"  t={t:6.1f}s  {a.severity:<7} person #{a.tracker_id}  score {a.score:.2f}  incident {a.incident_id}")
        for r in a.reasons:
            print(f"             - {r}")
    print("=" * 66)
    if args.out:
        print(f"Annotated video  : {args.out}")
    if args.record_dir:
        print(f"Events + clips   : {os.path.abspath(args.record_dir)}")


if __name__ == "__main__":
    main()
