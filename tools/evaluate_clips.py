"""
Score PharmaGuard on a folder of labelled clips (docs/EVALUATION_PLAN.md).

    python tools/evaluate_clips.py clips/labels.csv --config configs/demo_shelf.yaml --fps 5 --out results.csv

labels.csv columns: clip,scenario,expected[,notes][,config]   expected = none | review | alert
A per-clip `config` overrides --config (each camera view needs its own shelf zone).
Prints per-clip outcomes and the summary metrics for the report; writes per-clip rows to --out.
Every figure printed is measured on the clips given; nothing is estimated.
"""

import argparse
import os
import sys

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pharmaguard.config import PharmaGuardConfig  # noqa: E402
from pharmaguard.evaluation import load_labels, run_clip, score, write_results_csv  # noqa: E402


def pct(v):
    return "n/a" if v is None else f"{v:.0%}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels")
    ap.add_argument("--config", default=None, help="YAML config with the zones drawn for these clips")
    ap.add_argument("--fps", type=float, default=5.0,
                    help="Process at this rate of video time (default 5, close to the live tunnel)")
    ap.add_argument("--out", default="evaluation_results.csv")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    config = PharmaGuardConfig.load_from_yaml(args.config) if args.config else PharmaGuardConfig()
    results = []
    per_view: dict = {}
    for i, lab in enumerate(labels, 1):
        cfg = config
        if lab.config:
            cfg = per_view.setdefault(lab.config, PharmaGuardConfig.load_from_yaml(lab.config))
        r = run_clip(lab, cfg, args.fps)
        results.append(r)
        mark = "ERROR" if r.error else ("ok" if r.correct else "WRONG")
        print(f"[{i:>3}/{len(labels)}] {mark:<5} {r.clip:<28} {r.scenario:<22} expected {r.expected:<6} "
              f"got {r.predicted:<6} {r.error}")
    write_results_csv(results, args.out)
    s = score(results)
    print("=" * 72)
    print(f"Clips {s['clips']} (errors {s['errors']}): {s['concealment_clips']} concealment, {s['other_clips']} other")
    print(f"TP {s['tp']}  FN {s['fn']}  FP {s['fp']}  TN {s['tn']}")
    print(f"Recall (ALERT on concealment)     : {pct(s['recall'])}")
    print(f"Flag recall (REVIEW or ALERT)     : {pct(s['flag_recall'])}")
    print(f"Precision                         : {pct(s['precision'])}")
    print(f"False-alarm rate (ALERT on others): {pct(s['false_alarm_rate'])}")
    print(f"REVIEW rate on ordinary clips     : {pct(s['review_rate'])}")
    print(f"Clip accuracy                     : {pct(s['accuracy'])}")
    if s["median_first_alert_s"] is not None:
        print(f"Median time to first ALERT        : {s['median_first_alert_s']:.1f} s into the clip")
    print("-" * 72)
    for name, v in sorted(s["by_scenario"].items()):
        print(f"  {name:<28} {v['correct']}/{v['clips']} correct")
    print("=" * 72)
    print(f"Per-clip results: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
