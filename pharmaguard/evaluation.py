"""
Offline evaluation of PharmaGuard on labelled video clips.

Each clip is labelled with what SHOULD happen: ``none`` (ordinary behaviour, no REVIEW/ALERT),
``review`` (a REVIEW is acceptable or expected, no ALERT) or ``alert`` (staged concealment, one ALERT).
The pipeline runs on the clip's own timestamps (as ``tools/track_video.py``), so results do not
depend on machine speed. Scoring is a pure function and unit-tested.

Metrics reported (clip level, the unit an examiner can check by watching the clip):
* recall           concealment clips that raised an ALERT / all concealment clips
* flag recall      concealment clips that raised a REVIEW or ALERT / all concealment clips
* precision        ALERT clips that were concealment / all clips that raised an ALERT
* false-alarm rate non-concealment clips that raised an ALERT / all non-concealment clips
* review rate      ``none`` clips that raised only a REVIEW / all ``none`` clips (operator workload)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import csv
import os

LABELS = ("none", "review", "alert")


@dataclass
class ClipLabel:
    clip: str
    scenario: str
    expected: str
    notes: str = ""


@dataclass
class ClipResult:
    clip: str
    scenario: str
    expected: str
    predicted: str                     # none | review | alert
    alerts: int = 0
    reviews: int = 0
    first_alert_s: Optional[float] = None
    ids_issued: int = 0
    reasons: List[str] = field(default_factory=list)
    error: str = ""

    @property
    def correct(self) -> bool:
        if self.expected == "alert":
            return self.predicted == "alert"
        if self.expected == "review":
            return self.predicted in ("review", "none")   # ALERT on a review clip counts as a false alarm
        return self.predicted == "none"


def load_labels(path: str) -> List[ClipLabel]:
    """CSV with header ``clip,scenario,expected[,notes]``. Clip paths are relative to the CSV's folder."""
    base = os.path.dirname(os.path.abspath(path))
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            exp = (row.get("expected") or "").strip().lower()
            if exp not in LABELS:
                raise ValueError(f"{path}:{i}: expected must be one of {LABELS}, got {exp!r}")
            clip = (row.get("clip") or "").strip()
            if not clip:
                raise ValueError(f"{path}:{i}: clip is empty")
            out.append(ClipLabel(os.path.join(base, clip), (row.get("scenario") or "").strip(), exp,
                                 (row.get("notes") or "").strip()))
    return out


def predicted_level(severities: Iterable[str]) -> str:
    sev = set(severities)
    if "ALERT" in sev:
        return "alert"
    if "WARNING" in sev:
        return "review"
    return "none"


def _ratio(num: int, den: int) -> Optional[float]:
    return num / den if den else None


def score(results: List[ClipResult]) -> Dict[str, Any]:
    ok = [r for r in results if not r.error]
    pos = [r for r in ok if r.expected == "alert"]
    neg = [r for r in ok if r.expected != "alert"]
    plain = [r for r in ok if r.expected == "none"]
    tp = sum(r.predicted == "alert" for r in pos)
    fp = sum(r.predicted == "alert" for r in neg)
    times = [r.first_alert_s for r in pos if r.predicted == "alert" and r.first_alert_s is not None]
    by_scenario: Dict[str, Dict[str, int]] = {}
    for r in ok:
        s = by_scenario.setdefault(r.scenario or "(unnamed)", {"clips": 0, "correct": 0})
        s["clips"] += 1
        s["correct"] += int(r.correct)
    return {
        "clips": len(results), "errors": len(results) - len(ok),
        "concealment_clips": len(pos), "other_clips": len(neg),
        "tp": tp, "fn": len(pos) - tp, "fp": fp, "tn": len(neg) - fp,
        "recall": _ratio(tp, len(pos)),
        "flag_recall": _ratio(sum(r.predicted != "none" for r in pos), len(pos)),
        "precision": _ratio(tp, tp + fp),
        "false_alarm_rate": _ratio(fp, len(neg)),
        "review_rate": _ratio(sum(r.predicted == "review" for r in plain), len(plain)),
        "accuracy": _ratio(sum(r.correct for r in ok), len(ok)),
        "median_first_alert_s": sorted(times)[len(times) // 2] if times else None,
        "by_scenario": by_scenario,
    }


def run_clip(label: ClipLabel, config: Any, fps: Optional[float] = None) -> ClipResult:
    """Run a fresh pipeline over one clip in video time. Imports cv2/pipeline lazily (heavy)."""
    import cv2
    from pharmaguard.pipeline import PharmaGuardPipeline

    res = ClipResult(os.path.basename(label.clip), label.scenario, label.expected, "none")
    cap = cv2.VideoCapture(label.clip)
    if not cap.isOpened():
        res.error = "cannot open video"
        return res
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cfg = config.model_copy(deep=True)
    cfg.alert.alert_log_file = None       # offline evaluation never writes operational records
    cfg.recording.enabled = False
    pipeline = PharmaGuardPipeline(cfg)
    severities, ids = [], set()
    idx, next_t = -1, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            t = idx / src_fps
            if fps and t + 1e-9 < next_t:
                continue
            next_t = t + (1.0 / fps if fps else 0.0)
            out = pipeline.process_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), timestamp=t)
            ids.update(pipeline.tracker.tracked_registry.keys())
            for a in out.metadata.get("new_alerts", []):
                severities.append(a.severity)
                if a.severity == "ALERT":
                    res.alerts += 1
                    if res.first_alert_s is None:
                        res.first_alert_s = round(t, 2)
                elif a.severity == "WARNING":
                    res.reviews += 1
                res.reasons.extend(a.reasons)
    finally:
        cap.release()
        pipeline.close()
    res.predicted = predicted_level(severities)
    res.ids_issued = len(ids)
    return res


def write_results_csv(results: List[ClipResult], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "scenario", "expected", "predicted", "correct", "alerts", "reviews",
                    "first_alert_s", "ids_issued", "error", "reasons"])
        for r in results:
            w.writerow([r.clip, r.scenario, r.expected, r.predicted, r.correct, r.alerts, r.reviews,
                        "" if r.first_alert_s is None else r.first_alert_s, r.ids_issued, r.error,
                        " | ".join(dict.fromkeys(r.reasons))])
