"""Clip-level scoring and label loading for tools/evaluate_clips.py."""

import pytest

from pharmaguard.evaluation import ClipResult, load_labels, predicted_level, score


def R(expected, predicted, scenario="s", t=None):
    return ClipResult("c.mp4", scenario, expected, predicted, first_alert_s=t)


def test_predicted_level():
    assert predicted_level([]) == "none"
    assert predicted_level(["INFO", "WARNING"]) == "review"
    assert predicted_level(["WARNING", "ALERT"]) == "alert"


def test_score_counts_and_rates():
    results = [R("alert", "alert", "bag", 3.0), R("alert", "alert", "bag", 5.0), R("alert", "review", "pocket"),
               R("alert", "none", "pocket"), R("none", "none", "browse"), R("none", "review", "browse"),
               R("none", "alert", "own bag"), R("review", "review", "slow")]
    s = score(results)
    assert (s["tp"], s["fn"], s["fp"], s["tn"]) == (2, 2, 1, 3)
    assert s["recall"] == 0.5 and s["flag_recall"] == 0.75
    assert s["precision"] == pytest.approx(2 / 3)
    assert s["false_alarm_rate"] == 0.25
    assert s["review_rate"] == pytest.approx(1 / 3)
    assert s["median_first_alert_s"] == 5.0
    assert s["by_scenario"]["bag"] == {"clips": 2, "correct": 2}


def test_errors_excluded_and_empty_rates_are_none():
    bad = ClipResult("x.mp4", "s", "alert", "none", error="cannot open video")
    s = score([bad])
    assert s["errors"] == 1 and s["recall"] is None and s["precision"] is None


def test_load_labels(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("clip,scenario,expected,notes\nc01.mp4,browse,none,\nc02.mp4,bag,ALERT,staged\n")
    labels = load_labels(str(p))
    assert [l.expected for l in labels] == ["none", "alert"]
    assert labels[0].clip == str(tmp_path / "c01.mp4")
    p.write_text("clip,scenario,expected\nc.mp4,x,maybe\n")
    with pytest.raises(ValueError):
        load_labels(str(p))


def test_per_clip_config_column(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("clip,scenario,expected,notes,config\na.mp4,s,none,,view.yaml\nb.mp4,s,alert,,\n")
    a, b = load_labels(str(p))
    assert a.config == str(tmp_path / "view.yaml") and b.config == ""


def test_make_labels_tool(tmp_path):
    import subprocess, sys, os
    for folder, n in (("Shoplifting", 2), ("Normal", 3), ("other", 1)):
        d = tmp_path / folder / "sub"
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"v{i}.mp4").write_bytes(b"")
    tool = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools", "make_labels.py")
    out = subprocess.run([sys.executable, tool, str(tmp_path), "--map", "shoplifting=alert", "normal=none"],
                         capture_output=True, text=True, check=True).stdout
    assert "Skipped folders" in out and "other" in out
    labels = load_labels(str(tmp_path / "labels.csv"))
    assert sorted(l.expected for l in labels) == ["alert", "alert", "none", "none", "none"]
