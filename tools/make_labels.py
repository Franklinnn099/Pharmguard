"""
Build labels.csv for a downloaded dataset whose clips are sorted into class folders.

    python tools/make_labels.py datasets/mnnit --map shoplifting=alert normal=none --config ../../configs/mnnit_view.yaml

Every video under <root>/<class folder>/ (any depth) gets one row. Folder names are matched case-insensitively.
Folders not in --map are skipped and listed, so nothing is labelled by guesswork.
"""

import argparse
import csv
import os

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".webm")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--map", nargs="+", required=True, help="folder=label pairs, label in none|review|alert")
    ap.add_argument("--config", default="", help="Zone YAML for this dataset's camera view (relative to root)")
    ap.add_argument("--out", default=None, help="Default: <root>/labels.csv")
    args = ap.parse_args()

    mapping = {}
    for pair in args.map:
        folder, _, label = pair.partition("=")
        if label not in ("none", "review", "alert"):
            raise SystemExit(f"Bad label in {pair!r}: use none, review or alert")
        mapping[folder.lower()] = label
    rows, skipped = [], set()
    for entry in sorted(os.listdir(args.root)):
        path = os.path.join(args.root, entry)
        if not os.path.isdir(path):
            continue
        label = mapping.get(entry.lower())
        if label is None:
            skipped.add(entry)
            continue
        for dirpath, _, files in os.walk(path):
            for f in sorted(files):
                if f.lower().endswith(VIDEO_EXT):
                    rel = os.path.relpath(os.path.join(dirpath, f), args.root)
                    rows.append([rel.replace(os.sep, "/"), entry, label, "", args.config])
    out = args.out or os.path.join(args.root, "labels.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "scenario", "expected", "notes", "config"])
        w.writerows(rows)
    counts = {}
    for r in rows:
        counts[r[2]] = counts.get(r[2], 0) + 1
    print(f"Wrote {len(rows)} rows to {out}: {counts}")
    if skipped:
        print(f"Skipped folders (not in --map): {sorted(skipped)}")


if __name__ == "__main__":
    main()
