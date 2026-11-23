# PharmaGuard: Evaluation Plan (staged clips at the mock shelf)

**Purpose:** measure recall, precision and false-alarm rate on real video, not scripted skeletons.
No pharmacy is needed. Every clip is recorded at a mock shelf with consenting actors.
Camera placement, zone drawing and the 10 functional scenarios are in `docs/DEMO_STAGING.md`. This plan
adds a repeatable, scored test set for the report.

## 0. Option without filming: public staged datasets

Download on your own laptop (these sites need a browser login or click-through; cite them in the report):

| Dataset | What it is | Use it as |
|---|---|---|
| Shoplifting Dataset (2022), CV Lab MNNIT Allahabad, Mendeley Data, DOI 10.17632/r3yjf35hzr.1 | Staged in a lab, 640x480, 30 FPS, classes `normal` / `shoplifting` (items into bags or under clothing) | Closest to the mock-shelf demo. **Start here** |
| Shoplifting Dataset, Muneer & Saddique, Mendeley Data, DOI 10.17632/3pn2rnv2wg.1 | ~900 staged clips (450 / 450), single camera angle | Larger test set |
| UCF-Crime (shoplifting + normal classes) | Real, low-resolution overhead CCTV | Hard set only: wrists and hips are often not visible, so expect misses. Report separately |

**Easiest: run `colab/PharmaGuard_Evaluate.ipynb` in Colab.** It keeps the zip and results in Google Drive
(`MyDrive/PharmaGuard/`), helps you place the zone on a sample frame, and runs the steps below.

Steps (local alternative):
1. Unzip into `clips/<dataset>/` keeping the class folders.
2. Open one clip, take a screenshot, and draw the shelf zone in the **Zones** tab from that screenshot (or edit
   `configs/demo_shelf.yaml` coordinates). Save it as e.g. `configs/mnnit_view.yaml`. **Every camera view needs its own
   zone**; without the right zone no shelf contact is ever seen and every clip scores `none`. If a dataset has several
   camera angles, make one config per angle and split the clips by angle.
3. `python tools/make_labels.py clips/mnnit --map shoplifting=alert normal=none --config ../../configs/mnnit_view.yaml`
4. `python tools/evaluate_clips.py clips/mnnit/labels.csv --fps 5 --out results_mnnit.csv`

Be honest about the gap: these clips are not a pharmacy and not your camera placement. Report them as
"public staged retail datasets" and keep the mock-shelf set (below) for the pharmacy-specific claim if you can film it.

## 1. Setup (one afternoon)

| Item | Requirement |
|---|---|
| Location | HCI lab or a classroom. Same room for every clip; note lighting (daylight / fluorescent) |
| Shelf | Bookshelf or table with 2 levels. Products: empty medicine boxes, a few OTC packs, sachets, cosmetics |
| Counter | A desk 1-2 m from the shelf (for "pay at counter" clips) |
| Camera | The demo laptop webcam or a phone, ~45° to the side, chest to head height, 2-3 m away (DEMO_STAGING §1). Fixed for all clips |
| Recording | 720p or 640x480, 30 FPS, 10-30 s per clip. One action per clip. Start recording before the actor enters |
| Actors | 3-4 people. Vary clothing: hoodie, shirt with front pockets, trousers with side pockets, with/without handbag or backpack |
| Consent | Signed consent form per actor before filming (template in section 6). Ask your supervisor whether departmental ethics approval is needed |
| Zones | Draw the shelf zone once from this camera view and save `configs/eval_shelf.yaml`. Use it for every clip |

## 2. Clip set (36 clips)

Each actor performs each row the number of times shown, so the set is not one person's habits.
File names: `c01.mp4` ... `c36.mp4`. `expected` is what the system SHOULD do (rules in TECHNICAL_DECISIONS §7, §13, §20).

| # | Scenario (`scenario` column) | Script | Clips | `expected` | Rule it tests |
|---|---|---|---|---|---|
| A | `walk_past` | Walk across the frame, no reach | 3 | none | No shelf contact |
| B | `browse_chest` | Reach in 3-4 times, inspect items at chest height, put back | 4 | none | Chest-height handling is not a transfer |
| C | `pick_return_hip` | Take item, hold at hip < 4 s, put back, leave | 3 | none | Return cancels pocket evidence |
| D | `own_bag` | Open own handbag without touching the shelf | 3 | none | Bag reach without prior shelf contact |
| E | `phone_pocket` | Take own phone out of pocket, use it, put it back. No shelf contact | 3 | none | Pocket without prior shelf contact |
| F | `pay_counter` | Take item, carry it at chest height to the counter, "pay", leave | 3 | none | Normal purchase |
| G | `slow_transfer` | Take item, examine it 5+ s at chest, then put it in own bag | 2 | none | Slow transfer after examining is ordinary handling (§7: transfer must follow within 3 s). If you disagree with this rule, label these `review` and report it |
| H | `conceal_bag` | Reach in, within 1-2 s put item in handbag/backpack, walk out | 6 | alert | Shelf → bag → leave |
| I | `conceal_pocket` | Reach in, within 1-2 s put item in a front or side pocket (hold ~1 s, elbow bent), walk out | 6 | alert | Shelf → pocket → leave (§20 bent elbow) |
| J | `two_people_one_conceals` | Two actors at the shelf; only one conceals | 3 | alert | Only the concealer is flagged; check `ids_issued` = 2 |

Totals: 15 concealment clips (`alert`), 21 ordinary clips (`none`). Use `review` only for clips where a REVIEW is acceptable but an ALERT would be wrong.

**Hard cases to record as extra clips (label honestly, report separately):** concealment with the arm turned away
from the camera; item palmed into a sleeve; polythene bag (not a COCO class). These are known blind spots
(`docs/LIMITATIONS.md`) and should be reported as such, not hidden.

## 3. labels.csv

Put the clips and `labels.csv` in one folder (never commit them: they show people's faces):

```
clip,scenario,expected,notes
c01.mp4,walk_past,none,actor 1 hoodie
c02.mp4,browse_chest,none,actor 2
c13.mp4,conceal_bag,alert,actor 1 backpack
c20.mp4,conceal_pocket,alert,actor 3 side pocket
```

`expected` must be `none`, `review` or `alert`. Clip paths are relative to the CSV.

## 4. Run and score

```
python tools/evaluate_clips.py clips/labels.csv --config configs/eval_shelf.yaml --fps 5 --out results.csv
```

`--fps 5` processes the video the way the live tunnel delivers it (~5 FPS). Run once more at `--fps 10` to show how
sensitive the results are to frame rate. Optional: `python tools/track_video.py clip.mp4 --config ... --out annotated.mp4`
for any clip that came out WRONG, to see why.

The tool prints, all measured on your clips:

| Metric | Meaning | Report it as |
|---|---|---|
| Recall | Concealment clips that raised an ALERT | "x of 15 staged concealments raised an alert" |
| Flag recall | Concealment clips that raised REVIEW or ALERT | Staff were at least prompted to look |
| Precision | ALERT clips that were really concealment | How often an alarm is worth responding to |
| False-alarm rate | Non-concealment clips that raised an ALERT | Wrongly sounded alarms per ordinary visit |
| REVIEW rate | Ordinary clips that raised only a REVIEW | Operator workload |
| Median time to first ALERT | Seconds into the clip | Responsiveness |
| Per scenario | Correct / total | Which behaviours fail |

## 5. Rules for honest results

- Record all clips BEFORE tuning. Then split: tune thresholds on half the actors, report the final numbers on the
  other half only. Tuning on the clips you report inflates the results.
- Never delete a clip because it came out WRONG. Report it, and say which limitation explains it.
- State the conditions next to every number: one room, one camera angle, N actors, staged behaviour. A clip-level
  result on 36 staged clips is a prototype evaluation, not evidence of real-world accuracy.
- Paste the summary block and `results.csv` into the report appendix.

## 6. Consent form (minimum content)

> I agree to be filmed acting out shopping and staged concealment scenarios at a mock shelf for the PharmaGuard
> final-year project (BSc Information Technology, University of Ghana). The footage is used only to test the system,
> stored on the project laptop, not published, and deleted after the project is graded. I can withdraw at any time
> and my clips will be deleted.
> Name: ______  Signature: ______  Date: ______
