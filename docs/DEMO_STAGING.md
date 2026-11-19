# PharmaGuard — Demo Staging & Scenario Test Guide

PharmaGuard flags **potentially suspicious interaction sequences for staff review**. It does not determine
theft or intent. Every demo explanation should use that framing.

## 1. Camera placement (most important)

The system judges shelf interaction by **wrist keypoints** (YOLOv8-pose). Wrists must be visible.

```
            MOCK SHELF  (front faces the actor)
        ┌──────────────────────────┐
        │  ▢ ▢ ▢   ▢ ▢ ▢   ▢ ▢ ▢   │
        └──────────────────────────┘
                 ▲
                 │ actor stands here, facing the shelf
                 │
                 ●  actor
                             ◣  laptop webcam: 2-3 m away, ~45° to the side,
                                at chest height, seeing the shelf front AND the actor's reaching arm
```

- **Do not** put the camera directly behind the actor: their body hides the reaching arm (no contact is seen).
- **Do not** put it directly in front of the actor behind the shelf: the shelf hides the hands.
- Keep the actor's **hips visible** in frame. Pocket detection is disabled when hips are not visible (by design: no guessing).
- Use even lighting and avoid strong backlight from windows; pose confidence drops sharply in silhouette.
- Hands must leave the body to count: holding an item at chest height is never a shelf contact.

## 2. Zone setup

1. Start the app and take a screenshot of the annotated output with the empty scene.
2. Measure the shelf front as fractions of the image width/height.
3. Edit `configs/demo_shelf.yaml` → `zones[0].polygon_normalized`, and restart with `--config configs/demo_shelf.yaml`.
4. Check: when the actor's hand reaches into the shelf, the zone badge shows **HAND CONTACT** and the wrist dot turns **orange**.

Wrist dots: green = free, orange = in shelf, magenta = at pocket/bag. Person box: blue = normal, amber = REVIEW, red = ALERT.

## 3. Staged scenarios (mandate §6) and expected results

Record each one as a short clip on the demo laptop (any phone/webcam recorder, 10-30 s). Then run:

```
python tools/track_video.py clip.mp4 --fps 5 --config configs/demo_shelf.yaml --out clip_annotated.mp4
```

The report lists the IDs issued and every REVIEW/ALERT with its reasons. Fill in the "Observed" column.

| # | Scenario | Script for the actor | Expected | Observed |
|---|---|---|---|---|
| 1 | One person enters | Walk in, stand still 5 s | 1 ID, no alerts | |
| 2 | Multiple people | Two actors enter, stand apart | 2 IDs, no alerts | |
| 3 | Walk past shelf | Walk across the frame without reaching | 0 contacts, no alerts | |
| 4 | Browse | Reach in 4-5 times, look at items at chest height, put them back | Contacts, **no REVIEW/ALERT** | |
| 5 | Pick up & return | Take an item, hold it at the hip 2 s, put it back, leave | "returned to shelf", **no alerts** | |
| 6 | Reach own bag | With a handbag on the shoulder, open it without touching the shelf | BAG_REACH info only, **no alerts** | |
| 7a | Item to bag (staged) | Reach in, within 1-2 s put the hand into the handbag, walk out | **REVIEW, then one ALERT** on exit | |
| 7b | Item to pocket (staged) | Reach in, within 1-2 s hand to front pocket and hold, walk out | **REVIEW only** (no alarm) | |
| 8 | People cross | Two actors walk past each other in front of the shelf | IDs unchanged after crossing | |
| 9 | Occlusion | Actor walks behind another actor or a pillar for ~1 s | Same ID afterwards | |
| 10 | Camera interruption | Stop the webcam in the browser for 5 s, restart | Status NO SIGNAL, then LIVE; `1 interruption(s)` | |

**Acceptance:** scenarios 1-6, 8 and 9 produce **zero** REVIEW/ALERT events; 7a produces exactly one ALERT;
7b produces exactly one REVIEW. Record false alerts (alerts in 1-6) and misses (no alert in 7a) in PROJECT_STATUS.md.

## 4. Known false-positive and miss modes (state them honestly if asked)

- **Item held low at the hip, then carried away to a till**: this can produce a REVIEW, never an ALERT. That is intended: staff glance at it.
- **A relaxed hand resting at the hip** is geometrically "at pocket". It only counts within 3 s after the same hand left the shelf.
- **Concealment with the arm hidden from the camera** (behind the body) is missed. Camera placement (section 1) matters.
- **Small items palmed without visible arm movement** are missed: the system tracks hands, not products.
- Bags are COCO "handbag/backpack/suitcase". Unusual bags or bags held low and out of frame may not be detected; then only the weaker pocket path applies.

## 5. Fallback demonstration

If the Colab link fails on the day, play the annotated output of the staged clips produced by
`tools/track_video.py --out ...`. It is the same pipeline on recorded footage with identical results.
Keep these files on the laptop's local disk, not only on Drive.
