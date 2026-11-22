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

1. Start the webcam on **Live monitor**, then open the **Zones** tab and press **Capture snapshot from camera**.
2. Click the corners of the shelf front in order, name the zone, press **Add zone**, then **Apply to live system**.
3. Press **Save to file** (saves into `configs/demo_shelf.yaml`) so it survives a restart (`--config configs/demo_shelf.yaml`).
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
| 5 | Pick up & return | Take an item, hold it at the hip 2 s, put it back (within 4 s of reaching the hip), leave | "returned to shelf", **no alerts** | |
| 6 | Reach own bag | With a handbag on the shoulder, open it without touching the shelf | BAG_REACH info only, **no alerts** | |
| 7a | Item to bag (staged) | Reach in, within 1-2 s put the hand into the handbag, stay or walk out | **REVIEW at once, then one ALERT** 4 s later (or on leaving view) | |
| 7b | Item to pocket (staged) | Reach in, within 1-2 s hand to front pocket and hold ~1 s, stay or walk out | **One ALERT** 4 s after pocketing (or on leaving view) | |
| 8 | People cross | Two actors walk past each other in front of the shelf | IDs unchanged after crossing | |
| 9 | Occlusion | Actor walks behind another actor or a pillar for ~1 s | Same ID afterwards | |
| 10 | Camera interruption | Stop the webcam in the browser for 5 s, restart | Status NO SIGNAL, then LIVE; `1 interruption(s)` | |

**Acceptance:** scenarios 1-6, 8 and 9 produce **zero** REVIEW/ALERT events; 7a produces exactly one ALERT;
7b produces exactly one ALERT. Record false alerts (alerts in 1-6) and misses (no alert in 7a) in PROJECT_STATUS.md.

In the **Event review** tab, mark every recorded event **Confirmed** or **False alarm** after each run: the tab then shows the measured false-alarm rate to quote in the presentation.

## 4. Known false-positive and miss modes (state them honestly if asked)

- **Item held low at the hip for more than 4 s, or carried at hip height to a till**: this now raises an ALERT (rule: shelf -> pocket/bag and not returned within 4 s, TECHNICAL_DECISIONS §13). Hold items at chest height when examining them during the demo.
- **A relaxed hand resting at the hip** is geometrically "at pocket". It only counts within 3 s after the same hand left the shelf.
- **Concealment with the arm hidden from the camera** (behind the body) is missed. Camera placement (section 1) matters.
- **Small items palmed without visible arm movement** are missed: the system tracks hands, not products.
- Bags are COCO "handbag/backpack/suitcase". Unusual bags or bags held low and out of frame may not be detected; then only the weaker pocket path applies.

## 5. Alarm checks before presenting

1. Open the dashboard and **click anywhere once**: the line under the banner must say `Alarm sound: ready`. Browsers block sound until the page is clicked.
2. Press **Test alarm**: a two-tone siren plays for 5 s with a purple banner and a purple flashing border on the video.
3. Set the laptop volume to an audible level and unmute the browser tab.
4. During a staged concealment: an amber REVIEW banner appears at the transfer, then a red ALARM with siren when the actor leaves.
   Press **Acknowledge** (siren stops, banner shows who acknowledged), then **Reset** when resolved.

## 6. Evidence and records

- Each REVIEW/ALERT is stored in `data/pharmaguard.db` (Colab: `/content/pg_data`, copied to `MyDrive/PharmaGuard`) with a
  10 s annotated clip in `clips/` (5 s before, 5 s after the event).
- Staged recordings can also be turned into stored events + clips:
  `python tools/track_video.py clip.mp4 --config configs/demo_shelf.yaml --record-dir staged_events`.
- Records older than 7 days are deleted automatically. Only staged footage of consenting participants may be recorded.

## 7. Fallback demonstration

If the Colab link fails on the day, play the annotated output of the staged clips produced by
`tools/track_video.py --out ...`. It is the same pipeline on recorded footage with identical results.
Keep these files on the laptop's local disk, not only on Drive.
