"""
Zone editor for the PharmaGuard dashboard.

Operators click polygon corners on a snapshot of the live camera to outline shelf zones. Edits are made on a
working copy; nothing changes on the running system until **Apply**. **Save** writes the zones into a YAML
config (other settings in that file are preserved) so they survive restarts.

The geometry and validation helpers are pure functions (unit-tested); ``build_zones_tab`` wires them to Gradio.
"""

from typing import Any, Dict, List, Optional, Tuple
import os

import cv2
import gradio as gr
import numpy as np
import yaml

from pharmaguard.config import ShelfZone

DraftT = Dict[str, Any]          # {"points": [[x, y], ...] in pixels, "w": int, "h": int}
ZoneDicts = List[Dict[str, Any]]  # ShelfZone.model_dump() list (JSON-friendly for gr.State)

MIN_POINTS = 3
CONFIGS_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs"))


def resolve_config_path(path: str, base_dir: Optional[str] = None) -> str:
    """Only allow saving *.yaml files inside ``configs/``. The dashboard may be reachable via a public share
    link, so a browser-supplied path must never be able to overwrite arbitrary files on the server."""
    base_dir = base_dir or CONFIGS_DIR
    name = os.path.basename((path or "").strip().replace("\\", "/"))
    if not name.endswith((".yaml", ".yml")) or name in ("", ".yaml", ".yml"):
        raise ValueError("file name must end in .yaml")
    target = os.path.abspath(os.path.join(base_dir, name))
    if os.path.dirname(target) != os.path.abspath(base_dir):
        raise ValueError("zones can only be saved inside configs/")
    return target


def placeholder_frame(w: int = 640, h: int = 480) -> np.ndarray:
    img = np.full((h, w, 3), 40, np.uint8)
    for i, line in enumerate(["No camera frame yet.", "Start the webcam on the Live monitor tab,",
                              "then press 'Capture snapshot'."]):
        cv2.putText(img, line, (30, 200 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    return img


def new_draft(frame: np.ndarray) -> DraftT:
    h, w = frame.shape[:2]
    return {"points": [], "w": int(w), "h": int(h)}


def add_point(draft: DraftT, x: float, y: float) -> DraftT:
    d = {**draft, "points": list(draft["points"])}
    d["points"].append([float(np.clip(x, 0, d["w"] - 1)), float(np.clip(y, 0, d["h"] - 1))])
    return d


def undo_point(draft: DraftT) -> DraftT:
    return {**draft, "points": list(draft["points"])[:-1]}


def draft_to_zone(draft: DraftT, name: str, prolonged_presence_sec: float,
                  existing: ZoneDicts) -> Tuple[Optional[ShelfZone], str]:
    """Validate the draft and convert to a normalised ShelfZone. Returns (zone, error message)."""
    name = (name or "").strip()
    if not name:
        return None, "Give the zone a name."
    if any(z["name"] == name for z in existing):
        return None, f"A zone named '{name}' already exists - delete it first or choose another name."
    pts = draft["points"]
    if len(pts) < MIN_POINTS:
        return None, f"A zone needs at least {MIN_POINTS} points (have {len(pts)})."
    poly = np.array(pts, np.float32)
    area = 0.5 * abs(np.dot(poly[:, 0], np.roll(poly[:, 1], 1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], 1)))
    if area < 0.002 * draft["w"] * draft["h"]:
        return None, "Zone is too small or the points are in a line - click wider corners."
    norm = [[round(float(x) / draft["w"], 4), round(float(y) / draft["h"], 4)] for x, y in pts]
    return ShelfZone(name=name, polygon_normalized=norm, prolonged_presence_sec=prolonged_presence_sec), ""


def render_preview(frame: np.ndarray, zones: ZoneDicts, draft: Optional[DraftT]) -> np.ndarray:
    """Existing zones in cyan, the polygon being drawn in yellow with numbered corners."""
    img = frame.copy()
    h, w = img.shape[:2]
    overlay = img.copy()
    for z in zones:
        pts = np.array([[p[0] * w, p[1] * h] for p in z["polygon_normalized"]], np.int32)
        cv2.fillPoly(overlay, [pts], (0, 160, 200))
        cv2.polylines(img, [pts], True, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(img, z["name"], tuple(int(v) for v in pts[0] + [4, 18]), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 220, 255), 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
    if draft and draft["points"]:
        pts = np.array(draft["points"], np.int32)
        if len(pts) >= 2:
            cv2.polylines(img, [pts], len(pts) >= MIN_POINTS, (255, 220, 0), 2, cv2.LINE_AA)
        for i, (x, y) in enumerate(pts):
            cv2.circle(img, (int(x), int(y)), 6, (255, 220, 0), -1, cv2.LINE_AA)
            cv2.putText(img, str(i + 1), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                        1, cv2.LINE_AA)
    return img


def save_zones_yaml(path: str, zones: ZoneDicts, base_dir: Optional[str] = None) -> str:
    """Write ``zones`` into configs/<name>.yaml, keeping every other setting already in that file."""
    path = resolve_config_path(path, base_dir)
    data: Dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    data["zones"] = [ShelfZone(**z).model_dump() for z in zones]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path


def zones_summary(zones: ZoneDicts, live_names: List[str]) -> str:
    if not zones:
        return "_No zones in the working copy. Draw one and press **Add zone**._"
    rows = ["| Zone | Corners | Presence (s) | Live now |", "|---|---|---|---|"]
    for z in zones:
        rows.append(f"| {z['name']} | {len(z['polygon_normalized'])} | {z['prolonged_presence_sec']:.0f} | "
                    f"{'yes' if z['name'] in live_names else 'no (press Apply)'} |")
    return "\n".join(rows)


def build_zones_tab(pipeline: Any, config_path: Optional[str]) -> None:
    """Create the Zones tab contents (call inside a ``gr.Tab`` context)."""
    default_path = os.path.basename(config_path) if config_path else "demo_shelf.yaml"

    def live_dicts() -> ZoneDicts:
        return [z.model_dump() for z in pipeline.config.zones]

    def live_names() -> List[str]:
        return [z.name for z in pipeline.config.zones]

    initial_zones = live_dicts()
    initial_frame = placeholder_frame()

    gr.Markdown(
        "**Outline the front of each shelf** where hands reach in. 1) Capture a snapshot. 2) Click the corners "
        "in order. 3) Name it and press **Add zone**. 4) **Apply to live system**. 5) **Save** to keep it after a "
        "restart. Hands held in front of a person's own body never count as shelf contact, so the zone may "
        "overlap where people stand.")
    frame_state = gr.State(initial_frame)
    draft_state = gr.State(new_draft(initial_frame))
    zones_state = gr.State(initial_zones)
    with gr.Row():
        with gr.Column(scale=7):
            canvas = gr.Image(value=render_preview(initial_frame, initial_zones, None), type="numpy",
                              interactive=False, label="Click to add zone corners", elem_id="pg-zone-canvas")
        with gr.Column(scale=5):
            snap_btn = gr.Button("Capture snapshot from camera", variant="primary", size="sm")
            with gr.Row():
                undo_btn = gr.Button("Undo point", size="sm")
                clear_btn = gr.Button("Clear drawing", size="sm")
            name_box = gr.Textbox(label="Zone name", placeholder="e.g. Shelf_Painkillers")
            presence = gr.Number(value=20, label="Prolonged presence (s) - weak evidence only", precision=0)
            add_btn = gr.Button("Add zone", size="sm")
            delete_dd = gr.Dropdown(choices=[z["name"] for z in initial_zones], label="Delete zone", value=None)
            delete_btn = gr.Button("Delete selected zone", size="sm", variant="stop")
            apply_btn = gr.Button("Apply to live system", variant="primary")
            save_path = gr.Textbox(value=default_path, label="Save zones into configs/<file>.yaml")
            save_btn = gr.Button("Save to file", size="sm")
            status = gr.Markdown("")
            summary = gr.Markdown(zones_summary(initial_zones, live_names()))

    def redraw(frame, zones, draft):
        return render_preview(frame, zones, draft)

    def on_snapshot(zones):
        raw = pipeline.last_raw_frame
        frame = raw.copy() if raw is not None else placeholder_frame()
        msg = "Snapshot captured." if raw is not None else "No live frame yet - start the webcam first."
        draft = new_draft(frame)
        return frame, draft, redraw(frame, zones, draft), msg

    def on_click(frame, zones, draft, evt: gr.SelectData):
        x, y = evt.index[0], evt.index[1]
        if draft["w"] != frame.shape[1] or draft["h"] != frame.shape[0]:
            draft = new_draft(frame)
        draft = add_point(draft, x, y)
        return draft, redraw(frame, zones, draft), f"{len(draft['points'])} point(s)"

    def on_undo(frame, zones, draft):
        draft = undo_point(draft)
        return draft, redraw(frame, zones, draft)

    def on_clear(frame, zones):
        draft = new_draft(frame)
        return draft, redraw(frame, zones, draft)

    def on_add(frame, zones, draft, name, presence_s):
        zone, err = draft_to_zone(draft, name, float(presence_s or 20), zones)
        if zone is None:
            return zones, draft, redraw(frame, zones, draft), f"⚠️ {err}", gr.skip(), gr.skip()
        zones = zones + [zone.model_dump()]
        draft = new_draft(frame)
        return (zones, draft, redraw(frame, zones, draft), f"Added **{zone.name}** to the working copy. "
                "Press **Apply to live system** to start monitoring it.",
                gr.update(choices=[z["name"] for z in zones], value=None), zones_summary(zones, live_names()))

    def on_delete(frame, zones, draft, name):
        if not name:
            return zones, redraw(frame, zones, draft), "Choose a zone to delete.", gr.skip(), gr.skip()
        zones = [z for z in zones if z["name"] != name]
        return (zones, redraw(frame, zones, draft), f"Removed **{name}** from the working copy (press Apply).",
                gr.update(choices=[z["name"] for z in zones], value=None), zones_summary(zones, live_names()))

    def on_apply(zones):
        if not zones:
            return "⚠️ Refusing to apply an empty zone list: monitoring would silently stop.", gr.skip()
        pipeline.set_zones([ShelfZone(**z) for z in zones])
        return f"✅ Live system now monitors: {', '.join(z['name'] for z in zones)}", zones_summary(zones, live_names())

    def on_save(zones, path):
        if not zones:
            return "⚠️ Nothing to save."
        try:
            target = save_zones_yaml(path, zones)
            return (f"💾 Saved {len(zones)} zone(s) to `configs/{os.path.basename(target)}`. "
                    f"Start with `--config configs/{os.path.basename(target)}`.")
        except (OSError, yaml.YAMLError, ValueError) as exc:
            return f"⚠️ Save failed: {exc}"

    snap_btn.click(on_snapshot, [zones_state], [frame_state, draft_state, canvas, status])
    canvas.select(on_click, [frame_state, zones_state, draft_state], [draft_state, canvas, status])
    undo_btn.click(on_undo, [frame_state, zones_state, draft_state], [draft_state, canvas])
    clear_btn.click(on_clear, [frame_state, zones_state], [draft_state, canvas])
    add_btn.click(on_add, [frame_state, zones_state, draft_state, name_box, presence],
                  [zones_state, draft_state, canvas, status, delete_dd, summary])
    delete_btn.click(on_delete, [frame_state, zones_state, draft_state, delete_dd],
                     [zones_state, canvas, status, delete_dd, summary])
    apply_btn.click(on_apply, [zones_state], [status, summary])
    save_btn.click(on_save, [zones_state, save_path], [status])
