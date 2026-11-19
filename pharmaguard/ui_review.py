"""
Event review tab for the PharmaGuard dashboard.

Staff browse recorded REVIEW/ALERT events, read the reasons, watch the evidence clip, and record an outcome
(confirmed / false alarm / dismissed). Those outcomes give a MEASURED false-alarm rate for the system.
Formatting helpers are pure functions (unit-tested); ``build_review_tab`` wires them to Gradio.
"""

from html import escape
from typing import Any, Dict, List, Optional, Tuple
import os

import gradio as gr

from pharmaguard.storage import REVIEW_STATUSES, EventRecord, EventStore

TABLE_HEADERS = ["Time", "Severity", "Person", "Zone", "Score", "Review", "Acknowledged by", "Clip"]
STATUS_LABELS = {"open": "Open", "confirmed": "Confirmed", "false_alarm": "False alarm", "dismissed": "Dismissed"}
PLAYABLE = ("saved", "saved_not_browser_playable")


def events_table(store: EventStore, severity: str = "All", status: str = "All",
                 limit: int = 200) -> Tuple[List[List[Any]], List[str]]:
    """Rows for the table and the parallel list of event uids (selection -> uid)."""
    events = store.list_events(limit=limit,
                               severity=None if severity in ("All", "", None) else severity,
                               review_status=None if status in ("All", "", None) else status)
    rows = [[e.created_at, "ALERT" if e.severity == "ALERT" else "REVIEW", f"#{e.tracker_id}", e.zone,
             round(e.score or 0.0, 2), STATUS_LABELS.get(e.review_status, e.review_status),
             e.acknowledged_by or "", e.clip_status] for e in events]
    return rows, [e.event_uid for e in events]


def review_stats(store: EventStore) -> Dict[str, Any]:
    """Counts and the false-alarm rate measured from staff verdicts (None until something is reviewed)."""
    events = store.list_events(limit=100_000)
    out: Dict[str, Any] = {"total": len(events)}
    for sev, label in (("ALERT", "alerts"), ("WARNING", "reviews")):
        subset = [e for e in events if e.severity == sev]
        confirmed = sum(e.review_status == "confirmed" for e in subset)
        false = sum(e.review_status == "false_alarm" for e in subset)
        out[label] = {"count": len(subset), "confirmed": confirmed, "false_alarm": false,
                      "unreviewed": sum(e.review_status == "open" for e in subset),
                      "false_alarm_rate": (false / (confirmed + false)) if (confirmed + false) else None}
    return out


def format_stats_html(stats: Dict[str, Any]) -> str:
    def rate(r: Optional[float]) -> str:
        return "not measured yet (review some events)" if r is None else f"{r:.0%}"
    a, r = stats["alerts"], stats["reviews"]
    return (f"<div style='font-size:13px;line-height:1.6'>"
            f"<b>{stats['total']}</b> recorded events. "
            f"<b>ALERTs:</b> {a['count']} ({a['confirmed']} confirmed, {a['false_alarm']} false alarms, "
            f"{a['unreviewed']} unreviewed) - false-alarm rate <b>{rate(a['false_alarm_rate'])}</b>. "
            f"<b>REVIEWs:</b> {r['count']} ({r['confirmed']} confirmed, {r['false_alarm']} false, "
            f"{r['unreviewed']} unreviewed) - false-alarm rate <b>{rate(r['false_alarm_rate'])}</b>."
            f"<div style='color:#94a3b8;font-size:12px'>Rates count only events staff have marked confirmed or false "
            f"alarm.</div></div>")


def format_event_details(ev: Optional[EventRecord]) -> str:
    if ev is None:
        return "<div style='color:#64748b;font-style:italic'>Select an event in the table.</div>"
    color = "#ef4444" if ev.severity == "ALERT" else "#f59e0b"
    reasons = "".join(f"<li>{escape(r)}</li>" for r in ev.reasons) or "<li>(none recorded)</li>"
    ack = (f"{escape(ev.acknowledged_by)} at {escape(ev.acknowledged_at or '')}" if ev.acknowledged_by
           else "not acknowledged" if ev.severity == "ALERT" else "n/a (visual review only)")
    review = STATUS_LABELS.get(ev.review_status, ev.review_status)
    if ev.reviewed_by:
        review += f" by {escape(ev.reviewed_by)} at {escape(ev.reviewed_at or '')}"
    if ev.review_note:
        review += f" - &ldquo;{escape(ev.review_note)}&rdquo;"
    rows = [("Event ID", ev.event_uid), ("Time", ev.created_at), ("Camera", ev.camera_id),
            ("Person (track ID)", f"#{ev.tracker_id}"), ("Incident", str(ev.incident_id)), ("Zone", ev.zone),
            ("Category", ev.category), ("Suspicion score", f"{ev.score:.2f}"),
            ("Clip", f"{ev.clip_status}" + (f" ({ev.clip_codec})" if ev.clip_codec else ""))]
    table = "".join(f"<tr><td style='color:#94a3b8;padding-right:12px'>{escape(k)}</td><td>{escape(str(v))}</td></tr>"
                    for k, v in rows)
    return (f"<div style='border-left:4px solid {color};padding:6px 12px'>"
            f"<div style='font-weight:700;color:{color};font-size:15px'>{escape(ev.title)}</div>"
            f"<table style='font-size:13px;margin-top:6px'>{table}"
            f"<tr><td style='color:#94a3b8'>Alarm</td><td>{ack}</td></tr>"
            f"<tr><td style='color:#94a3b8'>Staff review</td><td>{review}</td></tr></table>"
            f"<div style='margin-top:6px;font-size:13px'><b>Why it was flagged</b><ul>{reasons}</ul></div>"
            f"<div style='color:#94a3b8;font-size:12px'>PharmaGuard flags interaction sequences for human review; "
            f"it does not determine theft or intent.</div></div>")


def clip_value(ev: Optional[EventRecord]) -> Optional[str]:
    if ev and ev.clip_status in PLAYABLE and ev.clip_path and os.path.exists(ev.clip_path):
        return ev.clip_path
    return None


def build_review_tab(pipeline: Any):
    """Create the Event review tab contents (call inside a ``gr.Tab`` context).

    Returns ``(load_fn, inputs, outputs)`` so the caller can refresh the table on page load / tab open,
    or None when recording is disabled."""
    store: Optional[EventStore] = pipeline.store
    if store is None:
        gr.Markdown("Event recording is disabled (`--no-recording`). Start without it to review events.")
        return None

    uids_state = gr.State([])
    selected_state = gr.State(None)
    stats_html = gr.HTML(format_stats_html(review_stats(store)))
    with gr.Row():
        sev_dd = gr.Dropdown(["All", "ALERT", "WARNING"], value="All", label="Severity (WARNING = REVIEW)", scale=1)
        status_dd = gr.Dropdown(["All", *REVIEW_STATUSES], value="All", label="Review status", scale=1)
        refresh_btn = gr.Button("Refresh", size="sm", scale=0)
    table = gr.Dataframe(headers=TABLE_HEADERS, value=[], interactive=False, wrap=True, max_height=320,
                         label="Recorded events (newest first) - click a row for details")
    with gr.Row():
        with gr.Column(scale=6):
            details = gr.HTML(format_event_details(None))
        with gr.Column(scale=6):
            clip = gr.Video(label="Evidence clip (5 s before, 5 s after)", interactive=False, autoplay=False)
    with gr.Row():
        verdict = gr.Radio([("Confirmed (genuinely suspicious)", "confirmed"),
                            ("False alarm (ordinary behaviour)", "false_alarm"),
                            ("Dismissed (cannot tell / test)", "dismissed"), ("Re-open", "open")],
                           label="Staff verdict", value=None)
        note = gr.Textbox(label="Note (optional)", placeholder="e.g. customer reading a label at the hip")
        save_btn = gr.Button("Save verdict", variant="primary", size="sm")
    feedback = gr.Markdown("")

    def load(sev, status):
        rows, uids = events_table(store, sev, status)
        return rows, uids, format_stats_html(review_stats(store))

    def select(uids, evt: gr.SelectData):
        row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        if row is None or row >= len(uids):
            return None, format_event_details(None), None, None, ""
        ev = store.get_event(uids[row])
        return ev.event_uid, format_event_details(ev), clip_value(ev), ev.review_status, ev.review_note or ""

    def save(uid, status, text, sev, flt, request: gr.Request = None):
        if not uid:
            return "Select an event first.", gr.skip(), gr.skip(), gr.skip(), gr.skip()
        if not status:
            return "Choose a verdict.", gr.skip(), gr.skip(), gr.skip(), gr.skip()
        operator = (getattr(request, "username", None) or "operator") if request is not None else "operator"
        store.set_review(uid, status, operator, text or "")
        pipeline._mirror_async()
        rows, uids = events_table(store, sev, flt)
        return (f"Saved: **{STATUS_LABELS[status]}** by {operator}.", format_event_details(store.get_event(uid)),
                rows, uids, format_stats_html(review_stats(store)))

    for trigger in (refresh_btn.click, sev_dd.change, status_dd.change):
        trigger(load, [sev_dd, status_dd], [table, uids_state, stats_html])
    table.select(select, [uids_state], [selected_state, details, clip, verdict, note])
    save_btn.click(save, [selected_state, verdict, note, sev_dd, status_dd],
                   [feedback, details, table, uids_state, stats_html])
    return load, [sev_dd, status_dd], [table, uids_state, stats_html]
