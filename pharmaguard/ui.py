"""
Gradio web interface module for PharmaGuard.

Real-time design (see TECHNICAL_DECISIONS.md §3):
* The webcam stream event carries ONLY the annotated frame. Text/HTML panels are refreshed by a
  1 Hz ``gr.Timer`` so per-frame websocket payloads stay small.
* The browser is asked for a 640x480 capture (Gradio otherwise requests 1920x1440) and the
  annotated frame is returned as JPEG (Gradio's default WebP encode measured ~0.5 s per 720p frame).
* Gradio's webcam client does not capture while a frame is pending, so frames never queue up;
  the server-side arrival interval is therefore the true end-to-end frame period.
"""

from html import escape
from typing import Any, Dict, Optional
import gradio as gr
import numpy as np

from pharmaguard.pipeline import PharmaGuardPipeline, PipelineOutput
from pharmaguard.config import PharmaGuardConfig
from pharmaguard.telemetry import TelemetrySnapshot
from pharmaguard.alarm import AlarmSnapshot
from pharmaguard.ui_review import build_review_tab
from pharmaguard.ui_zones import build_zones_tab
from pharmaguard.fast_stream import FAST_STREAM_HEAD, FAST_STREAM_HTML


# Custom styling for professional pharmacy security console
SECURITY_THEME_CSS = """
/* Dark surveillance operations aesthetic */
body, .gradio-container {
    background-color: #0b0f17 !important;
    color: #e2e8f0 !important;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif !important;
}

.header-banner {
    background: linear-gradient(90deg, #111827 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

.header-title {
    font-size: 24px;
    font-weight: 700;
    color: #38bdf8;
    margin: 0;
    letter-spacing: 0.5px;
}

.header-subtitle {
    font-size: 13px;
    color: #94a3b8;
    margin-top: 4px;
}

.metric-card {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 12px;
    text-align: center;
}

.alert-box {
    background-color: #181c24;
    border: 1px solid #451a1a;
    border-radius: 6px;
    padding: 10px;
    max-height: 240px;
    overflow-y: auto;
}

.event-box {
    background-color: #181c24;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 10px;
    max-height: 200px;
    overflow-y: auto;
}
"""

# Browser-side measurement of how many annotated frames are actually painted per second.
# This is the number the operator experiences; it complements the server-side cycle time.
CLIENT_FPS_HEAD = """
<script>
(() => {
  let painted = 0, lastSrc = null;
  const tick = () => {
    const img = document.querySelector('#pg-output img');
    if (img && img.src && img.src !== lastSrc) { lastSrc = img.src; painted++; }
  };
  const loop = () => { tick(); requestAnimationFrame(loop); };
  requestAnimationFrame(loop);
  let fastSeen = 0;
  setInterval(() => {
    const el = document.getElementById('pg-client-fps');
    const fastTotal = window.__pgFastPainted || 0;   // frames painted by the fast stream canvas
    if (el) el.textContent = (painted + fastTotal - fastSeen).toFixed(0);
    fastSeen = fastTotal;
    painted = 0;
  }, 1000);
})();
</script>
"""

# Browser alarm siren (Web Audio API, no audio file). The server renders alarm state into
# <div id="pg-alarm" data-sounding data-seq data-kind data-volume>; this script polls it every 250 ms.
# Browsers block audio until the user interacts with the page, so the status line says so explicitly.
ALARM_HEAD = """
<script>
(() => {
  let ctx = null, osc = null, gain = null, timer = null, lastSeq = -1, playing = false;
  const ensureCtx = () => {
    if (!ctx) { const AC = window.AudioContext || window.webkitAudioContext; if (AC) ctx = new AC(); }
    return ctx;
  };
  const unlock = () => { const c = ensureCtx(); if (c && c.state === 'suspended') c.resume(); };
  ['pointerdown', 'keydown', 'touchstart'].forEach(ev => document.addEventListener(ev, unlock, {capture: true}));
  const status = () => {
    const el = document.getElementById('pg-sound-status');
    if (!el) return;
    const ok = ctx && ctx.state === 'running';
    el.textContent = ok ? 'Alarm sound: ready' : 'Alarm sound: LOCKED by the browser - click anywhere on this page once to enable';
    el.style.color = ok ? '#34d399' : '#f87171';
  };
  const stop = () => {
    if (timer) clearInterval(timer);
    timer = null;
    try { if (osc) osc.stop(); } catch (e) {}
    osc = null;
    if (gain) gain.disconnect();
    gain = null;
    playing = false;
  };
  const start = (kind, vol) => {
    const c = ensureCtx(); if (!c) return;
    stop();
    gain = c.createGain(); gain.gain.value = vol; gain.connect(c.destination);
    osc = c.createOscillator(); osc.type = 'square'; osc.connect(gain); osc.start();
    const hiF = kind === 'TEST' ? 660 : 960, loF = kind === 'TEST' ? 520 : 640;
    let hi = true; osc.frequency.value = hiF;
    timer = setInterval(() => { hi = !hi; if (osc) osc.frequency.setValueAtTime(hi ? hiF : loF, c.currentTime); },
                        kind === 'TEST' ? 300 : 400);
    playing = true;
  };
  ensureCtx();
  setInterval(() => {
    status();
    const el = document.getElementById('pg-alarm');
    if (!el) return;
    const sounding = el.dataset.sounding === '1';
    const seq = parseInt(el.dataset.seq || '0', 10);
    if (sounding && (seq !== lastSeq || !playing)) { start(el.dataset.kind || 'ALERT', parseFloat(el.dataset.volume || '0.25')); lastSeq = seq; }
    else if (!sounding && playing) { stop(); }
    window.__pgAlarm = {playing, seq: lastSeq, ctxState: ctx ? ctx.state : 'unsupported'};  // automated checks
  }, 250);
})();
</script>
"""

DASHBOARD_HEAD = CLIENT_FPS_HEAD + ALARM_HEAD + FAST_STREAM_HEAD

SIGNAL_LABELS = {
    "LIVE": "LIVE",
    "NO_SIGNAL": "NO SIGNAL - camera stopped or connection lost",
    "STANDBY": "STANDBY - start the webcam on the left",
}


def format_alerts_html(alerts: list) -> str:
    """Format alerts as HTML cards with severity color chips."""
    if not alerts:
        return "<div style='color: #64748b; font-style: italic; padding: 8px;'>No active security alerts. Monitored zones secure.</div>"

    html = "<div style='display: flex; flex-direction: column; gap: 8px;'>"
    for a in alerts:
        color = "#ef4444" if a.severity == "ALERT" else "#f59e0b"
        bg_color = "rgba(239, 68, 68, 0.12)" if a.severity == "ALERT" else "rgba(245, 158, 11, 0.12)"
        html += f"""
        <div style='background: {bg_color}; border-left: 4px solid {color}; padding: 8px 12px; border-radius: 4px;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <strong style='color: {color}; font-size: 13px;'>[{escape(a.severity)}] {escape(a.title)}</strong>
                <span style='font-size: 11px; color: #94a3b8;'>{escape(a.timestamp_iso[-8:])}</span>
            </div>
            <div style='color: #cbd5e1; font-size: 12px; margin-top: 4px;'>{escape(a.message)}</div>
        </div>
        """
    html += "</div>"
    return html


def format_events_html(events: list) -> str:
    """Format event timeline items as a monospace log."""
    if not events:
        return "<div style='color: #64748b; font-style: italic; padding: 8px;'>System standby. Awaiting video stream...</div>"

    html = "<div style='font-family: monospace; font-size: 11.5px; display: flex; flex-direction: column; gap: 4px;'>"
    for e in events:
        cat_color = "#38bdf8" if e.category == "SECURITY" else "#a3e635" if e.category == "TRACKING" else "#94a3b8"
        html += f"""
        <div style='border-bottom: 1px solid #1e293b; padding-bottom: 2px;'>
            <span style='color: #64748b;'>{escape(e.timestamp_iso)}</span>
            <span style='color: {cat_color}; font-weight: 600; margin-left: 6px;'>[{escape(e.category)}]</span>
            <span style='color: #e2e8f0; margin-left: 6px;'>{escape(e.message)}</span>
        </div>
        """
    html += "</div>"
    return html


def format_telemetry(tele: TelemetrySnapshot, last: Optional[PipelineOutput], device: str) -> Dict[str, str]:
    """Human-readable telemetry strings. Every figure here is measured, none are targets."""
    live = tele.signal == "LIVE" and last is not None
    timings = last.metadata.get("stage_timings", {}) if last else {}
    res = last.metadata.get("resolution", "-") if last else "-"
    in_res = last.metadata.get("input_resolution", "-") if last else "-"
    status = f"{SIGNAL_LABELS.get(tele.signal, tele.signal)} | {device.upper()}"
    # Fast stream (pipelined POSTs): the browser reports its own round trip, which is the honest
    # network figure. The arrival-gap "outside" share is only meaningful with one frame in flight.
    if tele.client_rtt_p50_ms > 0:
        network = f"browser round trip p50 {tele.client_rtt_p50_ms:.0f} / p95 {tele.client_rtt_p95_ms:.0f} ms"
    else:
        network = f"transport+encode {tele.outside_p50_ms:.0f} ms"
    if tele.interruptions:
        status += f" | {tele.interruptions} interruption(s)"
    return {
        "status": status,
        "tracked": str(last.active_tracks_count) if live else "0",
        "ids": (", ".join(f"#{t}" for t in last.active_ids) or "None") if live else "None",
        "fps": f"{tele.delivered_fps:.1f}" if live else "0.0",
        "latency": (
            f"cycle p50 {tele.cycle_p50_ms:.0f} / p95 {tele.cycle_p95_ms:.0f} ms | "
            f"server {tele.processing_p50_ms:.0f} ms (det {timings.get('detect_ms', 0):.0f}) | "
            f"{network} | {in_res}->{res}"
        ) if live else "-",
    }


def format_alarm_html(snap: AlarmSnapshot, volume: float = 0.25) -> str:
    """Alarm banner. The data-* attributes drive the browser siren (ALARM_HEAD)."""
    attrs = (f"id='pg-alarm' data-sounding='{1 if snap.sounding else 0}' data-seq='{snap.sound_seq}' "
             f"data-kind='{escape(snap.kind)}' data-state='{escape(snap.state)}' data-volume='{volume:.2f}'")

    def box(bg: str, border: str, inner: str) -> str:
        return (f"<div {attrs} style='background:{bg};border:2px solid {border};border-radius:8px;"
                f"padding:10px 14px;margin:4px 0;'>{inner}</div>")

    def alert_lines(alerts: list) -> str:
        rows = []
        for a in alerts[-3:]:
            reasons = "".join(f"<li>{escape(r)}</li>" for r in a.reasons[:4])
            rows.append(
                f"<div style='margin-top:6px;color:#fee2e2;font-size:13px;'><b>Person #{a.tracker_id}</b> at "
                f"{escape(a.zone_name)} - {escape(a.timestamp_iso)} - score {a.score:.2f} - incident {a.incident_id}"
                f"<ul style='margin:2px 0 0 18px;font-size:12px;color:#fecaca;'>{reasons}</ul></div>")
        return "".join(rows)

    if snap.state == "ACTIVE":
        silenced = (" <span style='font-size:12px;color:#fca5a5;'>(sound auto-stopped after timeout - still "
                    "unacknowledged)</span>") if snap.silenced_reason == "timeout" else ""
        return box("rgba(220,38,38,0.25)", "#ef4444",
                   f"<div style='font-size:18px;font-weight:800;color:#fecaca;'>ALARM - suspicious interaction "
                   f"sequence, review now ({len(snap.active_alerts)} active){silenced}</div>"
                   f"<div style='font-size:12px;color:#fca5a5;'>Raised {escape(snap.raised_at_iso)}. "
                   f"Press <b>Acknowledge</b> to silence, <b>Reset</b> when resolved. "
                   f"This flags behaviour for review; it does not establish theft.</div>"
                   + alert_lines(snap.active_alerts))
    if snap.state == "ACKNOWLEDGED":
        return box("rgba(127,29,29,0.25)", "#991b1b",
                   f"<div style='font-size:15px;font-weight:700;color:#fecaca;'>Alarm acknowledged by "
                   f"{escape(snap.acknowledged_by)} at {escape(snap.acknowledged_at_iso)} - press Reset when "
                   f"resolved</div>" + alert_lines(snap.active_alerts))
    if snap.state == "TEST":
        return box("rgba(168,85,247,0.2)", "#a855f7",
                   "<div style='font-size:16px;font-weight:700;color:#e9d5ff;'>ALARM TEST - checking sound and "
                   "display (not a security event)</div>")
    if snap.latest_review is not None:
        r = snap.latest_review
        return box("rgba(245,158,11,0.15)", "#f59e0b",
                   f"<div style='font-size:14px;font-weight:700;color:#fde68a;'>Review suggested: person "
                   f"#{r.tracker_id} at {escape(r.zone_name)} ({escape(r.timestamp_iso[-8:])}, score "
                   f"{r.score:.2f}) - visual only, no alarm</div>"
                   f"<div style='font-size:12px;color:#fcd34d;'>{escape('; '.join(r.reasons[:3]))}</div>")
    return box("rgba(16,185,129,0.08)", "#065f46",
               "<div style='font-size:14px;color:#6ee7b7;'>Monitoring - no active alarm</div>")


def create_pharmaguard_ui(
    pipeline: PharmaGuardPipeline,
    config: Optional[PharmaGuardConfig] = None,
    config_path: Optional[str] = None,
) -> gr.Blocks:
    """Construct the Gradio dashboard interface."""
    cfg = config or pipeline.config
    stream_cfg = cfg.stream
    model_info = pipeline.detector.get_model_info()
    device = str(model_info.get("device", "cpu"))

    def process_webcam_stream(frame_in: Optional[np.ndarray]):
        """Per-frame stream handler: returns only the annotated frame (keep this path lean)."""
        if frame_in is None:
            return pipeline._create_empty_output(0.0, "STANDBY").annotated_frame_rgb
        return pipeline.process_frame(frame_in).annotated_frame_rgb

    def refresh_dashboard(cache: Dict[str, Any]):
        """1 Hz poller for panels. Unchanged HTML is skipped to avoid needless re-renders."""
        tele = pipeline.telemetry_snapshot()
        last = pipeline.last_output
        t = format_telemetry(tele, last, device)
        alerts_html = format_alerts_html(pipeline.alert_manager.get_recent_alerts(limit=10))
        events_html = format_events_html(pipeline.event_manager.get_recent(limit=15))
        cache = dict(cache or {})
        out_alerts = gr.skip() if cache.get("alerts") == alerts_html else alerts_html
        out_events = gr.skip() if cache.get("events") == events_html else events_html
        cache["alerts"], cache["events"] = alerts_html, events_html
        return (t["status"], t["tracked"], t["ids"], t["fps"], t["latency"], out_alerts, out_events, cache)

    alarm_volume = cfg.alarm.volume

    def _operator(request: Optional[gr.Request]) -> str:
        return (getattr(request, "username", None) or "operator") if request is not None else "operator"

    def refresh_alarm(last_html: str):
        html = format_alarm_html(pipeline.alarm.snapshot(), alarm_volume)
        return (gr.skip(), last_html) if html == last_html else (html, html)

    def on_acknowledge(request: gr.Request = None):
        pipeline.alarm.acknowledge(_operator(request))
        html = format_alarm_html(pipeline.alarm.snapshot(), alarm_volume)
        return html, html

    def on_alarm_reset(request: gr.Request = None):
        pipeline.alarm.reset(_operator(request))
        html = format_alarm_html(pipeline.alarm.snapshot(), alarm_volume)
        return html, html

    def on_alarm_test(request: gr.Request = None):
        if not pipeline.alarm.test(_operator(request)):
            gr.Warning("A real alarm is active - acknowledge/reset it before testing.")
        html = format_alarm_html(pipeline.alarm.snapshot(), alarm_volume)
        return html, html

    def on_reset_click():
        pipeline.reset_state()
        return (
            "0",
            "None",
            "0.0",
            "-",
            format_alerts_html([]),
            format_events_html([]),
            {},
        )

    def update_thresholds(conf: float, alert_threshold: float):
        # Detector floor stays low so ByteTrack's low-confidence stage keeps occluded people;
        # the operator-facing threshold governs which detections can start/drive tracks.
        trk = pipeline.tracker.config
        trk.track_high_threshold = conf
        trk.new_track_threshold = min(1.0, conf + 0.05)
        trk.track_low_threshold = min(trk.track_low_threshold, conf)
        beh = pipeline.behavior_analyzer.config
        beh.alert_threshold = alert_threshold
        # Keep REVIEW strictly below ALERT and above what browsing alone can reach.
        beh.review_threshold = max(beh.browsing_cap + 0.05, min(beh.review_threshold, alert_threshold - 0.1))
        return (f"Updated: person conf={conf:.2f}, alert at score {alert_threshold:.2f}, "
                f"review at {beh.review_threshold:.2f}")

    with gr.Blocks(title="PharmaGuard — Pharmacy AI Surveillance") as demo:
        # Custom CSS injection (Gradio 4/5/6 universal compatibility)
        gr.HTML(f"<style>{SECURITY_THEME_CSS}</style>")
        panel_cache = gr.State({})

        # Header banner
        with gr.Row(elem_classes=["header-banner"]):
            with gr.Column(scale=8):
                gr.HTML("""
                <div class='header-title'>🛡️ PHARMAGUARD — AI Pharmacy Security Monitor</div>
                <div class='header-subtitle'>Flags potentially suspicious shelf interactions for staff review - it does not determine theft or intent</div>
                """)
            with gr.Column(scale=4, min_width=200):
                gr.HTML(f"""
                <div style='text-align: right; padding-top: 4px;'>
                    <span style='background: #065f46; color: #34d399; font-size: 12px; font-weight: 600; padding: 4px 10px; border-radius: 9999px;'>
                        ● COMPUTE: {escape(device.upper())}
                    </span>
                    <div style='font-size: 11px; color: #64748b; margin-top: 4px;'>Engine: {escape(str(model_info.get('model_name')))} + ByteTrack (Kalman)</div>
                </div>
                """)

        # Alarm strip: banner + operator controls (always visible, above the video).
        alarm_cache = gr.State("")
        with gr.Row():
            with gr.Column(scale=9):
                alarm_box = gr.HTML(value=format_alarm_html(pipeline.alarm.snapshot(), alarm_volume))
                gr.HTML("<div id='pg-sound-status' style='font-size:12px;color:#f87171;'>"
                        "Alarm sound: waiting for page script...</div>")
            with gr.Column(scale=3, min_width=220):
                ack_btn = gr.Button("Acknowledge alarm (silence)", variant="stop", size="sm")
                alarm_reset_btn = gr.Button("Reset alarms (resolved)", variant="secondary", size="sm")
                test_btn = gr.Button("Test alarm", variant="secondary", size="sm")

        with gr.Tabs():
            with gr.Tab("Live monitor"):
                # Main surveillance view
                with gr.Row():
                    # Left column: Video streams
                    with gr.Column(scale=7):
                        # Primary path: pipelined HTTP fast stream (pharmaguard/fast_stream.py).
                        gr.HTML("<div style='font-weight:600;color:#38bdf8;font-size:13px;'>"
                                "Live camera (fast stream) - annotated output</div>")
                        gr.HTML(FAST_STREAM_HTML)
                        with gr.Accordion("Fallback: Gradio webcam stream (slower; use only if the fast "
                                          "stream does not start; never run both at once)", open=False):
                            with gr.Row():
                                webcam_input = gr.Image(
                                    sources=["webcam"],
                                    streaming=True,
                                    label="1. Camera input (click to start webcam)",
                                    type="numpy",
                                    webcam_options=gr.WebcamOptions(
                                        mirror=False,
                                        constraints={"video": {
                                            "width": {"ideal": stream_cfg.capture_width},
                                            "height": {"ideal": stream_cfg.capture_height},
                                        }},
                                    ),
                                )
                                annotated_output = gr.Image(
                                    label="2. PharmaGuard annotated output",
                                    type="numpy",
                                    format=stream_cfg.output_format,
                                    interactive=False,
                                    elem_id="pg-output",
                                )

                    # Right column: Telemetry & Alert Operations Console
                    with gr.Column(scale=5):
                        with gr.Row():
                            with gr.Column(scale=1):
                                status_val = gr.Textbox(value=SIGNAL_LABELS["STANDBY"], label="Camera / Stream Status", interactive=False)
                            with gr.Column(scale=1):
                                tracked_val = gr.Textbox(value="0", label="Persons Tracked", interactive=False)

                        with gr.Row():
                            with gr.Column(scale=1):
                                fps_val = gr.Textbox(value="0.0", label="Delivered FPS (end-to-end, server-measured)", interactive=False)
                            with gr.Column(scale=1):
                                gr.HTML(
                                    "<div style='font-size:12px;color:#94a3b8'>Displayed FPS (this browser)</div>"
                                    "<div id='pg-client-fps' style='font-size:20px;font-weight:600'>0</div>"
                                )

                        with gr.Row():
                            latency_val = gr.Textbox(value="-", label="Latency breakdown (measured)", interactive=False)

                        with gr.Row():
                            active_ids_val = gr.Textbox(value="None", label="Active Tracking IDs", interactive=False)

                        # Security Alerts Feed
                        gr.HTML("<div style='font-weight: 600; color: #f87171; font-size: 13px; margin-top: 10px;'>🚨 ALERTS FOR STAFF REVIEW</div>")
                        alerts_box = gr.HTML(value=format_alerts_html([]), elem_classes=["alert-box"])

                        # Activity & Audit Event Timeline
                        gr.HTML("<div style='font-weight: 600; color: #38bdf8; font-size: 13px; margin-top: 10px;'>📋 AUDIT & EVENT TIMELINE</div>")
                        events_box = gr.HTML(value=format_events_html([]), elem_classes=["event-box"])

                        # Controls
                        with gr.Accordion("⚙️ Detection & alert thresholds", open=False):
                            conf_slider = gr.Slider(
                                minimum=0.2, maximum=0.9, value=cfg.tracker.track_high_threshold, step=0.05,
                                label="Person confidence threshold (tracking)"
                            )
                            dwell_slider = gr.Slider(
                                minimum=0.6, maximum=2.0, value=cfg.behavior.alert_threshold, step=0.05,
                                label="Alert threshold (suspicion score; shelf->pocket kept 4 s ~1.05, shelf->bag kept 4 s ~1.3)"
                            )
                            apply_btn = gr.Button("Apply Settings", size="sm")
                            cfg_status = gr.Textbox(label="Config Status", interactive=False, max_lines=1)
                            reset_btn = gr.Button("Reset Tracking & Clear Alerts", variant="secondary", size="sm")

            review_tab = gr.Tab("Event review")
            with review_tab:
                review_loader = build_review_tab(pipeline)

            with gr.Tab("Zones"):
                build_zones_tab(pipeline, config_path)

        panel_outputs = [status_val, tracked_val, active_ids_val, fps_val, latency_val, alerts_box, events_box]

        apply_btn.click(fn=update_thresholds, inputs=[conf_slider, dwell_slider], outputs=[cfg_status])
        reset_btn.click(
            fn=on_reset_click,
            inputs=[],
            outputs=[tracked_val, active_ids_val, fps_val, latency_val, alerts_box, events_box, panel_cache],
        )

        # Per-frame path: image only. api_name lets tools/e2e_benchmark.py drive the same endpoint.
        webcam_input.stream(
            fn=process_webcam_stream,
            inputs=[webcam_input],
            outputs=[annotated_output],
            stream_every=stream_cfg.stream_every_sec,
            show_progress="hidden",
            concurrency_limit=1,
            concurrency_id="frame_pipeline",
            api_name="stream_frame",
        )

        # Request/response twin of the stream endpoint for tools/e2e_benchmark.py.
        # gradio_client cannot drive a streaming event one frame at a time (it holds the stream open),
        # so this hidden endpoint runs the SAME handler with the SAME JPEG output encoding.
        bench_in = gr.Image(type="numpy", visible=False)
        bench_out = gr.Image(type="numpy", format=stream_cfg.output_format, visible=False)
        bench_in.change(
            fn=process_webcam_stream,
            inputs=[bench_in],
            outputs=[bench_out],
            show_progress="hidden",
            concurrency_limit=1,
            concurrency_id="frame_pipeline",
            api_name="process_frame",
        )

        # Alarm: fast poll with a tiny payload (skipped when unchanged), plus immediate button feedback.
        alarm_outputs = [alarm_box, alarm_cache]
        ack_btn.click(fn=on_acknowledge, inputs=None, outputs=alarm_outputs, api_name="alarm_acknowledge")
        alarm_reset_btn.click(fn=on_alarm_reset, inputs=None, outputs=alarm_outputs, api_name="alarm_reset")
        test_btn.click(fn=on_alarm_test, inputs=None, outputs=alarm_outputs, api_name="alarm_test")
        alarm_timer = gr.Timer(cfg.alarm.poll_interval_sec)
        alarm_timer.tick(fn=refresh_alarm, inputs=[alarm_cache], outputs=alarm_outputs, show_progress="hidden")

        # Event review: refresh on page load and whenever the tab is opened.
        if review_loader is not None:
            load_fn, load_in, load_out = review_loader
            demo.load(load_fn, load_in, load_out)
            review_tab.select(load_fn, load_in, load_out)

        # Panels: 1 Hz, off the per-frame path.
        timer = gr.Timer(1.0)
        timer.tick(
            fn=refresh_dashboard,
            inputs=[panel_cache],
            outputs=panel_outputs + [panel_cache],
            show_progress="hidden",
        )

    return demo
