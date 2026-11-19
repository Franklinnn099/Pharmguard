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
  setInterval(() => {
    const el = document.getElementById('pg-client-fps');
    if (el) el.textContent = painted.toFixed(0);
    painted = 0;
  }, 1000);
})();
</script>
"""

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
            f"transport+encode {tele.outside_p50_ms:.0f} ms | {in_res}->{res}"
        ) if live else "-",
    }


def create_pharmaguard_ui(
    pipeline: PharmaGuardPipeline,
    config: Optional[PharmaGuardConfig] = None
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

    def update_thresholds(conf: float, dwell: float):
        pipeline.detector.config.confidence_threshold = conf
        for z in pipeline.config.zones:
            z.dwell_alert_threshold_sec = dwell
        return f"Updated: Conf={conf:.2f}, Dwell={dwell:.1f}s"

    with gr.Blocks(title="PharmaGuard — Pharmacy AI Surveillance") as demo:
        # Custom CSS injection (Gradio 4/5/6 universal compatibility)
        gr.HTML(f"<style>{SECURITY_THEME_CSS}</style>")
        panel_cache = gr.State({})

        # Header banner
        with gr.Row(elem_classes=["header-banner"]):
            with gr.Column(scale=8):
                gr.HTML("""
                <div class='header-title'>🛡️ PHARMAGUARD — AI Pharmacy Security Monitor</div>
                <div class='header-subtitle'>Flags potentially suspicious shelf interactions for staff review</div>
                """)
            with gr.Column(scale=4, min_width=200):
                gr.HTML(f"""
                <div style='text-align: right; padding-top: 4px;'>
                    <span style='background: #065f46; color: #34d399; font-size: 12px; font-weight: 600; padding: 4px 10px; border-radius: 9999px;'>
                        ● COMPUTE: {escape(device.upper())}
                    </span>
                    <div style='font-size: 11px; color: #64748b; margin-top: 4px;'>Engine: {escape(str(model_info.get('model_name')))} + IoU tracker</div>
                </div>
                """)

        # Main surveillance view
        with gr.Row():
            # Left column: Video streams
            with gr.Column(scale=7):
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
                with gr.Accordion("⚙️ Detection & Zone Configuration", open=False):
                    conf_slider = gr.Slider(
                        minimum=0.1, maximum=0.9, value=cfg.detector.confidence_threshold, step=0.05,
                        label="YOLO Person Confidence Threshold"
                    )
                    dwell_slider = gr.Slider(
                        minimum=2.0, maximum=30.0, value=cfg.zones[0].dwell_alert_threshold_sec, step=1.0,
                        label="Shelf Dwell Alert Threshold (seconds)"
                    )
                    apply_btn = gr.Button("Apply Settings", size="sm")
                    cfg_status = gr.Textbox(label="Config Status", interactive=False, max_lines=1)
                    reset_btn = gr.Button("Reset Tracking & Clear Alerts", variant="secondary", size="sm")

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

        # Panels: 1 Hz, off the per-frame path.
        timer = gr.Timer(1.0)
        timer.tick(
            fn=refresh_dashboard,
            inputs=[panel_cache],
            outputs=panel_outputs + [panel_cache],
            show_progress="hidden",
        )

    return demo
