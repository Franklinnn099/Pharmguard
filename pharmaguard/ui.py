"""
Gradio web interface module for PharmaGuard.
Provides a modern, high-contrast pharmacy security monitoring console
with real-time browser webcam streaming, low-latency queues, and live telemetry.
"""

from typing import Tuple, List, Optional
import gradio as gr
import numpy as np

from pharmaguard.pipeline import PharmaGuardPipeline, PipelineOutput
from pharmaguard.config import PharmaGuardConfig


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
                <strong style='color: {color}; font-size: 13px;'>[{a.severity}] {a.title}</strong>
                <span style='font-size: 11px; color: #94a3b8;'>{a.timestamp_iso[-8:]}</span>
            </div>
            <div style='color: #cbd5e1; font-size: 12px; margin-top: 4px;'>{a.message}</div>
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
            <span style='color: #64748b;'>{e.timestamp_iso}</span>
            <span style='color: {cat_color}; font-weight: 600; margin-left: 6px;'>[{e.category}]</span>
            <span style='color: #e2e8f0; margin-left: 6px;'>{e.message}</span>
        </div>
        """
    html += "</div>"
    return html


def create_pharmaguard_ui(
    pipeline: PharmaGuardPipeline,
    config: Optional[PharmaGuardConfig] = None
) -> gr.Blocks:
    """Construct the Gradio dashboard interface."""
    cfg = config or pipeline.config

    def process_webcam_stream(frame_in: Optional[np.ndarray]):
        """Stream handler callback invoked for each webcam frame."""
        if frame_in is None:
            # Return empty/standby states gracefully
            empty_out = pipeline._create_empty_output(0.0, "STANDBY")
            return (
                empty_out.annotated_frame_rgb,
                "STANDBY (Webcam Paused/Inactive)",
                "0",
                "None",
                "0.0",
                "0 ms",
                format_alerts_html([]),
                format_events_html(empty_out.recent_events)
            )

        # Process frame through full vision pipeline
        output: PipelineOutput = pipeline.process_frame(frame_in)

        active_ids_str = ", ".join(f"#{tid}" for tid in output.active_ids) if output.active_ids else "None"
        alerts_html = format_alerts_html(output.recent_alerts)
        events_html = format_events_html(output.recent_events)

        # Format fine-grained latency breakdown
        stage_timings = output.metadata.get("stage_timings", {})
        det_ms = stage_timings.get("detect_ms", 0.0)
        trk_ms = stage_timings.get("track_ms", 0.0)
        hud_ms = stage_timings.get("annotate_ms", 0.0)
        latency_str = f"{output.latency_ms:.0f} ms (Det: {det_ms:.0f}ms, Trk: {trk_ms:.0f}ms, HUD: {hud_ms:.0f}ms)"

        device_str = output.metadata.get("device", "cpu").upper()
        status_text = f"ONLINE [{output.connection_status}] | {device_str}"

        return (
            output.annotated_frame_rgb,
            status_text,
            str(output.active_tracks_count),
            active_ids_str,
            f"{output.fps:.1f}",
            latency_str,
            alerts_html,
            events_html
        )

    def on_reset_click():
        pipeline.reset_state()
        return (
            "0",
            "None",
            "0.0",
            "0 ms",
            format_alerts_html([]),
            format_events_html([])
        )

    def update_thresholds(conf: float, dwell: float):
        pipeline.detector.config.confidence_threshold = conf
        for z in pipeline.config.zones:
            z.dwell_alert_threshold_sec = dwell
        return f"Updated: Conf={conf:.2f}, Dwell={dwell:.1f}s"

    with gr.Blocks(title="PharmaGuard — Pharmacy AI Surveillance") as demo:
        # Custom CSS injection (Gradio 4/5/6 universal compatibility)
        gr.HTML(f"<style>{SECURITY_THEME_CSS}</style>")

        # Header banner
        with gr.Row(elem_classes=["header-banner"]):
            with gr.Column(scale=8):
                gr.HTML("""
                <div class='header-title'>🛡️ PHARMAGUARD — AI Pharmacy Security Monitor</div>
                <div class='header-subtitle'>Real-time suspicious interaction sequence detection & shelf dwell monitoring</div>
                """)
            with gr.Column(scale=4, min_width=200):
                device_badge = pipeline.detector.get_model_info().get("device", "cpu").upper()
                model_name = pipeline.detector.get_model_info().get("model_name", "yolov8n.pt")
                gr.HTML(f"""
                <div style='text-align: right; padding-top: 4px;'>
                    <span style='background: #065f46; color: #34d399; font-size: 12px; font-weight: 600; padding: 4px 10px; border-radius: 9999px;'>
                        ● SEC-OPS LIVE [{device_badge}]
                    </span>
                    <div style='font-size: 11px; color: #64748b; margin-top: 4px;'>Engine: {model_name} + ByteTrack</div>
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
                        label="1. Browser Camera Input (Live Feed - Click Record/Webcam to Start)",
                        type="numpy"
                    )
                    annotated_output = gr.Image(
                        label="2. PharmaGuard Real-Time AI Output (Annotated Feed)",
                        type="numpy",
                        interactive=False
                    )

            # Right column: Telemetry & Alert Operations Console
            with gr.Column(scale=5):
                # Status & Metrics Grid
                with gr.Row():
                    with gr.Column(scale=1):
                        status_val = gr.Textbox(value="ONLINE", label="System Status", interactive=False)
                    with gr.Column(scale=1):
                        tracked_val = gr.Textbox(value="0", label="Persons Tracked", interactive=False)

                with gr.Row():
                    with gr.Column(scale=1):
                        fps_val = gr.Textbox(value="0.0", label="Processing FPS", interactive=False)
                    with gr.Column(scale=1):
                        latency_val = gr.Textbox(value="0 ms", label="Frame Latency & Breakdown", interactive=False)

                with gr.Row():
                    active_ids_val = gr.Textbox(value="None", label="Active Tracking IDs", interactive=False)

                # Security Alerts Feed
                gr.HTML("<div style='font-weight: 600; color: #f87171; font-size: 13px; margin-top: 10px;'>🚨 REAL-TIME SECURITY ALERTS</div>")
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

                    apply_btn.click(
                        fn=update_thresholds,
                        inputs=[conf_slider, dwell_slider],
                        outputs=[cfg_status]
                    )

                    reset_btn.click(
                        fn=on_reset_click,
                        inputs=[],
                        outputs=[tracked_val, active_ids_val, fps_val, latency_val, alerts_box, events_box]
                    )

        # Wire webcam stream to pipeline processor with low-latency streaming parameters
        webcam_input.stream(
            fn=process_webcam_stream,
            inputs=[webcam_input],
            outputs=[
                annotated_output,
                status_val,
                tracked_val,
                active_ids_val,
                fps_val,
                latency_val,
                alerts_box,
                events_box
            ],
            stream_every=0.04,  # ~25 FPS stream rate (avoids default 0.5s = 2 FPS bottleneck)
            show_progress="hidden",
            concurrency_limit=1
        )

    return demo
