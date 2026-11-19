"""
Main application launcher for PharmaGuard.
Supports running in Google Colab (with public Gradio share link & secure auth) or locally on workstation.
"""

import argparse
import atexit
import logging
import os
import secrets
import sys
import torch

# Privacy: a surveillance system must not phone home. Must be set before gradio is imported.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from pharmaguard.config import PharmaGuardConfig
from pharmaguard.pipeline import PharmaGuardPipeline
from pharmaguard.ui import create_pharmaguard_ui, DASHBOARD_HEAD


def resolve_auth_credentials(args):
    """Resolve authentication credentials securely from Colab Secrets, Env Vars, or Ephemeral Generation.

    Never uses hardcoded default passwords.
    """
    if args.no_auth:
        return None

    username = args.auth_user or os.environ.get("PHARMAGUARD_AUTH_USER")
    password = args.auth_pass or os.environ.get("PHARMAGUARD_AUTH_PASS")

    # Attempt to load from Google Colab Secrets if running inside Colab
    if not (username and password):
        try:
            from google.colab import userdata
            username = username or userdata.get("PHARMAGUARD_AUTH_USER")
            password = password or userdata.get("PHARMAGUARD_AUTH_PASS")
        except Exception:
            pass

    # If auth requested or in public cloud share mode, ensure credentials exist
    if args.auth or args.colab or args.share:
        if not username:
            username = "security_operator"
        if not password:
            # Generate a cryptographically secure ephemeral session key
            password = secrets.token_urlsafe(12)
            print(f"[SECURITY] Generated one-time ephemeral session credentials:")
            print(f"           Username: {username}")
            print(f"           Password: {password}")
            print(f"           (To set your own, export PHARMAGUARD_AUTH_USER and PHARMAGUARD_AUTH_PASS)")
        return (username, password)

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="PharmaGuard Real-Time Pharmacy Security Monitoring")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--share", action="store_true", help="Generate public Gradio URL (Required for Colab)")
    parser.add_argument("--colab", action="store_true", help="Enable Colab-optimized settings (share=True, host=0.0.0.0)")
    parser.add_argument("--local", action="store_true", help="Local workstation mode (default): bind 127.0.0.1, no share link")
    parser.add_argument("--max-width", type=int, default=None, help="Server-side processing/return width (default 640)")
    parser.add_argument("--imgsz", type=int, default=None, help="Model inference size (default 640; 480/416 = faster)")
    parser.add_argument("--data-dir", type=str, default=None, help="Local event DB + clips directory (default data/)")
    parser.add_argument("--mirror-dir", type=str, default=None,
                        help="Durable copy target, e.g. /content/drive/MyDrive/PharmaGuard (Colab + Drive)")
    parser.add_argument("--camera-id", type=str, default=None, help="Camera identifier stored with every event")
    parser.add_argument("--no-recording", action="store_true", help="Disable the event database and evidence clips")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind Gradio UI (default 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default 0.0.0.0)")
    parser.add_argument("--model", type=str, default=None, help="YOLO model name override (e.g. yolov8n.pt)")
    parser.add_argument("--device", type=str, default=None, help="Device override (cuda, cpu, auto)")
    parser.add_argument("--auth", action="store_true", help="Require authentication for access")
    parser.add_argument("--no-auth", action="store_true", help="Disable authentication even in public share mode")
    parser.add_argument("--auth-user", type=str, default=None, help="Authentication username override")
    parser.add_argument("--auth-pass", type=str, default=None, help="Authentication password override")
    return parser.parse_args()


def main():
    args = parse_args()
    # Windows consoles/pipes may default to cp1252; never let a status emoji crash the server.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpx2", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # 1. Load Configuration
    if args.config and os.path.exists(args.config):
        print(f"[PharmaGuard] Loading configuration from {args.config}")
        config = PharmaGuardConfig.load_from_yaml(args.config)
    else:
        config = PharmaGuardConfig()

    # 2. Overrides from CLI
    if args.model:
        config.detector.model_name = args.model
    if args.device:
        config.detector.device = args.device
    if args.max_width:
        config.stream.max_process_width = args.max_width
    if args.imgsz:
        config.detector.imgsz = args.imgsz
    if args.data_dir:
        config.recording.data_dir = args.data_dir
    if args.mirror_dir:
        config.recording.mirror_dir = args.mirror_dir
    if args.camera_id:
        config.recording.camera_id = args.camera_id
    if args.no_recording:
        config.recording.enabled = False
    if args.local and (args.colab or args.share):
        raise SystemExit("--local cannot be combined with --colab/--share")

    print(f"[PharmaGuard] Initializing Computer Vision Pipeline...")
    print(f"  - Model: {config.detector.model_name}")
    print(f"  - CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  - Notice: No GPU detected; executing inference on CPU.")

    pipeline = PharmaGuardPipeline(config=config)
    atexit.register(pipeline.close)   # finish evidence clips and mirror the event DB on shutdown
    if config.recording.enabled:
        print(f"  - Event DB: {os.path.abspath(pipeline.store.db_path)}"
              + (f"  (mirrored to {config.recording.mirror_dir})" if config.recording.mirror_dir else ""))

    # 3. Build UI & Configure Queue for High-Speed Streaming
    print(f"[PharmaGuard] Building Gradio Security Operations Dashboard...")
    demo = create_pharmaguard_ui(pipeline=pipeline, config=config)

    # Queue is required for streaming; the stream event itself is limited to 1 concurrent frame.
    demo.queue(default_concurrency_limit=2)

    # 4. Resolve Authentication & Launch Options
    share_mode = args.share or args.colab
    server_name = args.host if (args.colab or args.share) else "127.0.0.1"
    auth_config = resolve_auth_credentials(args)

    print(f"\n=========================================================")
    print(f"🛡️  PharmaGuard Security Monitor Active")
    print(f"    Mode: {'Colab Cloud' if args.colab else 'Local Workstation'}")
    print(f"    Public Share Link: {share_mode}")
    print(f"    Access URL: http://{server_name}:{args.port}")
    if auth_config:
        print(f"    Authentication: Enabled (User: {auth_config[0]})")
    else:
        print(f"    Authentication: Disabled")
    print(f"=========================================================\n")

    demo.launch(
        server_name=server_name,
        server_port=args.port,
        share=share_mode,
        auth=auth_config,
        show_error=True,
        head=DASHBOARD_HEAD,
    )


if __name__ == "__main__":
    main()
