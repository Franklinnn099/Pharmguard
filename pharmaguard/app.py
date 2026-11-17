"""
Main application launcher for PharmaGuard.
Supports running in Google Colab (with public Gradio share link & authentication) or locally on workstation.
"""

import argparse
import os
import sys
import torch

from pharmaguard.config import PharmaGuardConfig
from pharmaguard.pipeline import PharmaGuardPipeline
from pharmaguard.ui import create_pharmaguard_ui


def parse_args():
    parser = argparse.ArgumentParser(description="PharmaGuard Real-Time Pharmacy Security Monitoring")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--share", action="store_true", help="Generate public Gradio URL (Required for Colab)")
    parser.add_argument("--colab", action="store_true", help="Enable Colab-optimized settings (share=True, host=0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind Gradio UI (default 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default 0.0.0.0)")
    parser.add_argument("--model", type=str, default=None, help="YOLO model name override (e.g. yolov8n.pt)")
    parser.add_argument("--device", type=str, default=None, help="Device override (cuda, cpu, auto)")
    parser.add_argument("--auth", action="store_true", help="Enable basic authentication (admin/pharmguard2026)")
    parser.add_argument("--auth-user", type=str, default="admin", help="Authentication username")
    parser.add_argument("--auth-pass", type=str, default="pharmguard2026", help="Authentication password")
    return parser.parse_args()


def main():
    args = parse_args()

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

    print(f"[PharmaGuard] Initializing Computer Vision Pipeline...")
    print(f"  - Model: {config.detector.model_name}")
    print(f"  - CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  - Notice: No GPU detected; executing inference on CPU.")

    pipeline = PharmaGuardPipeline(config=config)

    # 3. Build UI & Configure Queue for High-Speed Streaming
    print(f"[PharmaGuard] Building Gradio Security Operations Dashboard...")
    demo = create_pharmaguard_ui(pipeline=pipeline, config=config)

    # Enable low-latency WebSocket queueing
    demo.queue(default_concurrency_limit=2)

    # 4. Determine Launch Options
    share_mode = args.share or args.colab
    server_name = args.host if (args.colab or args.share) else "127.0.0.1"

    auth_config = (args.auth_user, args.auth_pass) if args.auth else None

    print(f"\n=========================================================")
    print(f"🛡️  PharmaGuard Security Monitor Active")
    print(f"    Mode: {'Colab Cloud' if args.colab else 'Local Workstation'}")
    print(f"    Public Share Link: {share_mode}")
    print(f"    Access URL: http://{server_name}:{args.port}")
    if auth_config:
        print(f"    Authentication: Enabled (User: {args.auth_user})")
    print(f"=========================================================\n")

    demo.launch(
        server_name=server_name,
        server_port=args.port,
        share=share_mode,
        auth=auth_config,
        show_error=True
    )


if __name__ == "__main__":
    main()
