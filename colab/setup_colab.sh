#!/bin/bash
# PharmaGuard Colab Environment Setup Script
set -e

echo "=== [PharmaGuard] Checking GPU Availability ==="
nvidia-smi || echo "Warning: nvidia-smi failed. Ensure GPU runtime is selected."

echo "=== [PharmaGuard] Installing Pinned Dependencies ==="
pip install -r requirements.txt

echo "=== [PharmaGuard] Testing Model Download & Preload ==="
python -c "
from ultralytics import YOLO
import torch
print('PyTorch CUDA:', torch.cuda.is_available())
model = YOLO('yolov8n.pt')
print('Model pre-cached successfully.')
"

echo "=== [PharmaGuard] Setup Complete ==="
