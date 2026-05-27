#!/bin/bash
# VideoGen Server — Apple Silicon (M1/M2/M3/M4)
# Runs LTX-Video on MPS (Metal GPU). No CUDA needed.
#
# Usage:
#   chmod +x start_mac_server.sh
#   ./start_mac_server.sh

set -e
cd "$(dirname "$0")"

echo "================================================"
echo "  VideoGen Server  (Apple Silicon / MPS)"
echo "================================================"
echo ""

# Install dependencies
echo "Checking dependencies..."
pip install fastapi "uvicorn[standard]" \
    "diffusers>=0.33.0" transformers accelerate sentencepiece \
    torch torchvision torchaudio \
    imageio imageio-ffmpeg python-multipart \
    openai-whisper librosa soundfile --quiet

echo ""

# Print local IP so you can update VideoGenApiClient.kt
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "unknown")
echo "Your Mac's local IP: $LOCAL_IP"
echo ""
echo "In VideoGenApiClient.kt set:"
echo "  private const val BASE_URL = \"http://$LOCAL_IP:8000/\""
echo ""
echo "Default model: LTX-Video on MPS (Metal GPU)"
echo "Override:  export VIDEOGEN_MODEL=wan-1.3b"
echo "Press Ctrl+C to stop."
echo ""

# Use LTX-Video by default on M4 Pro — fits in 24 GB unified memory
export VIDEOGEN_MODEL="${VIDEOGEN_MODEL:-Lightricks/LTX-Video-0.9.1}"
export VIDEOGEN_DEVICE="mps"

uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 1
