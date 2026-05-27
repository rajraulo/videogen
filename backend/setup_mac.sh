#!/bin/bash
# One-time setup for VideoGen on Apple Silicon.
# Run this once after cloning. Then use start_mac_server.sh every time.

set -e
cd "$(dirname "$0")"

echo "================================================"
echo "  VideoGen Setup  (Apple Silicon / MPS)"
echo "================================================"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -f ".venv/bin/activate" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo "Virtual environment ready."
echo ""

echo "Installing dependencies (one-time, ~2-3 min)..."
pip install --upgrade pip --quiet
pip install \
    fastapi "uvicorn[standard]" \
    "diffusers>=0.33.0" transformers accelerate sentencepiece \
    torch torchvision torchaudio \
    imageio imageio-ffmpeg python-multipart \
    openai-whisper librosa soundfile

echo ""
echo "================================================"
echo "  Setup complete!"
echo "  Start the server with: ./start_mac_server.sh"
echo "================================================"
