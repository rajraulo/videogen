#!/bin/bash
# VideoGen Server — Apple Silicon (M1/M2/M3/M4)
# Fast start — skips pip install if already set up.
# First time? Run: ./setup_mac.sh

set -e
cd "$(dirname "$0")"

echo "================================================"
echo "  VideoGen Server  (Apple Silicon / MPS)"
echo "================================================"
echo ""

# Activate venv if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Quick check — if uvicorn missing, prompt user to run setup first
if ! python -c "import uvicorn" &>/dev/null; then
    echo "ERROR: Dependencies not installed."
    echo "Run ./setup_mac.sh first, then re-run this script."
    exit 1
fi

# Detect if running inside Docker
if [ -f "/.dockerenv" ]; then
    # Inside Docker — show the host gateway IP (Mac's IP on Docker bridge)
    HOST_IP=$(ip route | awk '/default/ {print $3}' | head -1)
    echo "Running inside Docker container."
    echo "Make sure you started Docker with: -p 8000:8000"
    echo ""
    echo "For the Android app, use your Mac's WiFi IP."
    echo "Find it on your Mac by running (outside Docker):"
    echo "  ipconfig getifaddr en0"
    echo ""
    echo "Then in VideoGenApiClient.kt set:"
    echo "  private const val BASE_URL = \"http://<your-mac-wifi-ip>:8000/\""
else
    # Running natively on macOS
    LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || \
               ipconfig getifaddr en1 2>/dev/null || \
               ifconfig | grep "inet " | grep -v "127.0.0.1" | awk '{print $2}' | head -1)
    echo "Your Mac's local IP: $LOCAL_IP"
    echo "In VideoGenApiClient.kt set:"
    echo "  private const val BASE_URL = \"http://$LOCAL_IP:8000/\""
fi
echo ""
echo "Model: LTX-Video on MPS  |  Override: export VIDEOGEN_MODEL=wan-1.3b"
echo "Press Ctrl+C to stop."
echo ""

# Wan2.1-1.3B is the default — more stable on MPS than LTX-Video
# Stable Diffusion image server — real AI images animated with Ken Burns zoom
# Model: stabilityai/stable-diffusion-2-1 (~1.7 GB, works on MPS)
# ~20-30 sec per video on M4 Pro
export VIDEOGEN_MODEL="${VIDEOGEN_MODEL:-stabilityai/stable-diffusion-2-1}"

if [ -z "$HF_TOKEN" ]; then
    echo "TIP: set HF_TOKEN=hf_xxx for faster downloads"
fi

uvicorn api.image_server:app --host 0.0.0.0 --port 8000 --workers 1
