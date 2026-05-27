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

LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "unknown")
echo "Your Mac's local IP: $LOCAL_IP"
echo "In VideoGenApiClient.kt set:"
echo "  private const val BASE_URL = \"http://$LOCAL_IP:8000/\""
echo ""
echo "Model: LTX-Video on MPS  |  Override: export VIDEOGEN_MODEL=wan-1.3b"
echo "Press Ctrl+C to stop."
echo ""

export VIDEOGEN_MODEL="${VIDEOGEN_MODEL:-Lightricks/LTX-Video-0.9.1}"
export VIDEOGEN_DEVICE="mps"

uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 1
