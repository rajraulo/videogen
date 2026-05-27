# VideoGen — GPU Docker image
# Base: official PyTorch CUDA image
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies (ffmpeg for video I/O)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsm6 libxext6 git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Model cache directory (mount a volume here for persistence)
ENV HF_HOME=/models/cache
ENV TRANSFORMERS_CACHE=/models/cache
ENV VIDEOGEN_OUTPUT_DIR=/outputs
ENV VIDEOGEN_API_KEY=change-me-in-production

# Create output dir
RUN mkdir -p /outputs /models/cache

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "300"]
