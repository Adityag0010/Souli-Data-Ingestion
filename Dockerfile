# ─────────────────────────────────────────────────────────────────────────────
# Souli Pipeline — Production Dockerfile
# Runs on GCE VM alongside Ollama + Qdrant containers via docker-compose.gcp.yml
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    mpg123 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . /app

# Create data and outputs directories
RUN mkdir -p /app/data /app/outputs

ENV PYTHONUNBUFFERED=1
ENV SOULI_LOG_LEVEL=INFO
ENV SOULI_CONFIG=/app/configs/pipeline.gcp.yaml

CMD ["souli", "--help"]
