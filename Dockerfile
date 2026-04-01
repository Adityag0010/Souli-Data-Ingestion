# ─────────────────────────────────────────────────────────────────────────────
# Souli Pipeline — Production Dockerfile
# Runs on GCE VM alongside Ollama + Qdrant containers via docker-compose.gcp.yml
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04
# System deps + Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    git \
    curl \
    mpg123 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Install Python deps first (layer cache)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Copy source
COPY . /app

# Install the souli package (registers the `souli` CLI entrypoint)
RUN pip install .

# Create data and outputs directories
RUN mkdir -p /app/data /app/outputs
ENV PYTHONUNBUFFERED=1
ENV SOULI_LOG_LEVEL=INFO
ENV SOULI_CONFIG=/app/configs/pipeline.gcp.yaml
CMD ["souli", "--help"]
