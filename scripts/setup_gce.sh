#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_gce.sh — One-time GCE VM setup script
#
# Run this ONCE after creating the VM:
#   gcloud compute ssh souli-vm -- 'bash -s' < scripts/setup_gce.sh
#
# What it does:
#   1. Installs NVIDIA GPU drivers + CUDA toolkit
#   2. Installs Docker + NVIDIA Container Toolkit (so Docker can use the GPU)
#   3. Installs Docker Compose v2
#   4. Clones this repo to /opt/souli
#   5. Pulls Ollama models (llama3.1 + qwen2.5:1.5b)
#
# Recommended VM:  n1-standard-4  +  NVIDIA T4 GPU  +  Ubuntu 22.04 LTS
#   Cost: ~$0.35/hr GPU + ~$0.19/hr CPU = ~$0.54/hr
#   Persistent disk: 100 GB SSD (for models + data)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

echo "============================================================"
echo "  Souli GCE VM Setup"
echo "============================================================"

# ── 1. System packages ────────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    curl wget git unzip ca-certificates gnupg lsb-release \
    ffmpeg mpg123

# ── 2. NVIDIA GPU drivers ─────────────────────────────────────────────────────
echo "[2/6] Installing NVIDIA drivers..."
# Check if GPU is present
if lspci | grep -i nvidia > /dev/null; then
    # Install CUDA keyring
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends cuda-drivers
    echo "NVIDIA drivers installed."
else
    echo "No NVIDIA GPU detected — skipping GPU driver install."
    echo "NOTE: Ollama will run on CPU (slower). Recommend T4 GPU for production."
fi

# ── 3. Docker ─────────────────────────────────────────────────────────────────
echo "[3/6] Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sudo bash
    sudo usermod -aG docker "$USER"
    echo "Docker installed. You may need to re-login for group changes."
else
    echo "Docker already installed."
fi

# NVIDIA Container Toolkit (lets Docker use GPU)
if lspci | grep -i nvidia > /dev/null; then
    echo "  Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update -qq
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo "NVIDIA Container Toolkit installed."
fi

# ── 4. Docker Compose v2 ──────────────────────────────────────────────────────
echo "[4/6] Installing Docker Compose v2..."
if ! docker compose version &> /dev/null; then
    COMPOSE_VERSION="v2.27.1"
    sudo mkdir -p /usr/local/lib/docker/cli-plugins
    sudo curl -SL "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-x86_64" \
        -o /usr/local/lib/docker/cli-plugins/docker-compose
    sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
    echo "Docker Compose $(docker compose version) installed."
else
    echo "Docker Compose already installed: $(docker compose version)"
fi

# ── 5. Clone / update repo ────────────────────────────────────────────────────
echo "[5/6] Setting up /opt/souli..."
REPO_DIR="/opt/souli"
if [ ! -d "$REPO_DIR/.git" ]; then
    sudo git clone https://github.com/YOUR_ORG/souli-voice-pipeline.git "$REPO_DIR"
    sudo chown -R "$USER":"$USER" "$REPO_DIR"
    echo "Repo cloned to $REPO_DIR"
else
    cd "$REPO_DIR" && git pull
    echo "Repo updated."
fi

cd "$REPO_DIR"

# Copy .env if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "⚠️  Created .env from .env.example"
    echo "    Edit /opt/souli/.env and fill in LiveKit keys before starting."
    echo ""
fi

# ── 6. Pull Ollama models ─────────────────────────────────────────────────────
echo "[6/6] Starting Ollama and pulling models..."

# Start just Ollama first
docker compose -f docker-compose.gcp.yml up -d ollama

echo "  Waiting for Ollama to start..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "  Ollama is up."
        break
    fi
    sleep 3
done

echo "  Pulling llama3.1 (~4.7 GB)..."
docker compose -f docker-compose.gcp.yml exec ollama ollama pull llama3.1

echo "  Pulling qwen2.5:1.5b (~1 GB)..."
docker compose -f docker-compose.gcp.yml exec ollama ollama pull qwen2.5:1.5b

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Edit /opt/souli/.env (fill in LIVEKIT_URL, keys, etc.)"
echo "  2. Upload your data files:"
echo "     scp 'Souli_EnergyFramework_PW (1).xlsx' souli-vm:/opt/souli/data/"
echo "     scp souli_pipeline/data/videos.csv souli-vm:/opt/souli/data/"
echo "  3. Start all services:"
echo "     cd /opt/souli && docker compose -f docker-compose.gcp.yml up -d"
echo "  4. Run the full pipeline:"
echo "     docker compose -f docker-compose.gcp.yml exec souli \\"
echo "       souli run all \\"
echo "       --config /app/configs/pipeline.gcp.yaml \\"
echo "       --videos-csv /app/data/videos.csv \\"
echo "       --excel-path '/app/data/Souli_EnergyFramework_PW (1).xlsx' \\"
echo "       --merge"
echo "  5. Ingest into Qdrant:"
echo "     docker compose -f docker-compose.gcp.yml exec souli \\"
echo "       souli ingest --config /app/configs/pipeline.gcp.yaml"
echo "  6. Start a chat:"
echo "     docker compose -f docker-compose.gcp.yml exec souli \\"
echo "       souli chat --config /app/configs/pipeline.gcp.yaml \\"
echo "       --excel '/app/data/Souli_EnergyFramework_PW (1).xlsx'"
echo "============================================================"
