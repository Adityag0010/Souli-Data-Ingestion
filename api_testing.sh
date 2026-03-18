# Souli API — Testing Guide
# ============================================================
# All examples use curl. Replace VM_IP with your GCP VM's
# external IP (or localhost if running locally).
# ============================================================

#!/bin/bash
# Usage:
#   ./api_testing.sh              → uses localhost (local dev)
#   ./api_testing.sh gcp          → auto-fetches GCP VM IP
#   ./api_testing.sh 34.xx.xx.xx  → uses that specific IP

if [ "$1" = "gcp" ]; then
  # Auto-fetch from GCP — no hardcoding needed
  source .env
  VM_IP=$(gcloud compute instances describe $GCE_VM_NAME \
    --zone=$GCE_ZONE --project=$GCE_PROJECT \
    --format="get(networkInterfaces[0].accessConfigs[0].natIP)")
  echo "Using GCP IP: $VM_IP"
elif [ -n "$1" ]; then
  VM_IP="$1"   # IP passed directly as argument
else
  VM_IP="localhost"   # default: local dev
fi

BASE="http://${VM_IP}:8000"
echo "Testing: $BASE"
echo "---"

# Health check
echo "1. Health check:"
curl -s ${BASE}/health | python3 -m json.tool

# Greeting
echo -e "\n2. Greeting:"
curl -s -X POST ${BASE}/session/greeting \
  -F "session_id=test-001" | python3 -m json.tool

# Chat
echo -e "\n3. Chat:"
curl -s -X POST ${BASE}/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test-001","message":"I feel really overwhelmed"}' \
  | python3 -m json.tool



# ── 5. Check Session State ────────────────────────────────────────────────────
# See current phase, detected energy node, etc.

curl -s ${BASE}/session/test-user-001/state | python3 -m json.tool


# ── 6. Reset Session ──────────────────────────────────────────────────────────
# Start a fresh conversation (user taps "New Session")

curl -s -X POST ${BASE}/session/reset \
  -F "session_id=test-user-001" | python3 -m json.tool


# ── 7. Voice Chat ─────────────────────────────────────────────────────────────
# Upload a .wav audio file, get back MP3 audio.
# The transcript and reply text come back in response HEADERS.

# First record a test audio file (requires sox or ffmpeg):
# sox -n -r 16000 -c 1 test_voice.wav trim 0.0 3.0

# Then send it:
curl -s -X POST ${BASE}/voice \
  -F "session_id=test-user-001" \
  -F "audio=@/path/to/test_voice.wav" \
  --output reply.mp3 \
  --dump-header headers.txt

# Check headers for transcript and reply text:
cat headers.txt | grep -E "X-Transcript|X-Reply|X-Phase"

# Play the audio reply:
# mpv reply.mp3   OR   afplay reply.mp3  (macOS)


# ── 8. Open the interactive API docs ─────────────────────────────────────────
# FastAPI automatically generates a Swagger UI. Open in browser:
#   http://localhost:8000/docs      ← interactive, can test endpoints here
#   http://localhost:8000/redoc     ← clean documentation view

echo "Open http://${VM_IP}:${PORT}/docs in your browser for Swagger UI"


# ── Running Locally (without Docker) ─────────────────────────────────────────
# Install FastAPI and uvicorn first:
#   pip install fastapi uvicorn python-multipart

# Then run:
#   uvicorn souli_pipeline.api:app --host 0.0.0.0 --port 8000 --reload
#
# --reload means the server restarts automatically when you edit api.py


# ── Running on GCP (in Docker) ────────────────────────────────────────────────
# The souli-api service in docker-compose.gcp.yml handles this.
# After deploying, the API is at:
#   http://<GCP_VM_EXTERNAL_IP>:8000
#
# To open port 8000 on GCP firewall (run once):
#   gcloud compute firewall-rules create souli-api \
#     --allow=tcp:8000 \
#     --target-tags=souli-server \
#     --description="Souli FastAPI for mobile app"


# ── What mobile app team needs to know ───────────────────────────────────────
# Base URL: http://<GCP_VM_EXTERNAL_IP>:8000
#
# Endpoint summary:
#   GET  /health                     → service health check
#   POST /session/greeting           → form: session_id → get opening message
#   POST /chat                       → JSON: {session_id, message} → reply + metadata
#   POST /voice                      → form: session_id + audio file → MP3 + headers
#   GET  /session/{session_id}/state → current phase, energy_node, etc.
#   POST /session/reset              → form: session_id → fresh conversation
#
# Full interactive docs: http://<GCP_VM_EXTERNAL_IP>:8000/docs