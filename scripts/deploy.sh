#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy.sh — Deploy / redeploy Souli to GCE VM
#
# Usage:
#   ./scripts/deploy.sh                    # deploy to VM_NAME in .env
#   ./scripts/deploy.sh souli-vm us-central1-a my-project
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# Load .env if present
[ -f .env ] && source .env

VM_NAME="${1:-${GCE_VM_NAME:-souli-vm}}"
ZONE="${2:-${GCE_ZONE:-us-central1-a}}"
PROJECT="${3:-${GCE_PROJECT:-}}"

PROJECT_FLAG=""
[ -n "$PROJECT" ] && PROJECT_FLAG="--project $PROJECT"

REMOTE_DIR="/opt/souli"

echo "Deploying to VM: $VM_NAME (zone: $ZONE)"

# ── 1. Sync source code ───────────────────────────────────────────────────────
echo "[1/4] Syncing source code..."
gcloud compute scp --recurse $PROJECT_FLAG --zone "$ZONE" \
    souli_pipeline configs Dockerfile requirements.txt pyproject.toml \
    docker-compose.gcp.yml \
    "${VM_NAME}:${REMOTE_DIR}/"

# ── 2. Sync data files ────────────────────────────────────────────────────────
echo "[2/4] Syncing data files..."
gcloud compute scp $PROJECT_FLAG --zone "$ZONE" \
    "souli_pipeline/data/Souli_EnergyFramework_PW (1).xlsx" \
    souli_pipeline/data/videos.csv \
    "${VM_NAME}:${REMOTE_DIR}/data/" 2>/dev/null || echo "  (data files not found locally, skipping)"

# ── 3. Rebuild souli container ────────────────────────────────────────────────
echo "[3/4] Rebuilding souli container on VM..."
gcloud compute ssh $PROJECT_FLAG --zone "$ZONE" "$VM_NAME" -- \
    "cd ${REMOTE_DIR} && docker compose -f docker-compose.gcp.yml build souli"

# ── 4. Restart services ───────────────────────────────────────────────────────
echo "[4/4] Restarting services..."
gcloud compute ssh $PROJECT_FLAG --zone "$ZONE" "$VM_NAME" -- \
    "cd ${REMOTE_DIR} && docker compose -f docker-compose.gcp.yml up -d"

echo ""
echo "✓ Deployed to $VM_NAME"
echo ""
echo "Check status:"
echo "  gcloud compute ssh ${VM_NAME} --zone ${ZONE} -- 'cd /opt/souli && docker compose -f docker-compose.gcp.yml ps'"
echo ""
echo "Run pipeline:"
echo "  gcloud compute ssh ${VM_NAME} --zone ${ZONE} -- \\"
echo "    'cd /opt/souli && docker compose -f docker-compose.gcp.yml exec souli \\"
echo "     souli run all --config /app/configs/pipeline.gcp.yaml \\"
echo "     --videos-csv /app/data/videos.csv \\"
echo "     --excel-path \"/app/data/Souli_EnergyFramework_PW (1).xlsx\" --merge'"
