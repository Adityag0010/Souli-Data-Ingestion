#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# create_vm.sh — Create the GCE VM for Souli
#
# Usage:
#   ./scripts/create_vm.sh
#
# Requires:
#   gcloud CLI installed and authenticated
#   GCP project with Compute Engine API enabled
#   GPU quota approved (request at: console.cloud.google.com/iam-admin/quotas)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

[ -f .env ] && source .env

PROJECT="${GCE_PROJECT:-YOUR_GCP_PROJECT_ID}"
VM_NAME="${GCE_VM_NAME:-souli-vm}"
ZONE="${GCE_ZONE:-us-central1-a}"
MACHINE_TYPE="${GCE_MACHINE_TYPE:-n1-standard-4}"   # 4 vCPU, 15 GB RAM
GPU_TYPE="${GCE_GPU_TYPE:-nvidia-tesla-t4}"          # T4 GPU (~$0.35/hr)
DISK_SIZE="${GCE_DISK_SIZE:-100}"                    # GB — enough for models + data
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"

echo "Creating GCE VM: $VM_NAME"
echo "  Project:  $PROJECT"
echo "  Zone:     $ZONE"
echo "  Machine:  $MACHINE_TYPE"
echo "  GPU:      $GPU_TYPE"
echo "  Disk:     ${DISK_SIZE}GB SSD"
echo ""

gcloud compute instances create "$VM_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=${GPU_TYPE},count=1" \
    --maintenance-policy=TERMINATE \
    --restart-on-failure \
    --boot-disk-size="${DISK_SIZE}GB" \
    --boot-disk-type=pd-ssd \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --tags=souli-server,http-server,https-server \
    --metadata=enable-oslogin=true \
    --scopes=cloud-platform

echo ""
echo "✓ VM created: $VM_NAME"

# Create firewall rules for LiveKit WebRTC
echo ""
echo "Creating firewall rules..."
gcloud compute firewall-rules create souli-livekit-tcp \
    --project="$PROJECT" \
    --allow=tcp:7880,tcp:7881 \
    --target-tags=souli-server \
    --description="LiveKit HTTP and TURN/TCP" 2>/dev/null || echo "  (rule already exists)"

gcloud compute firewall-rules create souli-livekit-udp \
    --project="$PROJECT" \
    --allow=udp:50100-50200 \
    --target-tags=souli-server \
    --description="LiveKit WebRTC UDP range" 2>/dev/null || echo "  (rule already exists)"

gcloud compute firewall-rules create souli-qdrant \
    --project="$PROJECT" \
    --allow=tcp:6333,tcp:6334 \
    --target-tags=souli-server \
    --source-ranges=10.0.0.0/8 \
    --description="Qdrant (internal only)" 2>/dev/null || echo "  (rule already exists)"

echo ""
echo "✓ Firewall rules created."
echo ""
echo "Next: Run the setup script on the VM:"
echo "  gcloud compute ssh $VM_NAME --zone $ZONE --project $PROJECT \\"
echo "    -- 'bash -s' < scripts/setup_gce.sh"
