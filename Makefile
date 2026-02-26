.PHONY: install fmt lint run \
        gcp-vm gcp-setup gcp-deploy gcp-ssh gcp-logs gcp-status \
        gcp-pipeline gcp-ingest gcp-chat \
        compose-up compose-down compose-ps compose-logs \
        pull-models build

# ── Local dev ─────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

fmt:
	python -m pip install -q ruff && ruff format .

lint:
	python -m pip install -q ruff && ruff check .

run:
	souli --help

# ── Docker Compose (GCP / local with GPU) ────────────────────────────────────
build:
	docker compose -f docker-compose.gcp.yml build souli

compose-up:
	docker compose -f docker-compose.gcp.yml --env-file .env up -d

compose-down:
	docker compose -f docker-compose.gcp.yml down

compose-ps:
	docker compose -f docker-compose.gcp.yml ps

compose-logs:
	docker compose -f docker-compose.gcp.yml logs -f --tail=100

pull-models:
	@echo "Pulling llama3.1 and qwen2.5:1.5b into Ollama..."
	docker compose -f docker-compose.gcp.yml exec ollama ollama pull llama3.1
	docker compose -f docker-compose.gcp.yml exec ollama ollama pull qwen2.5:1.5b

# ── GCP VM management ─────────────────────────────────────────────────────────
# Load env vars from .env
include .env
export

gcp-vm:
	@echo "Creating GCE VM..."
	bash scripts/create_vm.sh

gcp-setup:
	@echo "Running one-time setup on GCE VM..."
	gcloud compute ssh $(GCE_VM_NAME) --zone $(GCE_ZONE) --project $(GCE_PROJECT) \
	    -- 'bash -s' < scripts/setup_gce.sh

gcp-deploy:
	@echo "Deploying to GCE VM..."
	bash scripts/deploy.sh $(GCE_VM_NAME) $(GCE_ZONE) $(GCE_PROJECT)

gcp-ssh:
	gcloud compute ssh $(GCE_VM_NAME) --zone $(GCE_ZONE) --project $(GCE_PROJECT)

gcp-logs:
	gcloud compute ssh $(GCE_VM_NAME) --zone $(GCE_ZONE) --project $(GCE_PROJECT) \
	    -- 'cd /opt/souli && docker compose -f docker-compose.gcp.yml logs -f --tail=100'

gcp-status:
	gcloud compute ssh $(GCE_VM_NAME) --zone $(GCE_ZONE) --project $(GCE_PROJECT) \
	    -- 'cd /opt/souli && docker compose -f docker-compose.gcp.yml ps'

# ── Run pipeline on GCP ───────────────────────────────────────────────────────
gcp-pipeline:
	@echo "Running full pipeline on GCE VM (energy + all videos)..."
	gcloud compute ssh $(GCE_VM_NAME) --zone $(GCE_ZONE) --project $(GCE_PROJECT) -- \
	    'cd /opt/souli && docker compose -f docker-compose.gcp.yml exec -T souli \
	     souli run all \
	       --config /app/configs/pipeline.gcp.yaml \
	       --videos-csv /app/data/videos.csv \
	       --excel-path "/app/data/Souli_EnergyFramework_PW (1).xlsx" \
	       --merge'

gcp-ingest:
	@echo "Ingesting YouTube chunks into Qdrant on GCE VM..."
	gcloud compute ssh $(GCE_VM_NAME) --zone $(GCE_ZONE) --project $(GCE_PROJECT) -- \
	    'cd /opt/souli && docker compose -f docker-compose.gcp.yml exec -T souli \
	     souli ingest --config /app/configs/pipeline.gcp.yaml'

gcp-chat:
	@echo "Starting chat session on GCE VM..."
	gcloud compute ssh $(GCE_VM_NAME) --zone $(GCE_ZONE) --project $(GCE_PROJECT) -- \
	    'cd /opt/souli && docker compose -f docker-compose.gcp.yml exec souli \
	     souli chat --config /app/configs/pipeline.gcp.yaml \
	       --excel "/app/data/Souli_EnergyFramework_PW (1).xlsx"'

gcp-start-vm:
	gcloud compute instances start $(GCE_VM_NAME) --zone $(GCE_ZONE) --project $(GCE_PROJECT)

gcp-stop-vm:
	@echo "Stopping VM to save costs (you won't be charged for compute while stopped)..."
	gcloud compute instances stop $(GCE_VM_NAME) --zone $(GCE_ZONE) --project $(GCE_PROJECT)
