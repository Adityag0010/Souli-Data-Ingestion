"""
Config loader with environment variable override support.

Env vars override YAML values so the same configs/pipeline.yaml works both
locally and on GCP — just change the env vars, not the file.

Override keys (all optional):
  OLLAMA_ENDPOINT          e.g. http://ollama:11434
  OLLAMA_CHAT_MODEL        e.g. llama3.1
  OLLAMA_TAGGER_MODEL      e.g. qwen2.5:1.5b
  QDRANT_HOST              e.g. qdrant
  QDRANT_PORT              e.g. 6333
  QDRANT_COLLECTION        e.g. souli_chunks
  LIVEKIT_URL              e.g. wss://your-project.livekit.cloud
  LIVEKIT_API_KEY
  LIVEKIT_API_SECRET
  LIVEKIT_ROOM
  GCS_BUCKET               e.g. souli-data  (used by pipeline for outputs)
  SOULI_OUTPUTS_DIR        e.g. /app/outputs or gs://souli-data/outputs
"""
from __future__ import annotations

import os
import yaml
from .config import PipelineConfig


def _apply_env_overrides(raw: dict) -> dict:
    """Patch raw YAML dict with environment variable values where set."""

    def env(key: str, default=None):
        return os.environ.get(key, default)

    # Ollama endpoint (conversation + llm)
    ollama_ep = env("OLLAMA_ENDPOINT")
    if ollama_ep:
        raw.setdefault("conversation", {})["ollama_endpoint"] = ollama_ep
        raw.setdefault("llm", {}).setdefault("ollama", {})["endpoint"] = ollama_ep

    chat_model = env("OLLAMA_CHAT_MODEL")
    if chat_model:
        raw.setdefault("conversation", {})["chat_model"] = chat_model

    tagger_model = env("OLLAMA_TAGGER_MODEL")
    if tagger_model:
        raw.setdefault("conversation", {})["tagger_model"] = tagger_model

    # Qdrant
    qdrant_host = env("QDRANT_HOST")
    if qdrant_host:
        raw.setdefault("retrieval", {})["qdrant_host"] = qdrant_host

    qdrant_port = env("QDRANT_PORT")
    if qdrant_port:
        raw.setdefault("retrieval", {})["qdrant_port"] = int(qdrant_port)

    qdrant_coll = env("QDRANT_COLLECTION")
    if qdrant_coll:
        raw.setdefault("retrieval", {})["qdrant_collection"] = qdrant_coll

    # LiveKit
    lk_url = env("LIVEKIT_URL")
    if lk_url:
        raw.setdefault("voice", {})["livekit_url"] = lk_url

    lk_key = env("LIVEKIT_API_KEY")
    if lk_key:
        raw.setdefault("voice", {})["livekit_api_key"] = lk_key

    lk_secret = env("LIVEKIT_API_SECRET")
    if lk_secret:
        raw.setdefault("voice", {})["livekit_api_secret"] = lk_secret

    lk_room = env("LIVEKIT_ROOM")
    if lk_room:
        raw.setdefault("voice", {})["room_name"] = lk_room

    # Outputs dir (GCS path or local)
    outputs_dir = env("SOULI_OUTPUTS_DIR")
    if outputs_dir:
        raw.setdefault("run", {})["outputs_dir"] = outputs_dir

    return raw


def load_config(path: str) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw = _apply_env_overrides(raw)
    return PipelineConfig.model_validate(raw)
