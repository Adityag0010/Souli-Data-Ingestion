"""
souli_pipeline/retrieval/qdrant_store_multi.py

Qdrant store for the multi-collection ingestion pipeline.

Handles:
  1. Ingesting typed extractor chunks into 5 specialist collections
  2. Ingesting general cleaned chunks into souli_chunks_improved (reuses existing function)
  3. Phase-aware retrieval — returns chunks from the right collections
     based on the current conversation phase, pre-labelled for prompt injection

Collections managed here:
  souli_healing      — healing principles
  souli_activities   — practices and exercises
  souli_stories      — metaphors, stories, signature phrases
  souli_commitment   — readiness challenge questions
  souli_patterns     — problem pattern descriptions

  souli_chunks_improved — general semantic collection (existing, managed by
                          qdrant_store_improved.py, just called from here)

Payload fields stored per chunk (in addition to the 5 typed collections):
  text, chunk_type, energy_node, problem_keywords, source_video, youtube_url
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_VECTOR_SIZE   = 384

_encoder_cache: dict = {}

# Payload field names — keep consistent across ingestion and retrieval
F_TEXT        = "text"
F_CHUNK_TYPE  = "chunk_type"
F_NODE        = "energy_node"
F_KEYWORDS    = "problem_keywords"
F_SOURCE      = "source_video"
F_URL         = "youtube_url"

# ---------------------------------------------------------------------------
# Collection routing map
# chunk_type (str) → Qdrant collection name
# ---------------------------------------------------------------------------

COLLECTION_MAP: Dict[str, str] = {
    "healing":    "souli_healing",
    "activities": "souli_activities",
    "stories":    "souli_stories",
    "commitment": "souli_commitment",
    "patterns":   "souli_patterns",
    # "general" is handled separately via qdrant_store_improved — not in this map
}

# Phase → which typed collections to search
# souli_chunks_improved is always searched as a fallback by the engine itself —
# these are the TYPED collections to add on top, per phase
_PHASE_COLLECTION_MAP: Dict[str, List[str]] = {
    "intake":       ["souli_patterns"],
    "sharing":      ["souli_stories", "souli_patterns"],          # early sharing
    "sharing_late": ["souli_healing", "souli_commitment"],         # turns 5+
    "deepening":    ["souli_healing", "souli_commitment"],
    "solution":     ["souli_activities", "souli_healing"],
    "summary":      ["souli_healing"],
}

# ---------------------------------------------------------------------------
# Qdrant client + collection helpers
# (mirrors pattern in qdrant_store_improved.py — no shared state)
# ---------------------------------------------------------------------------

def _get_qdrant_client(host: str = "localhost", port: int = 6333):
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=host, port=port, timeout=5)
        client.get_collections()
        return client
    except Exception:
        logger.warning("[MULTI] Qdrant not reachable at %s:%s — using in-memory.", host, port)
        from qdrant_client import QdrantClient
        return QdrantClient(":memory:")


def _ensure_collection(client, collection: str):
    from qdrant_client.models import Distance, VectorParams
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("[MULTI] Created Qdrant collection '%s'", collection)


def _get_encoder(model_name: str = _DEFAULT_MODEL):
    if model_name not in _encoder_cache:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("[MULTI] Loading embedding model: %s on %s", model_name, device)
        _encoder_cache[model_name] = SentenceTransformer(model_name, device=device)
    return _encoder_cache[model_name]


def _embed(texts: List[str], model_name: str = _DEFAULT_MODEL) -> List[List[float]]:
    return _get_encoder(model_name).encode(
        texts, convert_to_numpy=True, show_progress_bar=False
    ).tolist()


def _content_uuid(text: str, source: str, chunk_type: str) -> str:
    """
    Deterministic UUID from content + source + chunk_type.
    Prevents duplicates across re-ingestion AND across different collections
    for the same text (same story shouldn't collide with healing chunk).
    """
    key = f"{chunk_type}::{source}::{text[:400]}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


# ---------------------------------------------------------------------------
# Single-collection typed ingest
# ---------------------------------------------------------------------------

def ingest_typed_chunks(
    chunks: List[Dict],
    chunk_type: str,
    embedding_model: str = _DEFAULT_MODEL,
    host: str = "localhost",
    port: int = 6333,
    batch_size: int = 32,
) -> int:
    """
    Ingest a list of extractor output dicts into the correct typed collection.

    Args:
        chunks:          Output of any extractor function — list of dicts with
                         keys: text, chunk_type, energy_node, problem_keywords,
                               source_video, youtube_url
        chunk_type:      One of: healing, activities, stories, commitment, patterns
        embedding_model: Sentence transformer model name
        host, port:      Qdrant server connection

    Returns:
        Number of points successfully ingested.
    """
    from qdrant_client.models import PointStruct

    if chunk_type not in COLLECTION_MAP:
        logger.error("[MULTI] Unknown chunk_type '%s' — skipping ingest.", chunk_type)
        return 0

    collection = COLLECTION_MAP[chunk_type]

    # Filter out empty texts
    valid = [c for c in chunks if c.get("text", "").strip()]
    if not valid:
        logger.info("[MULTI] No valid chunks for type '%s' — nothing to ingest.", chunk_type)
        return 0

    # Drop exact text duplicates within this batch
    seen_texts: set = set()
    deduped = []
    for c in valid:
        t = c["text"].strip()
        if t not in seen_texts:
            seen_texts.add(t)
            deduped.append(c)

    if len(deduped) < len(valid):
        logger.info("[MULTI] Dropped %d duplicate chunks for type '%s'.", len(valid) - len(deduped), chunk_type)

    client = _get_qdrant_client(host, port)
    _ensure_collection(client, collection)

    total    = len(deduped)
    ingested = 0

    for batch_start in range(0, total, batch_size):
        batch = deduped[batch_start : batch_start + batch_size]
        texts   = [c["text"] for c in batch]
        vectors = _embed(texts, embedding_model)

        points = []
        for vec, chunk in zip(vectors, batch):
            source = chunk.get("source_video", "")
            payload: Dict[str, Any] = {
                F_TEXT:       chunk.get("text", ""),
                F_CHUNK_TYPE: chunk_type,
                F_NODE:       chunk.get("energy_node", ""),
                F_KEYWORDS:   chunk.get("problem_keywords", ""),
                F_SOURCE:     source,
                F_URL:        chunk.get("youtube_url", ""),
            }
            point_id = _content_uuid(payload[F_TEXT], source, chunk_type)
            points.append(PointStruct(id=point_id, vector=vec, payload=payload))

        client.upsert(collection_name=collection, points=points)
        ingested += len(points)
        logger.info(
            "[MULTI:%s] Ingested batch %d-%d / %d into '%s'",
            chunk_type, batch_start + 1, batch_start + len(batch), total, collection,
        )

    logger.info("[MULTI:%s] Total ingested: %d into '%s'", chunk_type, ingested, collection)
    return ingested


# ---------------------------------------------------------------------------
# Ingest all extractor outputs — the main entry point from the pipeline
# ---------------------------------------------------------------------------

def ingest_all_extractor_outputs(
    extractor_outputs: Dict[str, List[Dict]],
    embedding_model: str = _DEFAULT_MODEL,
    host: str = "localhost",
    port: int = 6333,
) -> Dict[str, int]:
    """
    Ingest all 5 extractor outputs into their respective typed collections.
    Also ingests the general collection (souli_chunks_improved) if df_cleaned is provided.

    Args:
        extractor_outputs: Dict of chunk_type → list of chunk dicts.
                           Output of run_extractors_from_density().
                           Keys: healing, activities, stories, commitment, patterns

    Returns:
        Dict of chunk_type → count of ingested chunks
        e.g. {"healing": 12, "activities": 0, "stories": 8, "commitment": 5, "patterns": 11}
    """
    counts: Dict[str, int] = {}

    for chunk_type, chunks in extractor_outputs.items():
        if chunk_type not in COLLECTION_MAP:
            logger.warning("[MULTI] Unknown chunk_type '%s' in extractor outputs — skipping.", chunk_type)
            continue

        count = ingest_typed_chunks(
            chunks=chunks,
            chunk_type=chunk_type,
            embedding_model=embedding_model,
            host=host,
            port=port,
        )
        counts[chunk_type] = count

    # Summary log
    total = sum(counts.values())
    logger.info("[MULTI] Typed ingest complete. Counts: %s | Total: %d", counts, total)
    return counts


# ---------------------------------------------------------------------------
# Phase-aware retrieval — returns chunks labelled by type for prompt injection
# ---------------------------------------------------------------------------

def _get_collections_for_phase(phase: str, turn_count: int = 0) -> List[str]:
    """
    Return the list of typed collections to search for a given conversation phase.

    The 'sharing' phase is split: early turns use stories+patterns,
    later turns (5+) use healing+commitment to start moving forward.
    """
    if phase == "sharing" and turn_count >= 5:
        return _PHASE_COLLECTION_MAP.get("sharing_late", [])
    return _PHASE_COLLECTION_MAP.get(phase, _PHASE_COLLECTION_MAP["sharing"])


def query_by_phase(
    user_text: str,
    phase: str,
    energy_node: Optional[str] = None,
    turn_count: int = 0,
    top_k: int = 2,
    embedding_model: str = _DEFAULT_MODEL,
    host: str = "localhost",
    port: int = 6333,
    score_threshold: float = 0.25,
) -> List[Dict]:
    """
    Retrieve chunks from the typed collections appropriate for the current phase.

    Each returned chunk includes a 'chunk_type' field so the counselor prompt
    can inject it with the right label (e.g. [HEALING PRINCIPLE], [ACTIVITY]).

    Args:
        user_text:    Current user message (used as query vector)
        phase:        Current conversation phase (intake/sharing/deepening/solution/summary)
        energy_node:  Detected energy node (used as Qdrant filter)
        turn_count:   Current turn number (used to split early/late sharing)
        top_k:        How many chunks to retrieve PER collection
        embedding_model, host, port, score_threshold: standard Qdrant params

    Returns:
        List of chunk dicts, each with keys:
            text, chunk_type, energy_node, problem_keywords, source_video, score
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    if not user_text or not user_text.strip():
        return []

    collections_to_search = _get_collections_for_phase(phase, turn_count)
    if not collections_to_search:
        logger.info("[MULTI:retrieval] No typed collections for phase '%s'.", phase)
        return []

    client   = _get_qdrant_client(host, port)
    existing = {c.name for c in client.get_collections().collections}

    query_vec = _embed([user_text], embedding_model)[0]

    results: List[Dict] = []

    for collection in collections_to_search:
        if collection not in existing:
            logger.debug("[MULTI:retrieval] Collection '%s' not found — skipping.", collection)
            continue

        query_filter = None
        if energy_node:
            query_filter = Filter(
                must=[FieldCondition(key=F_NODE, match=MatchValue(value=energy_node))]
            )

        try:
            hits = client.query_points(
                collection_name=collection,
                query=query_vec,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            ).points
        except AttributeError:
            # Older Qdrant client API
            hits = client.search(
                collection_name=collection,
                query_vector=query_vec,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )

        for hit in hits:
            p = hit.payload or {}
            results.append({
                "text":             p.get(F_TEXT, ""),
                "chunk_type":       p.get(F_CHUNK_TYPE, ""),
                "energy_node":      p.get(F_NODE, ""),
                "problem_keywords": p.get(F_KEYWORDS, ""),
                "source_video":     p.get(F_SOURCE, ""),
                "score":            round(hit.score, 4),
            })

    # Sort by score descending across all collections
    results.sort(key=lambda x: x["score"], reverse=True)

    logger.info(
        "[MULTI:retrieval] phase=%s turn=%d → searched %d collections, returned %d chunks",
        phase, turn_count, len(collections_to_search), len(results),
    )
    return results


# ---------------------------------------------------------------------------
# Utility — list all multi collections and their point counts
# ---------------------------------------------------------------------------

def get_collection_stats(host: str = "localhost", port: int = 6333) -> Dict[str, int]:
    """
    Return point counts for all typed collections.
    Useful for verifying ingestion was successful.
    """
    client = _get_qdrant_client(host, port)
    existing = {c.name for c in client.get_collections().collections}
    stats: Dict[str, int] = {}
    for chunk_type, collection in COLLECTION_MAP.items():
        if collection in existing:
            info = client.get_collection(collection)
            stats[collection] = info.points_count or 0
        else:
            stats[collection] = 0
    return stats