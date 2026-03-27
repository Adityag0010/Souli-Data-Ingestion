"""
souli_pipeline/retrieval/qdrant_store_improved.py

Qdrant ingest for the improved pipeline.
Uses the cleaned_chunks.xlsx output (field: "cleaned_text").

Key differences from qdrant_store.py:
  - Uses "cleaned_text" column, not "text"
  - Content-hash UUID (no duplicate ingestion)
  - Stores both cleaned_text AND original_text in payload for debugging
  - No quality gate needed here — segment_cleaner already guarantees clean input
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_VECTOR_SIZE   = 384

_encoder_cache: dict = {}

# Payload field names
F_TEXT         = "text"           # the cleaned text (used for retrieval display)
F_ORIGINAL     = "original_text"  # raw transcript before cleaning (for debugging)
F_NODE         = "energy_node"
F_SOURCE       = "source_video"
F_URL          = "youtube_url"
F_TOPIC_INDEX  = "topic_index"
F_START        = "start"
F_END          = "end"


def _content_uuid(text: str, source: str) -> str:
    """Deterministic UUID from content + source. Re-ingesting same video is idempotent."""
    key = f"{source}::{text[:400]}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


def _get_qdrant_client(host: str = "localhost", port: int = 6333):
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=host, port=port, timeout=5)
        client.get_collections()
        return client
    except Exception:
        logger.warning("Qdrant not reachable at %s:%s — using in-memory.", host, port)
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
        logger.info("Created Qdrant collection '%s'", collection)


def _get_encoder(model_name: str = _DEFAULT_MODEL):
    if model_name not in _encoder_cache:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", model_name)
        _encoder_cache[model_name] = SentenceTransformer(model_name)
    return _encoder_cache[model_name]


def _embed(texts: List[str], model_name: str) -> List[List[float]]:
    return _get_encoder(model_name).encode(
        texts, convert_to_numpy=True, show_progress_bar=False
    ).tolist()


def ingest_improved_chunks(
    df: pd.DataFrame,
    collection: str = "souli_chunks_improved",
    embedding_model: str = _DEFAULT_MODEL,
    host: str = "localhost",
    port: int = 6333,
    batch_size: int = 32,
) -> int:
    """
    Embed and upsert cleaned chunks into the improved Qdrant collection.

    Required column: cleaned_text
    Optional columns: original_text, energy_node, source_video,
                      youtube_url, topic_index, start, end

    Returns number of points ingested.
    """
    from qdrant_client.models import PointStruct

    if df.empty:
        logger.warning("Empty DataFrame — nothing to ingest.")
        return 0

    # Use cleaned_text as the embedding source
    text_col = "cleaned_text" if "cleaned_text" in df.columns else "text"
    df = df[df[text_col].notna() & (df[text_col].str.strip() != "")].copy()
    if df.empty:
        return 0

    # Drop exact-text duplicates
    before = len(df)
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    if len(df) < before:
        logger.info("Dropped %d exact duplicates.", before - len(df))

    client = _get_qdrant_client(host, port)
    _ensure_collection(client, collection)

    texts    = df[text_col].astype(str).tolist()
    total    = len(texts)
    ingested = 0

    for batch_start in range(0, total, batch_size):
        batch_texts = texts[batch_start : batch_start + batch_size]
        batch_rows  = df.iloc[batch_start : batch_start + batch_size]

        vectors = _embed(batch_texts, embedding_model)
        points  = []

        for vec, (_, row) in zip(vectors, batch_rows.iterrows()):
            cleaned_text  = str(row.get(text_col, ""))
            original_text = str(row.get("original_text", ""))
            source        = str(row.get("source_video", ""))

            payload: Dict[str, Any] = {
                F_TEXT        : cleaned_text,
                F_ORIGINAL    : original_text[:500] if original_text else "",  # cap for storage
                F_NODE        : str(row.get("energy_node", "")),
                F_SOURCE      : source,
                F_URL         : str(row.get("youtube_url", "")),
                F_TOPIC_INDEX : int(row.get("topic_index", 0)),
                F_START       : float(row.get("start", 0.0)),
                F_END         : float(row.get("end", 0.0)),
            }

            point_id = _content_uuid(cleaned_text, source)
            points.append(PointStruct(id=point_id, vector=vec, payload=payload))

        client.upsert(collection_name=collection, points=points)
        ingested += len(points)
        logger.info(
            "Ingested batch %d-%d / %d",
            batch_start + 1, batch_start + len(batch_texts), total,
        )

    logger.info("Total ingested into '%s': %d points", collection, ingested)
    return ingested


def query_improved_chunks(
    user_text: str,
    collection: str = "souli_chunks_improved",
    energy_node: Optional[str] = None,
    top_k: int = 3,
    embedding_model: str = _DEFAULT_MODEL,
    host: str = "localhost",
    port: int = 6333,
    score_threshold: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k chunks from the improved collection.
    API-compatible with qdrant_store.query_chunks().
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    if not user_text or not user_text.strip():
        return []

    client = _get_qdrant_client(host, port)
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        logger.warning("Collection '%s' not found.", collection)
        return []

    query_vec    = _embed([user_text], embedding_model)[0]
    query_filter = None
    if energy_node:
        query_filter = Filter(
            must=[FieldCondition(key=F_NODE, match=MatchValue(value=energy_node))]
        )

    try:
        # Try newer API first
        results = client.query_points(
            collection_name=collection,
            query=query_vec,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        ).points
    except AttributeError:
        results = client.search(
            collection_name=collection,
            query_vector=query_vec,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )

    out = []
    for r in results:
        p = r.payload or {}
        out.append({
            "text"        : p.get(F_TEXT, ""),
            "energy_node" : p.get(F_NODE, ""),
            "source_video": p.get(F_SOURCE, ""),
            "youtube_url" : p.get(F_URL, ""),
            "topic_index" : p.get(F_TOPIC_INDEX, 0),
            "score"       : round(r.score, 4),
        })
    return out