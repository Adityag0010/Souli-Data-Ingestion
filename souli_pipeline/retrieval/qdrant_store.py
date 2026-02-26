"""
Qdrant vector store for Souli YouTube teaching chunks.

Ingests tagged teaching chunks and supports energy-node-filtered retrieval.
All processing is local — no data leaves the machine.

Requirements:
    pip install qdrant-client sentence-transformers
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Qdrant payload field names
F_TEXT = "text"
F_NODE = "energy_node"
F_REASON = "energy_node_reason"
F_SOURCE = "source_video"
F_URL = "youtube_url"
F_CHUNK_TYPE = "chunk_type"
F_START = "start"
F_END = "end"

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output dim


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def _get_qdrant_client(host: str = "localhost", port: int = 6333):
    """Return a Qdrant client. Falls back to in-memory if server not reachable."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=host, port=port, timeout=5)
        # Ping the server
        client.get_collections()
        logger.info("Connected to Qdrant at %s:%s", host, port)
        return client
    except Exception:
        logger.warning(
            "Qdrant server not reachable at %s:%s — using in-memory mode.",
            host,
            port,
        )
        from qdrant_client import QdrantClient
        return QdrantClient(":memory:")


# ---------------------------------------------------------------------------
# Ensure collection exists
# ---------------------------------------------------------------------------

def ensure_collection(
    client,
    collection: str = "souli_chunks",
    vector_size: int = _VECTOR_SIZE,
):
    from qdrant_client.models import Distance, VectorParams

    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s'", collection)
    else:
        logger.debug("Collection '%s' already exists.", collection)


# ---------------------------------------------------------------------------
# Embed helper
# ---------------------------------------------------------------------------

def _embed_texts(texts: List[str], model_name: str = _DEFAULT_MODEL) -> List[List[float]]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def ingest_dataframe(
    df: pd.DataFrame,
    collection: str = "souli_chunks",
    embedding_model: str = _DEFAULT_MODEL,
    host: str = "localhost",
    port: int = 6333,
    batch_size: int = 64,
) -> int:
    """
    Embed rows from a teaching_ready DataFrame and upsert into Qdrant.

    Required columns: text
    Optional columns: energy_node, energy_node_reason, source_video,
                      youtube_url, chunk_type, start, end

    Returns the number of points ingested.
    """
    from qdrant_client.models import PointStruct

    if df.empty:
        logger.warning("Empty DataFrame — nothing to ingest.")
        return 0

    df = df[df["text"].notna() & (df["text"].str.strip() != "")].copy()
    if df.empty:
        return 0

    client = _get_qdrant_client(host, port)
    ensure_collection(client, collection)

    texts = df["text"].astype(str).tolist()
    total = len(texts)
    ingested = 0

    for batch_start in range(0, total, batch_size):
        batch_texts = texts[batch_start: batch_start + batch_size]
        batch_rows = df.iloc[batch_start: batch_start + batch_size]

        vectors = _embed_texts(batch_texts, embedding_model)

        points = []
        for i, (vec, (_, row)) in enumerate(zip(vectors, batch_rows.iterrows())):
            payload: Dict[str, Any] = {
                F_TEXT: str(row.get("text", "")),
                F_NODE: str(row.get("energy_node", "")),
                F_REASON: str(row.get("energy_node_reason", "")),
                F_SOURCE: str(row.get("source_video", "")),
                F_URL: str(row.get("youtube_url", "")),
                F_CHUNK_TYPE: str(row.get("chunk_type", "teaching")),
                F_START: float(row.get("start", 0.0)),
                F_END: float(row.get("end", 0.0)),
            }
            points.append(
                PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload)
            )

        client.upsert(collection_name=collection, points=points)
        ingested += len(points)
        logger.info(
            "Ingested batch %d-%d / %d",
            batch_start + 1,
            batch_start + len(batch_texts),
            total,
        )

    logger.info("Total ingested into '%s': %d points", collection, ingested)
    return ingested


def ingest_from_excel(
    path: str,
    collection: str = "souli_chunks",
    embedding_model: str = _DEFAULT_MODEL,
    host: str = "localhost",
    port: int = 6333,
) -> int:
    """Convenience wrapper: load teaching_ready.xlsx and ingest."""
    df = pd.read_excel(path)
    return ingest_dataframe(df, collection=collection, embedding_model=embedding_model, host=host, port=port)


# ---------------------------------------------------------------------------
# Query / Retrieval
# ---------------------------------------------------------------------------

def query_chunks(
    user_text: str,
    collection: str = "souli_chunks",
    energy_node: Optional[str] = None,
    top_k: int = 3,
    embedding_model: str = _DEFAULT_MODEL,
    host: str = "localhost",
    port: int = 6333,
    score_threshold: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k chunks from Qdrant similar to user_text.
    Optionally filter by energy_node.

    Returns list of dicts with keys: text, energy_node, reason, source_video,
    youtube_url, score.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    if not user_text or not user_text.strip():
        return []

    client = _get_qdrant_client(host, port)

    # Check collection exists
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        logger.warning("Collection '%s' not found — run `souli ingest` first.", collection)
        return []

    query_vec = _embed_texts([user_text], embedding_model)[0]

    query_filter = None
    if energy_node:
        query_filter = Filter(
            must=[FieldCondition(key=F_NODE, match=MatchValue(value=energy_node))]
        )

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
        out.append(
            {
                "text": p.get(F_TEXT, ""),
                "energy_node": p.get(F_NODE, ""),
                "reason": p.get(F_REASON, ""),
                "source_video": p.get(F_SOURCE, ""),
                "youtube_url": p.get(F_URL, ""),
                "score": round(r.score, 4),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Ingest from full pipeline output directory
# ---------------------------------------------------------------------------

def ingest_pipeline_outputs(
    outputs_dir: str,
    collection: str = "souli_chunks",
    embedding_model: str = _DEFAULT_MODEL,
    host: str = "localhost",
    port: int = 6333,
) -> int:
    """
    Walk an outputs directory tree and ingest every teaching_ready.xlsx found.
    Use after running: souli run videos --config ... --videos-csv ...
    """
    import os

    total = 0
    for root, _dirs, files in os.walk(outputs_dir):
        for fname in files:
            if fname == "teaching_ready.xlsx":
                path = os.path.join(root, fname)
                logger.info("Ingesting: %s", path)
                try:
                    n = ingest_from_excel(path, collection=collection, embedding_model=embedding_model, host=host, port=port)
                    total += n
                except Exception as exc:
                    logger.warning("Failed to ingest %s: %s", path, exc)
    logger.info("Grand total ingested: %d", total)
    return total
