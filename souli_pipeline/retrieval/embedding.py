"""
Local embedding only. Uses sentence-transformers (open-source, runs on your machine).
No data is sent to any external API — nothing is used for training.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

_encoders: Dict[str, Any] = {}
_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _get_encoder(model_name: str = _DEFAULT_MODEL):
    if model_name in _encoders and _encoders[model_name] is not None:
        return _encoders[model_name]
    try:
        from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
        enc = SentenceTransformer(model_name)
        _encoders[model_name] = enc
        return enc
    except Exception:
        return None


def embed(texts: List[str], model_name: str = _DEFAULT_MODEL) -> Optional[List[List[float]]]:
    """
    Embed texts locally. Returns None if sentence_transformers not installed.
    No data leaves the machine.
    """
    enc = _get_encoder(model_name)
    if enc is None:
        return None
    return enc.encode(texts, convert_to_numpy=True).tolist()


def embed_one(text: str, model_name: str = _DEFAULT_MODEL) -> Optional[List[float]]:
    """Embed a single string. Returns None if encoder unavailable."""
    result = embed([text], model_name=model_name)
    return result[0] if result else None


def available() -> bool:
    """True if local embedding is available (sentence_transformers installed)."""
    return _get_encoder() is not None
