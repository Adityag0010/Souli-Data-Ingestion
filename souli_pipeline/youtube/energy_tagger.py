"""
Energy node tagger for YouTube transcript chunks.

Uses a small Qwen model (via Ollama) to classify each teaching chunk into one
of the 5 Souli energy nodes and explain WHY that node applies.

All inference is local — no data sent to any external API.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

NODES = [
    "blocked_energy",
    "depleted_energy",
    "scattered_energy",
    "outofcontrol_energy",
    "normal_energy",
]

# Short descriptions fed to Qwen so it understands each node
NODE_DESCRIPTIONS = {
    "blocked_energy": (
        "Withdrawal, depression, procrastination, self-harm risk, emotional numbness, "
        "stuck in toxic cycles, life disconnection."
    ),
    "depleted_energy": (
        "Internally tired, lazy, low energy, fear of failure, feeling victimized, "
        "nobody values me, not completing things."
    ),
    "scattered_energy": (
        "Overwhelmed by external factors, stressed, burnout, overworked, anxious, "
        "energy all over the place, no sense of accomplishment."
    ),
    "outofcontrol_energy": (
        "Anger issues, physically restless, non-stop mind, emotional extremes, "
        "judgmental, impulsive, cannot control energy on any level."
    ),
    "normal_energy": (
        "Comfortable in life, looking for inner/outer growth, spiritual progression, "
        "happy, doing what they want to do."
    ),
}

_SYSTEM_PROMPT = """\
You are an inner-energy wellness analyst working with the Souli framework.
Your job is to read a counseling transcript chunk and decide which energy state it addresses.

The 5 energy nodes are:
{node_descriptions}

Respond with ONLY valid JSON — no explanation outside the JSON block.
"""

_USER_PROMPT = """\
Transcript chunk:
\"\"\"
{text}
\"\"\"

Return JSON with exactly these keys:
- "energy_node": one of {nodes}
- "reason": one sentence explaining why this chunk maps to that node (max 25 words)
"""


def _build_system() -> str:
    desc_lines = "\n".join(
        f"- {node}: {desc}" for node, desc in NODE_DESCRIPTIONS.items()
    )
    return _SYSTEM_PROMPT.format(node_descriptions=desc_lines)


def tag_chunk(
    text: str,
    ollama_model: str = "qwen2.5:1.5b",
    ollama_endpoint: str = "http://localhost:11434",
    timeout_s: int = 60,
) -> Dict[str, str]:
    """
    Tag a single chunk with energy_node + reason using Qwen via Ollama.
    Returns {"energy_node": "...", "reason": "..."}.
    Falls back to keyword heuristic if Ollama is unavailable.
    """
    from ..llm.ollama import OllamaLLM
    from ..energy.normalize import infer_node

    text = (text or "").strip()
    if not text:
        return {"energy_node": "blocked_energy", "reason": "Empty chunk."}

    llm = OllamaLLM(
        model=ollama_model,
        endpoint=ollama_endpoint,
        timeout_s=timeout_s,
        temperature=0.1,
        num_ctx=2048,
    )

    if not llm.is_available():
        logger.warning("Ollama not available — using keyword fallback for energy tagging.")
        node = infer_node(text, "")
        return {"energy_node": node, "reason": "keyword_fallback"}

    prompt = _USER_PROMPT.format(
        text=text[:1200],  # cap input to keep it fast
        nodes=json.dumps(NODES),
    )

    try:
        raw = llm.generate(
            prompt=prompt,
            system=_build_system(),
            temperature=0.1,
            format="json",
        )
        data = _parse_json(raw)
        node = str(data.get("energy_node", "")).strip().lower()
        reason = str(data.get("reason", "")).strip()
        if node not in NODES:
            # Qwen gave something off — normalize it
            from ..energy.normalize import normalize_node
            node = normalize_node(node, NODES) or infer_node(text, "")
        return {"energy_node": node, "reason": reason or "No reason provided."}
    except Exception as exc:
        logger.warning("Qwen tagging failed (%s) — keyword fallback.", exc)
        node = infer_node(text, "")
        return {"energy_node": node, "reason": "keyword_fallback"}


def tag_dataframe(
    df,
    text_col: str = "text",
    ollama_model: str = "qwen2.5:1.5b",
    ollama_endpoint: str = "http://localhost:11434",
    timeout_s: int = 60,
    log_every: int = 10,
):
    """
    Add 'energy_node' and 'energy_node_reason' columns to a DataFrame.
    Processes only rows where text_col has content.
    Returns the modified DataFrame (in-place modification).
    """
    import pandas as pd

    nodes_out: List[str] = []
    reasons_out: List[str] = []

    total = len(df)
    for i, row in enumerate(df.itertuples(index=False), 1):
        text = str(getattr(row, text_col, "") or "").strip()
        if not text:
            nodes_out.append("")
            reasons_out.append("")
            continue

        result = tag_chunk(
            text,
            ollama_model=ollama_model,
            ollama_endpoint=ollama_endpoint,
            timeout_s=timeout_s,
        )
        nodes_out.append(result["energy_node"])
        reasons_out.append(result["reason"])

        if i % log_every == 0 or i == total:
            logger.info("Tagged %d/%d chunks", i, total)

    df = df.copy()
    df["energy_node"] = nodes_out
    df["energy_node_reason"] = reasons_out
    return df


def _parse_json(raw: str) -> Dict:
    """Parse JSON from Qwen output, tolerating minor noise."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
    return {}
