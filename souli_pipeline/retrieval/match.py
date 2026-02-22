"""
Match user venting/problem text to:
1) Diagnosis: energy_node + aspect + closest problem from gold
2) Framework solution (practices, meditations, etc.) for that node
3) Teaching content from YouTube (teaching cards) for that node

All processing is local. No data is sent to any external API.
"""
from __future__ import annotations
import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional

from ..energy.normalize import infer_node
from .embedding import embed, embed_one, available as embedding_available

# Column names used in pipeline outputs
PROB_COL = "Problem statement"
ASPECT_COL = "Aspects of Woman Track"
NODE_COL = "energy_node/energy block behind it/ inner block"
BLOCKS_COL = "deeper_blocks/ pshychlogical issues"
DUAL_COL = "Duality Check"
# Teaching cards
CARD_NODE_COL = "Mapped energy_node"
CARD_CONCEPT = "Concept/Principle"
CARD_EXPLANATION = "Core explanation"
CARD_APPLIES = "When it applies"
CARD_EXAMPLE = "Concrete example (1-2 lines)"


def _cosine_sim(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def load_gold(gold_path: str, nodes_allowed: List[str]) -> pd.DataFrame:
    """Load gold.xlsx and ensure required columns exist."""
    df = pd.read_excel(gold_path)
    df.columns = [str(c).strip() for c in df.columns]
    if NODE_COL not in df.columns:
        raise ValueError(f"Gold must have column '{NODE_COL}'")
    df[NODE_COL] = df[NODE_COL].astype(str).str.strip().str.lower()
    df = df[df[NODE_COL].isin(nodes_allowed)].copy()
    return df


def load_teaching_cards(path_or_dir: str) -> pd.DataFrame:
    """
    Load teaching cards from a single xlsx file or from a directory
    (merged_teaching_cards.xlsx or teaching_cards.xlsx).
    """
    if os.path.isfile(path_or_dir):
        df = pd.read_excel(path_or_dir)
    elif os.path.isdir(path_or_dir):
        merged = os.path.join(path_or_dir, "merged_teaching_cards.xlsx")
        if os.path.isfile(merged):
            df = pd.read_excel(merged)
        else:
            # Collect all teaching_cards.xlsx in subdirs
            frames = []
            for root, _dirs, files in os.walk(path_or_dir):
                if "teaching_cards.xlsx" in files:
                    frames.append(pd.read_excel(os.path.join(root, "teaching_cards.xlsx")))
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        raise FileNotFoundError(path_or_dir)
    if df.empty:
        return df
    df.columns = [str(c).strip() for c in df.columns]
    # Normalize node column name
    if "Mapped energy_node" in df.columns and "energy_node" not in df.columns:
        df["energy_node"] = df["Mapped energy_node"].astype(str).str.strip().str.lower()
    elif "energy_node" not in df.columns and NODE_COL in df.columns:
        df["energy_node"] = df[NODE_COL].astype(str).str.strip().str.lower()
    return df


def diagnose(
    user_text: str,
    gold_df: pd.DataFrame,
    nodes_allowed: List[str],
    embedding_model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Diagnose user input: infer energy_node and optionally match to closest problem in gold.
    Uses local embedding if available, else keyword-based infer_node. No external API.
    """
    user_text = (user_text or "").strip()
    if not user_text:
        return {
            "energy_node": "blocked_energy",
            "aspect": "Unknown",
            "matched_problem": None,
            "confidence": "keyword_fallback",
            "framework_row": None,
        }

    # 1) Try embedding-based match to gold problem statements
    if embedding_model and embedding_available():
        try:
            q_emb = embed_one(user_text, model_name=embedding_model)
            if q_emb is not None and PROB_COL in gold_df.columns:
                problems = gold_df[PROB_COL].fillna("").astype(str).tolist()
                p_embs = embed(problems, model_name=embedding_model)
                if p_embs and len(p_embs) == len(problems):
                    scores = [_cosine_sim(q_emb, p) for p in p_embs]
                    best_idx = max(range(len(scores)), key=lambda i: scores[i])
                    if scores[best_idx] > 0.3:
                        row = gold_df.iloc[best_idx]
                        fw_row = _framework_row_from_gold_row(row)
                        return {
                            "energy_node": str(row.get(NODE_COL, "")).strip().lower(),
                            "aspect": str(row.get(ASPECT_COL, "Unknown")).strip(),
                            "matched_problem": str(row.get(PROB_COL, "")).strip(),
                            "confidence": "embedding_match",
                            "similarity": round(scores[best_idx], 4),
                            "framework_row": fw_row,
                        }
        except Exception:
            pass

    # 2) Keyword-based fallback: infer energy_node from user text
    node = infer_node(user_text, "")
    if not node or node not in nodes_allowed:
        node = "blocked_energy"

    # Get first gold row for this node to pull framework solution
    subset = gold_df[gold_df[NODE_COL] == node]
    if not subset.empty:
        row = subset.iloc[0]
        fw_row = _framework_row_from_gold_row(row)
        return {
            "energy_node": node,
            "aspect": str(row.get(ASPECT_COL, "Unknown")).strip(),
            "matched_problem": str(row.get(PROB_COL, "")).strip() or None,
            "confidence": "keyword_fallback",
            "framework_row": fw_row,
        }

    return {
        "energy_node": node,
        "aspect": "Unknown",
        "matched_problem": None,
        "confidence": "keyword_fallback",
        "framework_row": None,
    }


def _framework_row_from_gold_row(row: pd.Series) -> Dict[str, Any]:
    """Extract framework-style fields from an enriched gold row."""
    fw_cols = [
        "typical_signs",
        "primary_healing_principles",
        "primary_practices ( 7 min quick relief)",
        "deeper_meditations_program ( 7 day quick recovery)",
        "longer_program ( 7 month resilience building)",
        "Caution",
        "Health",
    ]
    return {c: str(row.get(c, "")).strip() for c in fw_cols if c in row.index}


def get_teaching_for_node(
    cards_df: pd.DataFrame,
    energy_node: str,
    user_text: Optional[str] = None,
    embedding_model: Optional[str] = None,
    max_items: int = 5,
) -> List[Dict[str, Any]]:
    """Get teaching content (cards) for the given energy_node. Optionally rank by similarity to user_text."""
    if cards_df.empty or "energy_node" not in cards_df.columns:
        return []
    energy_node = (energy_node or "").strip().lower()
    subset = cards_df[cards_df["energy_node"].astype(str).str.strip().str.lower() == energy_node].copy()
    if subset.empty:
        return []

    out_cols = [CARD_CONCEPT, CARD_EXPLANATION, CARD_APPLIES, CARD_EXAMPLE]
    out_cols = [c for c in out_cols if c in subset.columns]
    if not out_cols:
        out_cols = [c for c in subset.columns if not str(c).startswith("_")]

    # Optionally rank by similarity to user text
    if user_text and embedding_model and embedding_available() and CARD_EXPLANATION in subset.columns:
        try:
            q_emb = embed_one(user_text, model_name=embedding_model)
            if q_emb is not None:
                texts = subset[CARD_EXPLANATION].fillna("").astype(str).tolist()
                t_embs = embed(texts, model_name=embedding_model)
                if t_embs:
                    subset = subset.copy()
                    subset["_sim"] = [_cosine_sim(q_emb, t) for t in t_embs]
                    subset = subset.sort_values("_sim", ascending=False)
        except Exception:
            pass

    rows = subset.head(max_items)
    result = []
    for _, r in rows.iterrows():
        item = {c: str(r.get(c, "")).strip() for c in out_cols}
        if "source_video" in r.index and pd.notna(r.get("source_video")):
            item["source_video"] = str(r["source_video"]).strip()
        result.append(item)
    return result


def run_match(
    user_query: str,
    gold_path: str,
    nodes_allowed: List[str],
    teaching_path: Optional[str] = None,
    embedding_model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
    top_k_teaching: int = 5,
) -> Dict[str, Any]:
    """
    Full flow: diagnose user query → get framework solution + teaching content.
    All local. No data sent to any API.
    """
    gold_df = load_gold(gold_path, nodes_allowed)
    diagnosis = diagnose(user_query, gold_df, nodes_allowed, embedding_model=embedding_model, top_k=top_k_teaching)

    teaching_content: List[Dict[str, Any]] = []
    if teaching_path and os.path.exists(teaching_path):
        cards_df = load_teaching_cards(teaching_path)
        teaching_content = get_teaching_for_node(
            cards_df,
            diagnosis["energy_node"],
            user_text=user_query,
            embedding_model=embedding_model,
            max_items=top_k_teaching,
        )

    return {
        "query": user_query,
        "diagnosis": {
            "energy_node": diagnosis["energy_node"],
            "aspect": diagnosis["aspect"],
            "matched_problem": diagnosis.get("matched_problem"),
            "confidence": diagnosis.get("confidence", "keyword_fallback"),
        },
        "framework_solution": diagnosis.get("framework_row") or {},
        "teaching_content": teaching_content,
        "local_only": True,
    }
