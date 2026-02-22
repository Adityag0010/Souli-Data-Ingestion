"""Clean and merge caption segments before chunking."""
from __future__ import annotations
import re
from typing import List, Dict

def strong_clean_text(t: str) -> str:
    """Remove fillers, repeated phrases, and very short fragments."""
    if not t:
        return ""
    t = str(t).strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\b(\w+)( \1\b){2,}", r"\1", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(\w+\s+\w+)( \1\b){1,}", r"\1", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(\w+)\.\s+\1\.", r"\1.", t, flags=re.IGNORECASE)
    if len(t.split()) < 3:
        return ""
    return t.strip()


def light_dedupe_text(t: str) -> str:
    """Light dedupe for merge step (no drop)."""
    t = (t or "").strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\b(\w+)( \1\b){2,}", r"\1", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(\w+\s+\w+)( \1\b){1,}", r"\1", t, flags=re.IGNORECASE)
    return t.strip()


def merge_micro_segments(
    segs: List[Dict],
    min_dur: float = 0.35,
    min_words: int = 2,
    max_gap: float = 0.20,
) -> List[Dict]:
    """Merge tiny/overlapping segments into fewer, longer ones."""
    merged: List[Dict] = []
    cur: Dict | None = None

    def wc(x: str) -> int:
        return len(re.findall(r"\w+", x or ""))

    for s in segs:
        st = float(s["start"])
        en = float(s["end"])
        tx = light_dedupe_text(s.get("text", ""))
        if not tx:
            continue

        if cur is None:
            cur = {"start": st, "end": en, "text": tx}
            continue

        gap = st - cur["end"]
        dur = cur["end"] - cur["start"]

        if (dur < min_dur) or (wc(cur["text"]) < min_words) or (gap <= max_gap):
            cur["text"] = (cur["text"] + " " + tx).strip()
            cur["end"] = max(cur["end"], en)
        else:
            merged.append(cur)
            cur = {"start": st, "end": en, "text": tx}

    if cur:
        merged.append(cur)
    return merged


def clean_and_merge_segments(
    segments: List[Dict],
    min_dur: float = 0.35,
    min_words: int = 2,
    max_gap: float = 0.20,
) -> List[Dict]:
    """Apply strong clean then merge micro segments."""
    cleaned: List[Dict] = []
    for s in segments:
        txt = strong_clean_text(s.get("text", ""))
        if not txt:
            continue
        cleaned.append({
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": txt,
        })
    return merge_micro_segments(cleaned, min_dur=min_dur, min_words=min_words, max_gap=max_gap)
