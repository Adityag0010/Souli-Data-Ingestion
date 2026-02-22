from __future__ import annotations
import re
from typing import List, Dict

FILLERS = [" uh ", " um ", " you know ", " hmm ", " ah ", " like "]

def normalize_text(t: str) -> str:
    x = " " + (t or "").strip() + " "
    for f in FILLERS:
        x = x.replace(f, " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x

def dedupe_repeats_in_chunk(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    parts = re.split(r"(?<=[\.\?\!])\s+|\n+", t)
    cleaned = []
    last = ""
    for p in parts:
        p2 = re.sub(r"\s+", " ", p).strip()
        if not p2:
            continue
        if p2.lower() == last.lower():
            continue
        cleaned.append(p2)
        last = p2
    out = " ".join(cleaned)
    out = re.sub(r"\b(\w+\s+\w+)\s+\1\b", r"\1", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    return out

def chunk_by_time_and_words(
    segments: List[Dict],
    max_seconds: float = 55,
    max_words: int = 220,
    max_gap: float = 1.3,
    min_words_to_split: int = 35
) -> List[Dict]:
    chunks = []
    cur = {"start": None, "end": None, "text": "", "words": 0}
    prev_end = None

    def flush():
        nonlocal cur
        txt = cur["text"].strip()
        if txt:
            chunks.append({
                "start": cur["start"],
                "end": cur["end"],
                "words": cur["words"],
                "text": dedupe_repeats_in_chunk(txt)
            })
        cur = {"start": None, "end": None, "text": "", "words": 0}

    for s in segments:
        txt = normalize_text(s.get("text", ""))
        if not txt:
            continue

        if cur["start"] is None:
            cur["start"] = float(s["start"])
            prev_end = float(s["start"])

        gap = float(s["start"]) - float(prev_end) if prev_end is not None else 0.0
        duration = float(s["end"]) - float(cur["start"])

        if gap > max_gap and cur["words"] >= min_words_to_split:
            flush()
            cur["start"] = float(s["start"])

        cur["text"] = (cur["text"] + " " + txt).strip()
        cur["end"] = float(s["end"])
        cur["words"] += len(txt.split())
        prev_end = float(s["end"])

        if duration >= max_seconds or cur["words"] >= max_words:
            flush()

    flush()
    return chunks

def split_by_words(text: str, max_words: int = 220, overlap: int = 20) -> List[str]:
    words = (text or "").split()
    if len(words) <= max_words:
        return [text]
    out = []
    i = 0
    while i < len(words):
        j = min(len(words), i + max_words)
        out.append(" ".join(words[i:j]))
        if j == len(words):
            break
        i = max(0, j - overlap)
    return out

def chunk_dedupe_heavy(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    parts = re.split(r"(?<=[\.\?\!])\s+|\n+", t)
    cleaned = []
    seen = set()
    for p in parts:
        p2 = re.sub(r"\s+", " ", p).strip()
        if not p2:
            continue
        key = p2.lower()
        if key in seen:
            continue
        seen.add(key)
        if len(p2.split()) <= 2:
            continue
        cleaned.append(p2)
    out = " ".join(cleaned)
    out = re.sub(r"\b(\w+\s+\w+)\s+\1\b", r"\1", out, flags=re.IGNORECASE)
    out = re.sub(r"\b(\w+\s+\w+\s+\w+)\s+\1\b", r"\1", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    return out
