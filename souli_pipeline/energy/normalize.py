from __future__ import annotations
import re
import pandas as pd
from rapidfuzz import process, fuzz
from typing import List, Tuple

def normalize_aspect(x, aspects_allowed: List[str]) -> str:
    s = ("" if pd.isna(x) else str(x)).strip()
    if not s:
        return "Unknown"
    match = process.extractOne(s, aspects_allowed, scorer=fuzz.WRatio)
    return match[0] if match and match[1] >= 70 else "Unknown"

def normalize_node(x, nodes_allowed: List[str]) -> str:
    s = ("" if pd.isna(x) else str(x)).strip().lower()
    if not s:
        return ""
    s = s.replace(" ", "_").replace("/", "_")
    s = re.sub(r"[^a-z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")

    mapping = {
        "depleted": "depleted_energy",
        "depletedenergy": "depleted_energy",
        "blocked": "blocked_energy",
        "scattered": "scattered_energy",
        "out_of_control_energy": "outofcontrol_energy",
        "outofcontrol": "outofcontrol_energy",
        "normal": "normal_energy",
    }
    if s in mapping:
        return mapping[s]
    if s in nodes_allowed:
        return s

    for n in nodes_allowed:
        if n in s:
            return n

    match = process.extractOne(s, nodes_allowed, scorer=fuzz.WRatio)
    return match[0] if match and match[1] >= 75 else ""

def infer_node(problem: str, blocks: str) -> str:
    t = f"{problem} {blocks}".lower()

    depleted_kw = ["tired", "burnout", "burnt out", "exhaust", "fatigue", "low energy", "drained", "no motivation"]
    scattered_kw = ["overwhelm", "too much", "multitask", "stress", "anxious", "anxiety", "pressure", "restless", "racing mind"]
    out_kw = ["anger", "rage", "impulsive", "reactive", "panic", "explode", "overreact", "shouting"]
    blocked_kw = ["fear", "inadequacy", "failure", "self doubt", "confidence", "stuck", "procrast", "avoid", "guilt", "shame"]

    def hit(keywords): return any(k in t for k in keywords)

    if hit(out_kw): return "outofcontrol_energy"
    if hit(depleted_kw): return "depleted_energy"
    if hit(scattered_kw): return "scattered_energy"
    if hit(blocked_kw): return "blocked_energy"
    return "blocked_energy"

def normalize_blocks(x) -> str:
    s = "" if pd.isna(x) else str(x).strip()
    s = re.sub(r"\s+", " ", s)
    if not s:
        return ""
    s = re.sub(r"\b\d+\.\s*", "", s)
    parts = re.split(r"•|\n|,|;|\/", s)
    parts = [re.sub(r"\s+", " ", p).strip(" -–—\t") for p in parts]
    parts = [p for p in parts if p and p.lower() not in ["nan", "none"]]
    seen, out = set(), []
    for p in parts:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return " / ".join(out)

def blocks_count(s: str) -> int:
    s = (s or "").strip()
    if not s:
        return 0
    return len([p.strip() for p in s.split("/") if p.strip()])
