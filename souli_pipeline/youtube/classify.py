from __future__ import annotations
import re

PROBLEM_STARTERS = [
    "how do i", "why do i", "i feel", "i am", "i'm", "how can i", "what should i",
    "i keep", "i can’t", "i can't"
]

TEACH_PATTERNS = [
    r"\bthe thing is\b",
    r"\bthat means\b",
    r"\bthis is why\b",
    r"\bthe trap is\b",
    r"\bwhen we\b",
    r"\bwe develop\b",
    r"\byou have to\b",
    r"\bwe need to\b",
    r"\bit comes from\b",
    r"\bthe point is\b",
    r"\bfor example\b",
    r"\bin india\b",
    r"\bold saying\b",
    r"\bso check\b",
]

LOGISTICS_PATTERNS = [
    r"\bwe will meet\b",
    r"\bat three\b",
    r"\broom\b",
    r"\bgarden\b",
    r"\bmic\b",
]

def clean_text(x: str) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()

def uniq_ratio(t: str) -> float:
    words = re.findall(r"[a-z']+", t.lower())
    if len(words) < 30:
        return 1.0
    return len(set(words)) / len(words)

def is_problem(t: str) -> bool:
    t2 = clean_text(t).lower()
    return any(t2.startswith(s) for s in PROBLEM_STARTERS)

def is_teaching(t: str) -> bool:
    t2 = clean_text(t).lower()
    return any(re.search(p, t2) for p in TEACH_PATTERNS)

def is_logistics(t: str) -> bool:
    t2 = clean_text(t).lower()
    return any(re.search(p, t2) for p in LOGISTICS_PATTERNS)

def classify(text: str, min_words_noise: int = 25, min_words_teaching: int = 30) -> str:
    t2 = clean_text(text)
    low = t2.lower()

    if is_problem(low):
        return "problem"
    if is_teaching(low) and len(low.split()) >= min_words_teaching:
        return "teaching"
    if is_logistics(low):
        return "noise"
    if len(low.split()) < min_words_noise:
        return "noise"
    if uniq_ratio(low) < 0.25:
        return "noise"
    return "teaching"
