from __future__ import annotations
import re

def meaning_score(t: str) -> int:
    t = str(t or "").strip().lower()
    if len(t.split()) < 35:
        return 0
    score = 0
    if any(k in t for k in ["because", "so", "therefore", "that is", "means", "example", "trap", "principle"]):
        score += 2
    bad = sum(1 for w in t.split() if len(w) <= 2)
    if bad / max(1, len(t.split())) < 0.20:
        score += 1
    words = re.findall(r"[a-z']+", t)
    if len(words) > 40:
        uniq = len(set(words)) / len(words)
        if uniq > 0.35:
            score += 2
    return score

def alpha_ratio(t: str) -> float:
    t = str(t or "")
    a = sum(ch.isalpha() for ch in t)
    return a / max(1, len(t))

def uniq_word_ratio(t: str) -> float:
    words = re.findall(r"[a-z']+", str(t or "").lower())
    if len(words) < 30:
        return 1.0
    return len(set(words)) / len(words)

def short_token_ratio(t: str) -> float:
    toks = re.findall(r"\w+", str(t or "").lower())
    if len(toks) < 20:
        return 1.0
    short = sum(1 for w in toks if len(w) <= 2)
    return short / len(toks)

def repeated_ngram_count(t: str, n=2) -> int:
    words = re.findall(r"[a-z']+", str(t or "").lower())
    if len(words) < 40:
        return 0
    grams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    return len(grams) - len(set(grams))

def fragment_count(t: str) -> int:
    parts = re.split(r"[\.!?]+", str(t or ""))
    parts = [p.strip() for p in parts if p.strip()]
    tiny = sum(1 for p in parts if len(p.split()) <= 3)
    return tiny

def junk_score_generic(t: str) -> int:
    t = str(t or "").strip()
    if not t:
        return 10
    score = 0
    ar = alpha_ratio(t)
    uw = uniq_word_ratio(t)
    st = short_token_ratio(t)
    rep2 = repeated_ngram_count(t, n=2)
    rep3 = repeated_ngram_count(t, n=3)
    fr = fragment_count(t)

    if ar < 0.55: score += 3
    elif ar < 0.65: score += 1

    if uw < 0.28: score += 3
    elif uw < 0.35: score += 2
    elif uw < 0.42: score += 1

    if st > 0.28: score += 2
    elif st > 0.22: score += 1

    if rep3 > 25: score += 2
    if rep2 > 35: score += 2

    if fr >= 6: score += 2
    elif fr >= 4: score += 1
    return score
