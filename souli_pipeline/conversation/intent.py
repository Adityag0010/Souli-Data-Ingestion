"""
Intent detector — decides whether the user just wants to be heard (vent mode)
or is actively looking for guidance/solutions (solution mode).

Uses keyword heuristics first; can be upgraded to LLM classification if needed.
"""
from __future__ import annotations

import re
from typing import Literal

IntentType = Literal["venting", "solution", "unclear"]

# ---------------------------------------------------------------------------
# Keyword patterns
# ---------------------------------------------------------------------------

_SOLUTION_PATTERNS = [
    r"\bwhat (can|should|do) i\b",
    r"\bhow (can|do|should) i\b",
    r"\bhelp me\b",
    r"\btell me (what|how)\b",
    r"\bgive me\b",
    r"\bi (need|want) (advice|help|guidance|solution|answer)\b",
    r"\bwhat (is|are) the (solution|answer|way|steps)\b",
    r"\bshow me\b",
    r"\bwhat should\b",
    r"\bplease (help|tell|guide)\b",
    r"\bi (don't|do not) know what to do\b",
    r"\bi('m| am) (lost|confused|stuck)\b",
    r"\bfix (this|it|my)\b",
]

_VENTING_PATTERNS = [
    r"\bjust (want|need) to (vent|talk|share|say)\b",
    r"\bjust listen\b",
    r"\bi('m| am) (just|only) (sharing|venting|talking)\b",
    r"\bnot (looking for|asking for) (advice|solution|help)\b",
    r"\bdon't (need|want) advice\b",
    r"\bi know what (i'm|i am) doing\b",
    r"\bjust (feel|feeling)\b",
    r"\bi('m| am) (sad|upset|hurt|angry|frustrated|tired|exhausted|overwhelmed)\b",
    r"\bnobody (understands|listens)\b",
    r"\bi feel (so|really|very|completely)\b",
]

_UNCLEAR_PATTERNS = [
    r"\bi (don't|do not) (know|understand)\b",
    r"\bmaybe\b",
    r"\bi('m| am) not sure\b",
]


def detect_intent(text: str, history_texts: list[str] | None = None) -> IntentType:
    """
    Detect user intent from current message and optionally recent history.
    Returns 'venting', 'solution', or 'unclear'.
    """
    combined = text.lower()
    if history_texts:
        # Look at last 2 turns for context
        combined = " ".join([combined] + [h.lower() for h in history_texts[-2:]])

    # Solution signals are strong — check first
    for pat in _SOLUTION_PATTERNS:
        if re.search(pat, combined):
            return "solution"

    # Explicit venting signals
    for pat in _VENTING_PATTERNS:
        if re.search(pat, combined):
            return "venting"

    # Default to venting for short emotional statements
    words = combined.split()
    if len(words) < 15:
        return "venting"

    return "unclear"


def nudge_toward_intent(turn_count: int, max_intake: int) -> bool:
    """
    Returns True if we should gently ask the user whether they want solutions,
    based on how long the conversation has been going.
    """
    return turn_count >= max_intake


INTENT_BRIDGE = (
    "I've been listening carefully to everything you've shared. "
    "I feel like I'm starting to understand what's going on inside. "
    "Would you like to just keep talking and be heard — "
    "or would it feel helpful to explore some practices and guidance that might bring some relief?"
)
