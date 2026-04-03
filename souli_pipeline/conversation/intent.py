"""
Intent detector — decides whether the user just wants to be heard (vent mode),
is openly sharing (sharing mode), or is actively looking for guidance/solutions
(solution mode).

Uses keyword heuristics first; can be upgraded to LLM classification if needed.
"""
from __future__ import annotations

import re
from typing import Literal

IntentType = Literal["venting", "sharing", "solution", "unclear"]

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
    r"\bi want (to do|something|to find|to feel)\b",
    r"\bmake (things|it|this) better\b",
    r"\bfeel better\b",
    r"\bget better\b",
    r"\bwhat (can i|should i) do\b",
    r"\bi('m| am) looking for\b",
    r"\bhelp me (get|feel|be|find)\b",
    r"\bwant (to get|to feel|things) better\b",
    r"\bwhat do i do\b",
    r"\bdo something\b",
    # Hinglish / casual solution requests
    r"\bsolution (do|de|bao|btao|na|chahiye)\b",
    r"\b(btao|batao)\b",
    r"\b(bata|bata do|bata na)\b",
    r"\bkya karu\b",
    r"\bkya (karna|karna chahiye|kare|karein)\b",
    r"\b(tell|give).{0,10}solution\b",
    r"\bsolution\b",
    r"\badvice\b",
    r"\bguidance\b",
    r"\bkya (sochta|lagta|bolte)\b",
    r"\b(suggest|suggestion)\b",
    r"\bbhaag (jau|jaun|jau kya)\b",
]

_VENTING_PATTERNS = [
    r"\bjust (want|need) to (vent|talk|say)\b",
    r"\bjust listen\b",
    r"\bi('m| am) (just|only) (venting|talking)\b",
    r"\bnot (looking for|asking for) (advice|solution|help)\b",
    r"\bdon't (need|want) advice\b",
    r"\bi know what (i'm|i am) doing\b",
]

# Sharing patterns: user is open, talkative, wants to be heard and understood
_SHARING_PATTERNS = [
    r"\bjust (want|need) to (share|open up|express)\b",
    r"\bi('m| am) (just|only) sharing\b",
    r"\bwanted to (tell|share|talk about)\b",
    r"\bi feel (so|really|very|completely)\b",
    r"\bi('ve| have) been (feeling|going through|dealing with)\b",
    r"\brecently\b",
    r"\blately\b",
    r"\bsince (last|the|a)\b",
]

_UNCLEAR_PATTERNS = [
    r"\bi (don't|do not) (know|understand)\b",
    r"\bmaybe\b",
    r"\bi('m| am) not sure\b",
]

# ---------------------------------------------------------------------------
# Summary confirmation detection
# ---------------------------------------------------------------------------

_SUMMARY_YES_PATTERNS = [
    r"\b(yes|yeah|yep|yup|correct|right|exactly|precisely|that('?s| is) right)\b",
    r"\byes.{0,20}(help|move|proceed|go ahead|ready)\b",
    r"\b(haan|ha|sahi|bilkul|theek hai)\b",  # Hinglish
    r"\bthat('?s| is) (correct|right|it|accurate)\b",
    r"\byou('?ve| have) (got|understood|captured) (it|me|that)\b",
    r"\bpretty much\b",
    r"\bmore or less\b",
]

_SUMMARY_WANT_MORE_TALK_PATTERNS = [
    r"\b(but|also|and also|actually|wait)\b.{0,30}(want|need|like) to (share|talk|say|add)\b",
    r"\bthere('?s| is) (more|something else|another thing)\b",
    r"\bi (also|want to) (add|mention|tell you)\b",
    r"\bnot (quite|exactly|fully|completely)\b",
    r"\bkind of but\b",
    r"\bnot really\b",
]


def detect_summary_response(text: str) -> Literal["confirmed", "wants_more", "correction", "unclear"]:
    """
    Detect how the user responded to Souli's summary confirmation message.

    Returns:
        "confirmed"    — user agrees with summary and is ready to move forward
        "wants_more"   — user agrees but wants to keep sharing before moving on
        "correction"   — user disagrees or wants to correct the summary
        "unclear"      — couldn't determine
    """
    t = (text or "").lower().strip()

    # Short affirmative: "yes", "yeah", "correct", etc. → confirmed
    short_yes = re.match(r"^(yes|yeah|yep|yup|correct|right|exactly|haan|ha|sahi|hmm|ok|okay|sure)[\.,!]*$", t)
    if short_yes:
        return "confirmed"

    # Yes + want to talk more
    for pat in _SUMMARY_WANT_MORE_TALK_PATTERNS:
        if re.search(pat, t):
            return "wants_more"

    # Affirmative
    for pat in _SUMMARY_YES_PATTERNS:
        if re.search(pat, t):
            # Check if they also want to add more
            if any(re.search(p, t) for p in _SUMMARY_WANT_MORE_TALK_PATTERNS):
                return "wants_more"
            return "confirmed"

    # Negation / correction signals
    negation_words = ["no", "not really", "not quite", "that's not", "you got it wrong",
                      "nahi", "nope", "incorrect", "wrong", "misunderstood"]
    if any(w in t for w in negation_words):
        return "correction"

    # If it's a long message with new info, probably adding more
    if len(t.split()) > 20:
        return "wants_more"

    return "unclear"


def detect_intent(text: str, history_texts: list[str] | None = None) -> IntentType:
    """
    Detect user intent from current message and optionally recent history.
    Returns 'sharing', 'venting', 'solution', or 'unclear'.
    """
    combined = text.lower()
    if history_texts:
        combined = " ".join([combined] + [h.lower() for h in history_texts[-2:]])

    # Solution signals are strong — check first
    for pat in _SOLUTION_PATTERNS:
        if re.search(pat, combined):
            return "solution"

    # Explicit sharing signals
    for pat in _SHARING_PATTERNS:
        if re.search(pat, combined):
            return "sharing"

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