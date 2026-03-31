"""
Intake questioner — asks empathetic, probing questions to understand which
energy state the user is experiencing.

Questions are derived directly from the Inner Energy Framework's typical_signs,
so each question maps naturally to symptoms of a specific energy node.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Framework-grounded opening probes
# Each question gently probes for signs of a specific energy node.
# We use these as RAG context hints — we don't dump them all at once.
# ---------------------------------------------------------------------------

ENERGY_PROBES: Dict[str, List[str]] = {
    "blocked_energy": [
        "What does a normal day look like for you right now?",
        "Is there something you used to do that you've kind of stopped doing lately?",
        "When did things start feeling this way — was it gradual or did something happen?",
    ],
    "depleted_energy": [
        "What's been taking the most out of you lately?",
        "When did you last feel like you had enough energy to just... be yourself?",
        "Is there something you feel like you're always giving but rarely getting back?",
    ],
    "scattered_energy": [
        "What's been the loudest thing in your head lately — the one thing that won't quiet down?",
        "Does it feel like everything needs your attention at once, or is there one thing pulling you most?",
        "When you picture a calmer version of your day, what does that look like?",
    ],
    "outofcontrol_energy": [
        "Is there a particular situation lately where you felt like things just got away from you?",
        "What does it feel like just before things get intense — is there a pattern you've noticed?",
        "When things feel out of hand, what's your usual first instinct?",
    ],
    "normal_energy": [
        "What's been on your mind lately — is there something you feel like you're ready to work on?",
        "Is there an area of your life that feels like it's calling for more attention right now?",
        "What would things look like if they felt a little more fulfilling than they do now?",
    ],
}

# ---------------------------------------------------------------------------
# Sharing probes — lighter, open-ended, for users who are already opening up.
# Less clinical, more "I'm curious, tell me more" energy.
# ---------------------------------------------------------------------------

SHARING_PROBES: Dict[str, List[str]] = {
    "blocked_energy": [
        "How long has this been sitting with you?",
        "What does a typical day feel like for you right now?",
    ],
    "depleted_energy": [
        "What's been the heaviest part of all this for you?",
        "When did you last feel like yourself?",
    ],
    "scattered_energy": [
        "What's been the loudest thing on your mind lately?",
        "Is there one thing that, if it were sorted, would make the rest feel more manageable?",
    ],
    "outofcontrol_energy": [
        "What do you think has been building up the most?",
        "Is there a particular moment or situation that keeps coming up for you?",
    ],
    "normal_energy": [
        "What's been calling your attention lately — what feels like it wants to shift?",
        "What would growth or fulfilment look like for you right now?",
    ],
}

# First message Souli sends — warm intro, asks for name
GREETING_MESSAGE = (
    "Hi, I'm Souli. I'm here to sit with you — to understand your soul, your emotions, "
    "and walk alongside you. Before we begin, could you tell me your name?"
)

# Opening question after name is collected — light, not heavy
OPENING_QUESTION = (
    "I'm here with you. What's been on your mind or in your heart lately? "
    "You can share as much or as little as you feel comfortable with."
)

# Follow-up when user gives short answers
SHORT_ANSWER_FOLLOW_UPS = [
    "I hear you. Can you tell me a little more about what that feels like for you?",
    "That makes sense. Has this been going on for a while, or did something shift recently?",
    "Thank you for sharing that. When did you first start feeling this way?",
    "I'm listening. Is there a specific situation or feeling that's been coming up most often?",
]

# Commitment check questions (drawn from ExpressionsMapping "Reality Commitment Check")
COMMITMENT_CHECKS: Dict[str, str] = {
    "blocked_energy": "Am I ready to feel discomfort in order to heal and reconnect?",
    "depleted_energy": "Am I ready to choose myself — even if it means disappointing others?",
    "scattered_energy": "Am I ready to set boundaries and take charge of my life?",
    "outofcontrol_energy": "Am I ready to pause before reacting and find balance?",
    "normal_energy": "Am I ready to grow beyond my comfort zone?",
}

# ---------------------------------------------------------------------------
# Rich-message detection
# ---------------------------------------------------------------------------

# Emotional / situational keywords that signal the user has already shared
# meaningful context in their opening message.
_RICH_EMOTION_WORDS = {
    "breakup", "broke up", "divorce", "cheated", "cheating", "fired", "lost my job",
    "lost", "death", "died", "grief", "grieving", "depressed", "depression", "anxiety",
    "anxious", "panic", "crying", "cried", "suicidal", "self harm", "harming",
    "abuse", "abused", "trauma", "traumatic", "lonely", "alone", "abandoned",
    "rejected", "failure", "failed", "worthless", "hopeless", "helpless",
    "overwhelmed", "exhausted", "burnout", "burnt out", "numb", "empty",
    "stressed", "pressure", "fight", "argument", "relationship", "parents",
    "family", "marriage", "husband", "wife", "partner", "boyfriend", "girlfriend",
    "job", "career", "money", "financial", "debt", "sick", "illness", "hospital",
    "heartbreak", "heartbroken", "miss", "missing", "hate myself", "hate my",
    "can't sleep", "can't eat", "can't focus", "can't stop", "can't cope",
    # Hinglish
    "bahut", "zyada", "dard", "pareshan", "takleef", "akela", "akeli",
    "toot gaya", "toot gayi", "ro raha", "ro rahi", "nahi rehna",
}

_RICH_MIN_WORDS = 12   # message must have at least this many words to be "rich"
_RICH_MIN_EMOTIONAL_HITS = 1  # at least one emotional keyword


def is_rich_message(text: str) -> bool:
    """
    Returns True if the user's message already contains enough context
    (situation + emotion) to skip deep intake and move quickly toward summary.

    Criteria:
    - At least _RICH_MIN_WORDS words
    - At least one emotional/situational keyword
    """
    if not text:
        return False
    text_lower = text.lower()
    word_count = len(text_lower.split())
    if word_count < _RICH_MIN_WORDS:
        return False
    hits = sum(1 for kw in _RICH_EMOTION_WORDS if kw in text_lower)
    return hits >= _RICH_MIN_EMOTIONAL_HITS


# ---------------------------------------------------------------------------
# Accessor functions
# ---------------------------------------------------------------------------

def get_greeting() -> str:
    return GREETING_MESSAGE


def get_opening() -> str:
    return OPENING_QUESTION


def get_probe(energy_node: str, used_indices: List[int]) -> Optional[str]:
    """
    Return the next unused probe question for the given energy_node.
    Returns None if all probes have been used.
    """
    probes = ENERGY_PROBES.get(energy_node, [])
    for i, q in enumerate(probes):
        if i not in used_indices:
            return q
    return None


def get_sharing_probe(energy_node: str, used_indices: List[int]) -> Optional[str]:
    """
    Return the next unused *sharing* probe for the given energy_node.
    Sharing probes are lighter and less clinical than standard intake probes.
    Returns None if all sharing probes have been used.
    """
    probes = SHARING_PROBES.get(energy_node, [])
    for i, q in enumerate(probes):
        if i not in used_indices:
            return q
    return None


def get_short_follow_up(turn: int) -> str:
    return SHORT_ANSWER_FOLLOW_UPS[turn % len(SHORT_ANSWER_FOLLOW_UPS)]


def get_commitment_check(energy_node: str) -> str:
    return COMMITMENT_CHECKS.get(
        energy_node,
        "Are you ready to take the first small step toward feeling better?",
    )


def is_short_answer(text: str, min_words: int = 8) -> bool:
    return len((text or "").split()) < min_words