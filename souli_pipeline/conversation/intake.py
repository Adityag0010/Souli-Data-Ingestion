"""
Intake questioner — asks empathetic, probing questions to understand which
energy state the user is experiencing.

Questions are derived directly from the Inner Energy Framework's typical_signs,
so each question maps naturally to symptoms of a specific energy node.
"""
from __future__ import annotations

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Framework-grounded opening probes
# Each question gently probes for signs of a specific energy node.
# We use these as RAG context hints — we don't dump them all at once.
# ---------------------------------------------------------------------------

ENERGY_PROBES: Dict[str, List[str]] = {
    "blocked_energy": [
        "Are you finding it hard to connect with the world around you right now?",
        "Have you been withdrawing from people or things you used to enjoy?",
        "Does it feel like you're just going through the motions, feeling numb inside?",
    ],
    "depleted_energy": [
        "Do you feel like you're running on empty, even when you haven't done much?",
        "Are you finding it hard to complete things you start, or feeling like nobody really values you?",
        "Does it feel like life is taking from you more than it gives back?",
    ],
    "scattered_energy": [
        "Does it feel like you have too much on your plate and can't find your footing?",
        "Are you feeling overwhelmed, stressed, or anxious even though you're trying hard?",
        "Is your mind jumping between too many things, making it hard to feel satisfied with anything?",
    ],
    "outofcontrol_energy": [
        "Are you noticing strong emotions — like anger, frustration, or restlessness — that feel hard to manage?",
        "Does it feel like your mind or emotions are constantly running on overdrive?",
        "Are you reacting more intensely than you'd like in certain situations?",
    ],
    "normal_energy": [
        "It sounds like you're in a fairly stable place — are you looking to grow further or find more purpose?",
        "What does inner growth or deeper fulfillment mean to you right now?",
    ],
}

# Opening question — always asked first, neutral and warm
OPENING_QUESTION = (
    "I'm here with you. Before we go anywhere, I just want to understand — "
    "what's been weighing on your heart lately? You can share as much or as little as you feel comfortable with."
)

# Follow-up when user gives short answers
SHORT_ANSWER_FOLLOW_UPS = [
    "I hear you. Can you tell me a little more about what that feels like for you?",
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


def get_short_follow_up(turn: int) -> str:
    return SHORT_ANSWER_FOLLOW_UPS[turn % len(SHORT_ANSWER_FOLLOW_UPS)]


def get_commitment_check(energy_node: str) -> str:
    return COMMITMENT_CHECKS.get(
        energy_node,
        "Are you ready to take the first small step toward feeling better?",
    )


def is_short_answer(text: str, min_words: int = 8) -> bool:
    return len((text or "").split()) < min_words
