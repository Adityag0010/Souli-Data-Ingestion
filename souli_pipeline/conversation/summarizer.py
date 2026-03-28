"""
Summarizer — generates a concise empathetic summary of what Souli has
understood about the user's situation and asks for confirmation.

Called when ConversationEngine decides enough context has been gathered
to wrap up the intake/sharing phase and move toward intent/solution.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Summary confirmation template
# ---------------------------------------------------------------------------


TONE_STYLES = [
    "Gentle and reflective",
    "Warm and soulful",
    "Direct and grounded",
    "Deeply empathetic and quiet",
    "Supportive and attentive"
]

OPENING_PHRASES = [
    "So from what you've shared, it sounds like"
    "I've been listening closely, and it feels like",
    "If I'm hearing you correctly, it sounds as though",
    "It seems like what's weighing on you is",
    "From everything you've shared, I'm picking up on",
    "It sounds like you're navigating a space where"
]

CLOSING_INVITATIONS = [
    "Does that sit right with you? If it does, we can look at some ways to ease this, or we can just stay here if you have more to say.",
    "Am I capturing that correctly? We can explore some support together whenever you're ready, or just keep talking.",
    "Does that resonate? I'm here to help find a way forward, but there's no rush if you need to share more.",
    "Is that how it feels for you? I'd love to help you find some balance, or we can simply hold this space a bit longer."
]


def build_dynamic_system_prompt(user_name: Optional[str] = None) -> str:
    """Creates a unique system instruction for every call."""
    style = random.choice(TONE_STYLES)
    opening = random.choice(OPENING_PHRASES)
    closing = random.choice(CLOSING_INVITATIONS)
    
    name_clause = f"Address the user as {user_name}." if user_name else "Be intimate but respectful."
    
    return (
        f"You are Souli, a warm empathetic companion. {name_clause} "
        f"Your tone today is {style}. Your goal is to synthesize the user's situation. "
        f"\n\nSTRICT CONSTRAINTS:\n"
        f"1. Start your response with a variation of: '{opening}...'\n"
        f"2. Write ONE clear, heartfelt summary sentence of their struggle.\n"
        f"3. End your response with this exact sentiment (but you may tweak the words slightly): '{closing}'\n"
        f"4. Do NOT use robotic phrases like 'In summary' or 'To conclude'.\n"
        f"5. Total response should be under 80 words."
    )
# ---------------------------------------------------------------------------
# Main Summary Logic
# ---------------------------------------------------------------------------

def generate_summary(
    user_text_buffer: str,
    energy_node: Optional[str],
    user_name: Optional[str] = None,
    ollama_model: str = "llama3.1",
    ollama_endpoint: str = "http://localhost:11434",
    temperature: float = 0.75,
) -> str:
    """
    Synthesizes user input into a dynamic, empathetic summary.
    Replaces static templates with a randomized structural prompt.
    """
    try:
        from ..llm.ollama import OllamaLLM

        llm = OllamaLLM(
            model=ollama_model,
            endpoint=ollama_endpoint,
            temperature=temperature,
            num_ctx=4096,
        )

        if not llm.is_available():
            logger.warning("Summary generation failed — using fallback.")
            return _fallback_summary(energy_node, user_name)

        # Build the dynamic instructions
        system_instruction = build_dynamic_system_prompt(user_name)

        prompt = (
            f"User's shared context:\n"
            f"\"\"\"\n{user_text_buffer[:1500].strip()}\n\"\"\"\n\n"
            f"Generate the empathetic summary and check-in now:"
        )

        # Generate the entire block (Summary + Confirmation + CTA) in one go
        full_response = llm.generate(prompt=prompt, system=system_instruction)
        
        return full_response.strip().strip('"')

    except Exception as exc:
        logger.error("Dynamic summary generation failed: %s", exc)
        return _fallback_summary(energy_node, user_name)

def _fallback_summary(energy_node: Optional[str], user_name: Optional[str]) -> str:
    """A slightly improved fallback if the LLM is down."""
    stubs = {
        "blocked_energy": "you're feeling a bit stuck, like things have come to a standstill",
        "depleted_energy": "you're carrying a lot right now and feeling quite drained",
        "scattered_energy": "everything feels a bit chaotic and overwhelming at the moment",
        "outofcontrol_energy": "there's a lot of intense emotion building up inside — anger, restlessness, or reactions that feel bigger than you'd like them to be",
        "normal_energy": "you're in a relatively stable place but looking for more meaning, growth, or a deeper sense of fulfilment",
    }
    stub = stubs.get(energy_node, "you're going through a lot of changes right now")
    name_addr = f"{user_name}, " if user_name else ""
    return f"{name_addr}It sounds like {stub}. Did I get that right? I'm here if you want to explore some help or just keep talking."