"""
Summarizer — generates a concise empathetic summary of what Souli has
understood about the user's situation and asks for confirmation.

Called when ConversationEngine decides enough context has been gathered
to wrap up the intake/sharing phase and move toward intent/solution.
"""
from __future__ import annotations

import logging
import random
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Summary confirmation template
# ---------------------------------------------------------------------------




TONE_STYLES = [
    "Direct and grounded",
    "Warm but concise",
    "Friendly and honest",
    "Deeply empathetic and quiet in starting phase",
    "Conversational and clear"
]

OPENING_PHRASES = [
    "So from what you've shared, it sounds like",
    "From what you've told me,",
    "I've been listening closely, and it feels like",
    "If I'm hearing you correctly, it sounds as though",
    "It seems like what's weighing on you is",
    "From everything you've shared, I'm picking up on",
    "It sounds like you're navigating a space where"
]

CLOSING_INVITATIONS = [
    "Does that feel right? We can dig into some ways to work through this, or keep talking if there's more.",
    "Am I getting that right? Happy to explore what might help, or just keep listening.",
    "Does that match what you're feeling? We can look at what to do about it, or you can say more.",
    "Sound about right? We can figure out a next step together, or just stay here a bit longer."
]


def build_dynamic_system_prompt(user_name: Optional[str] = None) -> str:
    """Creates a unique system instruction for every call."""
    style = random.choice(TONE_STYLES)
    opening = random.choice(OPENING_PHRASES)
    closing = random.choice(CLOSING_INVITATIONS)
    
    name_clause = f"Address the user as {user_name}." if user_name else "Be intimate but respectful."
    
    return (
        f"You are Souli — a warm, direct companion. {name_clause} "
        f"Your tone is {style}. Summarize what you've understood, like a friend reflecting back, not a therapist. "
        f"\n\nSTRICT CONSTRAINTS:\n"
        f"1. Start with: '{opening}'\n"
        f"2. Write ONE clear summary sentence of their struggle. "
        f"   ONLY use words and ideas from what they actually said — do not interpret or fill gaps. "
        f"   If they only said they feel tired and overwhelmed, say exactly that. "
        f"   Do NOT invent relationships, people, places, or situations they did not mention.\n"
        f"3. End with: '{closing}'\n"
        f"4. NO repeated reassurance. NO 'I hear you', 'That makes sense', 'You are not alone'. "
        f"   One small warm line max if needed — not more.\n"
        f"5. Sound like a smart friend, not a formal therapist. Total under 70 words."
    )
# ---------------------------------------------------------------------------
# Main Summary Logic
# ---------------------------------------------------------------------------

def generate_summary(
    user_text_buffer: str,
    energy_node: Optional[str],
    user_name: Optional[str] = None,
    problem_messages: Optional[list] = None,
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
            num_ctx=2048,
        )

        if not llm.is_available():
            logger.warning("Summary generation failed — using fallback.")
            return _fallback_summary(energy_node, user_name)
        
        
        if problem_messages and len(problem_messages) >= 2:
            content_to_summarize = "\n".join(f"- {m}" for m in problem_messages)
        else:
            content_to_summarize = user_text_buffer[:1500].strip()

        # Build the dynamic instructions
        system_instruction = build_dynamic_system_prompt(user_name)

        prompt = (
            f"Here is exactly what the user said, word for word:\n"
            f"\"\"\"\n{content_to_summarize}\n\"\"\"\n\n"
            f"Generate the empathetic but straight forward summary now. "
            f"Only reference what is literally written above. Do not add anything else."
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
        "blocked_energy": "you've been feeling unseen — like no matter where you go or what you do, people aren't really noticing you or valuing what you bring",
        "depleted_energy": "you've been giving a lot and running low — like there's not much left for yourself right now",
        "scattered_energy": "everything feels like it's pulling at you at once and you can't find a moment to breathe",
        "outofcontrol_energy": "there are some really intense emotions building up that feel hard to manage right now",
        "normal_energy": "you're in a relatively okay place but something is still off and you're trying to figure out what",
    }
    name_addr = f"{user_name}, " if user_name else ""
    return (
        f"{name_addr}{stubs.get(energy_node, 'you are carrying something heavy right now')}. "
        f"Does that feel close to what you've been experiencing? "
        f"We can look at some ways to ease this together, or keep talking if there's more."
    )



# ── Node display labels ────────────────────────────────────────────────────
_NODE_LABELS = {
    "blocked_energy":      "Blocked Energy",
    "depleted_energy":     "Depleted Energy",
    "scattered_energy":    "Scattered Energy",
    "outofcontrol_energy": "Out-of-Control Energy",
    "normal_energy":       "Normal / Growth Energy",
}



# ── Keyword-based fallback reasoning (used when Ollama is down) ───────────
_FALLBACK_REASONS = {
    "blocked_energy": (
        "You seem emotionally stuck or withdrawn, struggling to move forward."
    ),
    "depleted_energy": (
        "You sound drained and exhausted, running low on inner energy."
    ),
    "scattered_energy": (
        "You seem overwhelmed with too much happening, unable to find focus."
    ),
    "outofcontrol_energy": (
        "Your emotions or reactions seem hard to manage right now."
    ),
    "normal_energy": (
        "You seem to be in a stable place, looking for further growth."
    ),
}

"""
    This is called ONLY at summary time (once per session) — not every turn.
    It produces a short ≤30-word statement explaining WHY the primary node
    was chosen, based on what the user actually said.
"""

def generate_node_reasoning(
    problem_messages: list,
    primary_node: str,
    secondary_node: Optional[str] = None,
    ollama_model: str = "llama3.1",
    ollama_endpoint: str = "http://localhost:11434",
    timeout_s: int = 10,
) -> str:
    """
    Generate a SHORT (≤30 word) statement explaining why the primary energy
    node was chosen for this user, based on what they actually shared.
 
    Called ONCE — only when the chatbot is about to summarize.
    NOT called every turn.
 
    Returns a string like:
      "You've been juggling too many things at once with no breathing room,
       pulling your energy in all directions."
 
    Falls back to a rule-based sentence if Ollama is unavailable.
    """
    primary_label = _NODE_LABELS.get(primary_node, primary_node.replace("_", " ").title())
    secondary_label = _NODE_LABELS.get(secondary_node, "") if secondary_node else None
 
    # Build the context from last 5 problem messages
    recent = [m for m in (problem_messages or []) if len(m.split()) >= 4][-5:]
    if not recent:
        return _FALLBACK_REASONS.get(primary_node, "")
 
    context_text = "\n".join(f"- {m}" for m in recent)
 
    secondary_hint = (
        f'\nNote: there is also a secondary pattern of "{secondary_label}" in what they shared.'
        if secondary_label else ""
    )
 
    system = (
        "You are an energy analyst for the Souli wellness framework. "
        "Your job is to write ONE SHORT sentence (maximum 30 words) that explains "
        "WHY a person's energy pattern was identified. "
        "Write in second person ('You...'). "
        "Be specific — reference what they actually said. "
        "Do NOT name the energy node label. "
        "Do NOT use therapy jargon. "
        "Output ONLY the sentence, nothing else."
    )
 
    prompt = (
        f"The person's energy pattern was identified as: {primary_label}.{secondary_hint}\n\n"
        f"What they shared across the conversation:\n{context_text}\n\n"
        f"Write the one-sentence explanation (max 30 words):"
    )
 
    try:
        from souli_pipeline.llm.ollama import OllamaLLM
 
        llm = OllamaLLM(
            model=ollama_model,
            endpoint=ollama_endpoint,
            timeout_s=timeout_s,
            temperature=0.4,   # low temp — we want precise, not creative
            num_ctx=2048,
        )
 
        if not llm.is_available():
            logger.debug("Ollama offline — using fallback node reasoning")
            return _FALLBACK_REASONS.get(primary_node, "")
 
        raw = llm.generate(prompt=prompt, system=system)
        reasoning = raw.strip().strip('"').strip("'")
 
        # Safety: truncate if LLM went over 30 words
        words = reasoning.split()
        if len(words) > 35:
            reasoning = " ".join(words[:30]) + "."
 
        return reasoning if reasoning else _FALLBACK_REASONS.get(primary_node, "")
 
    except Exception as exc:
        logger.warning("generate_node_reasoning failed: %s — using fallback", exc)
        return _FALLBACK_REASONS.get(primary_node, "")
 