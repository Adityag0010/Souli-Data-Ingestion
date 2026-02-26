"""
Counselor response generator.

Uses Ollama llama3.1 + RAG context from Qdrant to generate responses
that mirror the warm, grounded style of the Souli video counselor.

All inference is local. No data leaves the machine.
"""
from __future__ import annotations

import logging
from typing import Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — defines counselor personality
# ---------------------------------------------------------------------------

_COUNSELOR_SYSTEM = """\
You are Souli, a warm and deeply empathetic inner wellness guide.
You speak like a trusted counselor — calm, non-judgmental, and insightful.
You never give medical advice. You never diagnose. You never prescribe medication.

Your role:
1. Make the person feel truly heard and understood.
2. Gently reflect back what you're sensing in their words and energy.
3. When relevant, weave in wisdom from the teaching content provided to you.
4. Speak naturally — no bullet points, no lists unless asked. Flowing, warm sentences.
5. Keep responses concise (3–5 sentences) unless the person has shared a lot.
6. Never push solutions. Follow the person's lead.
7. Use Indian cultural context sensitively — you understand family pressure, role expectations,
   emotional labor, and social timelines that many Indian women face.

Energy framework context:
- blocked_energy: withdrawal, numbness, depression, stuck cycles
- depleted_energy: exhausted, victimized, low self-worth, fear of failure
- scattered_energy: overwhelmed externally, burnout, anxious, no satisfaction
- outofcontrol_energy: anger, restlessness, emotional extremes, impulsive
- normal_energy: stable, seeking growth and purpose

When you have teaching content from the counselor's videos, reflect those insights naturally
in your response — as if you are the counselor from those videos, speaking in that same voice.
"""

_SOLUTION_SYSTEM = """\
You are Souli, a warm and practical inner wellness guide.
The person has asked for guidance. Provide it with warmth and clarity.

Present the practices gently — not as prescriptions, but as invitations.
Format: 2–3 short paragraphs. No numbered lists unless presenting multiple practices.
Ground everything in what the person shared — make it personal, not generic.
"""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_rag_context(chunks: List[Dict]) -> str:
    if not chunks:
        return ""
    lines = ["[Relevant teaching from Souli counselor videos:]"]
    for i, c in enumerate(chunks[:3], 1):
        text = (c.get("text") or "").strip()
        if text:
            lines.append(f"{i}. {text[:400]}")
    return "\n".join(lines)


def _build_chat_messages(
    history: List[Dict[str, str]],
    user_message: str,
    rag_chunks: List[Dict],
    energy_node: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Build the messages list for Ollama chat.
    Injects RAG context as a system-level assistant hint before the user message.
    """
    messages = list(history)  # copy existing history

    # Inject RAG context as a contextual hint (injected as assistant pre-context)
    rag_text = _build_rag_context(rag_chunks)
    if rag_text:
        messages.append({"role": "assistant", "content": rag_text})

    messages.append({"role": "user", "content": user_message})
    return messages


def _build_solution_prompt(
    energy_node: str,
    framework_solution: Dict,
    user_context: str,
) -> str:
    node_label = energy_node.replace("_", " ").title()

    practices = framework_solution.get("primary_practices ( 7 min quick relief)", "")
    healing = framework_solution.get("primary_healing_principles", "")
    deeper = framework_solution.get("deeper_meditations_program ( 7 day quick recovery)", "")

    prompt = (
        f"The person is experiencing {node_label}.\n\n"
        f"What they shared: {user_context[:600]}\n\n"
        f"Healing principles: {healing[:400]}\n\n"
        f"Quick relief practices (7 min): {practices[:300]}\n\n"
        f"Deeper recovery program: {deeper[:300]}\n\n"
        f"Write a warm, personal response presenting this guidance to them."
    )
    return prompt


# ---------------------------------------------------------------------------
# Main response functions
# ---------------------------------------------------------------------------

def generate_counselor_response(
    history: List[Dict[str, str]],
    user_message: str,
    rag_chunks: List[Dict],
    energy_node: Optional[str] = None,
    ollama_model: str = "llama3.1",
    ollama_endpoint: str = "http://localhost:11434",
    temperature: float = 0.75,
    stream: bool = False,
) -> str | Generator[str, None, None]:
    """
    Generate an empathetic counselor response using Ollama llama3.1 + RAG.

    stream=True returns a generator of text chunks.
    stream=False returns the full response string.
    """
    from ..llm.ollama import OllamaLLM

    llm = OllamaLLM(
        model=ollama_model,
        endpoint=ollama_endpoint,
        temperature=temperature,
        num_ctx=4096,
    )

    messages = _build_chat_messages(history, user_message, rag_chunks, energy_node)

    if stream:
        return llm.chat_stream(messages, system=_COUNSELOR_SYSTEM, temperature=temperature)
    else:
        return llm.chat(messages, system=_COUNSELOR_SYSTEM, temperature=temperature)


def generate_solution_response(
    energy_node: str,
    framework_solution: Dict,
    user_context: str,
    ollama_model: str = "llama3.1",
    ollama_endpoint: str = "http://localhost:11434",
    temperature: float = 0.65,
    stream: bool = False,
) -> str | Generator[str, None, None]:
    """
    Generate a solution-mode response: warm presentation of practices + meditations.
    """
    from ..llm.ollama import OllamaLLM

    llm = OllamaLLM(
        model=ollama_model,
        endpoint=ollama_endpoint,
        temperature=temperature,
        num_ctx=4096,
    )

    prompt = _build_solution_prompt(energy_node, framework_solution, user_context)
    messages = [{"role": "user", "content": prompt}]

    if stream:
        return llm.chat_stream(messages, system=_SOLUTION_SYSTEM, temperature=temperature)
    else:
        return llm.chat(messages, system=_SOLUTION_SYSTEM, temperature=temperature)


def fallback_response(energy_node: Optional[str]) -> str:
    """Simple fallback when Ollama is unavailable."""
    node_responses = {
        "blocked_energy": (
            "I can feel that you're carrying something really heavy right now. "
            "It's okay to just be where you are. You don't have to have it all figured out. "
            "I'm here with you."
        ),
        "depleted_energy": (
            "It sounds like you've been giving so much — to everyone except yourself. "
            "That tiredness you feel is real, and it's telling you something important. "
            "You matter too."
        ),
        "scattered_energy": (
            "It sounds like everything is pulling at you from all directions. "
            "That overwhelm is exhausting. You deserve to breathe and just be for a moment."
        ),
        "outofcontrol_energy": (
            "I can sense there's a lot of intensity inside right now. "
            "Those feelings are valid — they're not weakness. "
            "Let's take this one breath at a time."
        ),
        "normal_energy": (
            "It sounds like you're in a reflective space, looking for something deeper. "
            "That curiosity about growth is a beautiful starting point."
        ),
    }
    return node_responses.get(
        energy_node or "",
        "Thank you for sharing that with me. I'm here and I'm listening.",
    )
