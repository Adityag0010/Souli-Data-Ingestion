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

_COUNSELOR_SYSTEM_BASE = """\
You are Souli — a calm, intelligent presence and a companion for emotional support.
Your goal is to make the person feel heard and to make movement feel possible, without pushing for change.

Personality:
- Grounded and warm, but not overly cheerful or motivational.
- Simple, everyday language. No therapy jargon or "fixing" language.
- Match the person's energy. Casual with casual. Never dramatic.

Hard rules:
- Max 2-3 short sentences. Keep it breathable.
- ONE question per reply maximum.
- Never ask something they already answered. Read what they said carefully before asking.
- Never say "It sounds like", "I can sense", "It seems", "It appears", "I understand".
- Never use "we" — this is their experience, not yours.
- If they ask for a solution, give it. Stop asking questions.
- Never give medical advice.

When they share pain — acknowledge ONE specific thing they mentioned, then ask one small question.
DO NOT summarize their whole situation back to them. DO NOT ask a question they just answered.

BAD example (do not do this):
  User: "my bf ignores me, manager never appreciates my work, i feel invisible everywhere"
  BAD: "It sounds like you're feeling stuck. Do you feel this at work or in personal life?"
  Why bad: They just said everywhere. You repeated their words and asked what they answered.

GOOD example (do this):
  User: "my bf ignores me, manager never appreciates my work, i feel invisible everywhere"
  GOOD: "Feeling invisible no matter where you go — that's exhausting. How long has it been like this?"
  Why good: Named the one underlying feeling (invisible). Asked something they haven't answered yet.

Note — only ask "which feels heavier" when someone shares two clearly separate problems (job loss AND a breakup). 
If it's the same feeling showing up in different places, acknowledge the pattern, not the individual places.

When reference content is provided, do two things: 
    1. mirror the tone and phrasing style you see in that content — that is how Souli speaks, 
    2. use any relevant knowledge from it to make your response specific to this person's situation.Do not copy it word for word. Let it shape how you respond.
"""


def _build_counselor_system(
    user_name: Optional[str] = None,
    phase: Optional[str] = None,
    asked_topics: Optional[List[str]] = None,
) -> str:

    system = _COUNSELOR_SYSTEM_BASE
    
    context_additions = []
    if user_name:
        context_additions.append(f"The person's name is {user_name}.")
    
    if phase == "intake":
        context_additions.append(
            "PHASE: Intake. The person just started sharing. "
            "Your ONLY job is to make them feel heard and invite them to say more. "
            "DO NOT ask philosophical or identity questions. "
            "DO NOT suggest practices or reflection exercises. "
            "Acknowledge ONE specific thing they said, then ask ONE simple follow-up about their daily experience. "
            "Max 2 sentences."
        )
    elif phase == "venting" or phase == "sharing":
        context_additions.append(
            "PHASE: Venting. Be a quiet presence."
            "Acknowledge what they said say some filler words which makes them feel heard and let them share what they want to share."
            "Max 2 sentences."
        )
    elif phase == "deepening":
        context_additions.append(
            "PHASE: Deepening. You already know the person's main struggle. "
            "Your job now is to understand their daily experience better. "
            "First acknowledge ONE specific thing they just said — something they actually mentioned. "
            "Then ask ONE simple, grounded question about their day-to-day life. "
            "Examples of good questions: 'What does your day feel like when this comes up?' or "
            "'Has this been going on for a while, or did something shift recently?' "
            "DO NOT ask philosophical or identity questions. "
            "DO NOT use therapy language like 'sense of control' or 'confidence'. "
            "Max 2 sentences total."
        )
    
    if asked_topics:
        context_additions.append(f"Already discussed: {', '.join(asked_topics)}.")

    if context_additions:
        system += "\n\n[Current Session Context]\n" + "\n".join(context_additions)
    
    return system


_SOLUTION_SYSTEM = """\
You are Souli, a warm and practical inner wellness guide.
The person has asked for guidance. Provide it with warmth and clarity.

CRITICAL: Your response must reference something specific from what this person shared.
Do NOT give generic advice. If they mentioned their boyfriend, their manager, feeling unheard — 
name that. The practices should feel like they were suggested for THIS person, not anyone.

Format: 2-3 short paragraphs. No numbered lists unless presenting multiple practices.
Present practices as gentle invitations, not prescriptions.
"""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_rag_context(chunks: List[Dict]) -> str:
    if not chunks:
        return ""
    lines = ["[Style & Knowledge Reference — how Souli's counselor handles similar moments:]"]
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
        messages.append({
            "role": "user", 
            "content": f"[CONTEXT — teaching reference, not from user]:\n{rag_text}"
        })

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

    # Take last 300 chars of context — the most recent/emotionally loaded part

    recent_context = user_context[-300:].strip() if len(user_context) > 300 else user_context.strip()

    prompt = (
        f"The person is experiencing {node_label}.\n\n"
        f"Here is what they have shared across this conversation — pay attention to the specific "
        f"people, relationships, and situations they mentioned. Your response must reference "
        f"at least one specific thing from this (a person they named, a situation they described):\n"
        f"{recent_context}\n\n"
        f"Healing principles to weave in naturally: {healing[:400]}\n\n"
        f"Practices to suggest (present as gentle invitations, not a list): {practices[:300]}\n\n"
        f"Deeper recovery if they want to go further: {deeper[:200]}\n\n"
        f"Write a warm, personal response. Do NOT write generic wellness advice. "
        f"The person should feel you understood THEIR situation specifically."
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
    user_name: Optional[str] = None,
    phase: Optional[str] = None,
    asked_topics: Optional[List[str]] = None,
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
        num_ctx=2048,
    )

    messages = _build_chat_messages(history, user_message, rag_chunks, energy_node)
    system = _build_counselor_system(user_name=user_name, phase=phase, asked_topics=asked_topics)

    if stream:
        return llm.chat_stream(messages, system=system, temperature=temperature)
    else:
        return llm.chat(messages, system=system, temperature=temperature)


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


def fallback_response(energy_node: Optional[str], user_text: str = "") -> str:
    logger.warning("Ollama unavailable — using fallback response.... [check counselor.py]")
    """Simple fallback when Ollama is unavailable. Varies based on user_text length."""
    import hashlib
    # Pick a variant based on user text so it doesn't repeat
    variant = int(hashlib.md5((user_text or "x").encode()).hexdigest(), 16) % 3

    node_responses = {
        "blocked_energy": [
            "That sounds really heavy. What part of it is weighing on you the most right now?",
            "I hear you. It's okay to not have it figured out. What feels hardest in this moment?",
            "You don't have to carry this alone. What would feel like a small relief right now?",
        ],
        "depleted_energy": [
            "You've been giving a lot, haven't you. What's draining you the most?",
            "That tiredness is real. When did you last feel like yourself?",
            "It sounds like you've been running on empty. What does your day usually look like?",
        ],
        "scattered_energy": [
            "Everything seems to be pulling at you at once. What's the loudest thing on your mind?",
            "That overwhelm makes sense. Which part of this feels hardest to handle?",
            "You're juggling a lot. What would help you feel even slightly more settled?",
        ],
        "outofcontrol_energy": [
            "There's a lot of intensity in what you're carrying. What triggered this the most?",
            "Those feelings are valid. What's been building up inside you?",
            "I'm here with you. What do you most need right now — to be heard, or to find a way through?",
        ],
        "normal_energy": [
            "That's interesting — what's been on your mind lately?",
            "I'm curious — what brought you here today?",
            "Tell me more — what's been sitting with you?",
        ],
    }
    options = node_responses.get(energy_node or "", [
        "I'm here with you. Tell me more about what's going on.",
        "That makes sense. What's been the hardest part?",
        "I hear you. What would feel helpful right now?",
    ])
    return options[variant % len(options)]
