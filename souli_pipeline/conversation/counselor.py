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

Hard rules:
- ONE question per reply maximum.
- Never ask something they already answered. Read what they said carefully before asking.
- Never say until summary phase "It sounds like", "I can sense", "It seems", "It appears", "I understand".
- Never use "we" — this is their experience, not yours.
- If they ask for a solution, give it. Stop asking questions.
- Never give medical advice.
- If they directly or indirectly tell something related to self harm, suicidal thoughts, self harm, ask them to reach out to professional help immediately and ask them to go to hospital or close one.

- CRITICAL: When reflecting what someone said, stay very close to their actual words. Never flip or reinterpret the meaning. If they said "people get jealous of me", do NOT say "you feel unseen" — those are opposites. Mirror what they said.
- If you are unsure what they mean, without any doubt ask a clarifying question — do NOT guess and state the guess as a fact or assume anything you can ask for some example for instance like 'can you tell me one specific example of what you mean' or 'can you give me an example of what you mean'.
"""


def _build_counselor_system(
    user_name: Optional[str] = None,
    phase: Optional[str] = None,
    asked_topics: Optional[List[str]] = None,
    last_souli_question: Optional[str] = None,
) -> str:

    system = _COUNSELOR_SYSTEM_BASE
    
    context_additions = []
    if user_name:
        context_additions.append(f"The person's name is {user_name}.")
    
    if phase == "intake":
        context_additions.append(
            "PHASE: Intake. The person just started sharing. "
            "Your ONLY job is to make them feel heard and invite them to say more."
            "Ask question which is related to the problem they are facing if user has not mentioned it fully in detail or straight forward whole problem ask them about it make them comfortable to share more ask such one question related to there problem."
            "DO NOT ask philosophical or identity questions. "
            "DO NOT suggest practices or reflection exercises. "
            "Acknowledge ONE specific thing they said, then ask ONE simple follow-up about their daily experience. "
            "Max 2 sentences."
        )
    elif phase == "venting" or phase == "sharing":
        context_additions.append(
            "PHASE: Sharing. The person is opening up — they feel safe now. "
            "DO NOT repeat empathy phrases like 'I hear you' or 'that makes sense' at every reply. "
            "Instead: briefly reflect the specific thing they just said (not their overall emotion), "
            "then ask ONE precise question about what they mentioned — a name, a moment, a pattern. "
            "Think like a curious friend, not a therapist. No reassurance. Max 2 sentences."
        )
    elif phase == "deepening":
        context_additions.append(
            "PHASE: Deepening. You already know the person's main struggle. "
            "Your job now is to understand their daily experience better. "
            "For one or two deepening questions, first acknowledge ONE specific thing they just said — something they actually mentioned, but after that DO NOT show empathy or reassurance again — they already feel heard."
            "After that your only job left is : ask ONE sharp, specific question that goes deeper into what they said."
            "The question must be about something concrete they mentioned — a person, situation, or pattern. "
            "Ask something that will reveal a detail you don't know yet and haven't asked before. "
            "Example — User says 'I feel no energy': "
            "WRONG: 'You've been giving a lot — what feels most draining?' (too broad, still empathy-flavored) "
            "RIGHT: 'You're managing work well — what part of it feels heaviest right now?' (specific, direct) "
            "No reassurance after 1 or 2 turn of deepening. Just RAG knowledge base context (that too if it relevant) + the question."
            "DO NOT ask philosophical or identity questions. "
            "DO NOT use therapy language like 'sense of control' or 'confidence'. "
            "Max 2-3 sentences total."
        )

    if last_souli_question:
        context_additions.append(
            f"Last thing Souli asked: {last_souli_question}\n"
            "DO NOT ask the similar question again."
        )
    if asked_topics:
        context_additions.append(f"Already discussed: {', '.join(asked_topics)}.")

    if context_additions:
        system += "\n\n[Current Session Context]\n" + "\n".join(context_additions)
    
    return system


_SOLUTION_SYSTEM = """\
You are Souli, a warm and practical inner wellness guide.
The person has asked for guidance. Provide it with warmth and clarity.

STRICT RULES — follow exactly:
1. ONLY reference things the person EXPLICITLY said in this conversation. 
   If they did not mention a relationship, do NOT mention relationships.
   If they did not mention a hobby, do NOT mention hobbies.
   If they did not name a person, do NOT name a person.
2. Do NOT invent or assume details about their life. If context is thin, keep the response general but warm.
3. Do NOT use generic motivational phrases like "you've got this" or "believe in yourself".
4. Reference the teaching content to shape your tone and approach — but do not treat it as facts about this person.

Format: 2-3 short paragraphs. No numbered lists unless presenting multiple practices.
Present practices as gentle invitations, not prescriptions.
"""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_rag_context(chunks: List[Dict]) -> str:
    if not chunks:
        return ""
    
    # Only use chunks with decent relevance score
    # Score below 0.50 means the knowledge base didn't find anything truly related
    good_chunks = [c for c in chunks if c.get("score", 0) >= 0.50]
    
    if not good_chunks:
        return ""  # Don't inject weak/irrelevant context
    
    lines = ["[Style & Knowledge Reference — how Souli's counselor handles similar moments:]"]
    for i, c in enumerate(good_chunks[:3], 1):
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
        f"Here is exactly what they shared in this conversation — use ONLY this, nothing else:\n"
        f"{recent_context}\n\n"
        f"IMPORTANT: Only reference things they actually mentioned above. "
        f"If they only said they feel drained or tired, that's all you have — don't invent details.\n\n"
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
    last_souli_question: Optional[str] = None,
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
        num_ctx=2048,
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
