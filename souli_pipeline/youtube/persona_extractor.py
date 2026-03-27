"""
souli_pipeline/youtube/persona_extractor.py

Two responsibilities:
  1. extract_from_video()  — reads a full cleaned transcript and extracts
     a ~100 word persona snippet capturing this coach's distinctive style.

  2. merge_persona()       — merges an existing persona string with a new
     snippet, keeping the result under max_words and non-repetitive.

The output file (coach_persona.txt) is read at conversation time and injected
into the counselor system prompt so the LLM mirrors the coach's style.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = """\
You are analysing a wellness counseling video transcript to capture the coach's unique style.
Your output will be used to train an AI to respond in the same manner.

Focus ONLY on what is distinctive and specific — not generic traits.
Output a compact description under 120 words covering these dimensions:
  1. Opening style: how does the coach frame or introduce a problem?
  2. Signature phrases or vocabulary they repeat
  3. How they validate or acknowledge the listener's feelings
  4. Metaphors or analogies they use
  5. Sentence rhythm: short punchy sentences? Long flowing ones? Mix?
  6. How they close or resolve a teaching point

Be concrete. Use actual phrases from the transcript where possible.
Output only the persona description. No headings, no preamble.
"""

_EXTRACT_USER = """\
Analyse this counseling transcript and describe the coach's style:

TRANSCRIPT:
\"\"\"
{transcript}
\"\"\"

Coach's style (under 120 words):"""


_MERGE_SYSTEM = """\
You are maintaining a compact profile of a wellness coach's communication style.
You will be given an existing profile and a new observation from a different video.

Your task: merge them into ONE updated profile.
Rules:
  - Keep ONLY what is specific and distinctive — remove generic statements
  - Remove exact duplicates and near-duplicates
  - If both sources say the same thing differently, keep the more specific version
  - Final output MUST be under {max_words} words
  - Output only the merged profile. No headings, no preamble, no explanation.
"""

_MERGE_USER = """\
EXISTING PROFILE:
\"\"\"
{existing}
\"\"\"

NEW OBSERVATION (from latest video):
\"\"\"
{new_snippet}
\"\"\"

Updated merged profile (under {max_words} words):"""


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_from_video(
    cleaned_transcript: str,
    ollama_model: str = "llama3.1",
    ollama_endpoint: str = "http://localhost:11434",
    num_ctx: int = 8192,
    temperature: float = 0.4,
) -> str:
    """
    Extract a ~100 word persona snippet from a full cleaned transcript.

    Returns the snippet string, or "" if extraction fails.

    Args:
        cleaned_transcript: All cleaned segments joined into one string
        ollama_model:        Model to use
        ollama_endpoint:     Ollama server URL
        num_ctx:             Context window (8192 allows full cleaned transcript)
        temperature:         Slightly creative to capture nuance
    """
    transcript = (cleaned_transcript or "").strip()
    if not transcript:
        return ""

    try:
        from souli_pipeline.llm.ollama import OllamaLLM

        llm = OllamaLLM(
            model=ollama_model,
            endpoint=ollama_endpoint,
            timeout_s=180,
            temperature=temperature,
            num_ctx=num_ctx,
        )

        if not llm.is_available():
            logger.warning("Ollama not available — skipping persona extraction.")
            return ""

        # Cap input to avoid exceeding context (8k context ~ 6000 words)
        prompt = _EXTRACT_USER.format(transcript=transcript[:5000])
        snippet = llm.generate(prompt=prompt, system=_EXTRACT_SYSTEM, temperature=temperature)
        snippet = snippet.strip()

        # Basic validation — should be at least 30 words
        if len(snippet.split()) < 30:
            logger.warning("Persona snippet too short (%d words) — discarding.", len(snippet.split()))
            return ""

        logger.info("Extracted persona snippet (%d words).", len(snippet.split()))
        return snippet

    except Exception as e:
        logger.warning("Persona extraction failed: %s", e)
        return ""


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_persona(
    existing_persona: str,
    new_snippet: str,
    ollama_model: str = "llama3.1",
    ollama_endpoint: str = "http://localhost:11434",
    num_ctx: int = 8192,
    temperature: float = 0.3,
    max_words: int = 350,
) -> str:
    """
    Merge existing persona string with a new snippet from the latest video.
    Returns the merged string capped at max_words.

    If existing_persona is empty, returns new_snippet directly (capped).
    If new_snippet is empty, returns existing_persona unchanged.
    """
    existing = (existing_persona or "").strip()
    new_snip  = (new_snippet or "").strip()

    if not existing:
        # First video — just use the snippet directly, cap at max_words
        words = new_snip.split()
        return " ".join(words[:max_words])

    if not new_snip:
        return existing

    try:
        from souli_pipeline.llm.ollama import OllamaLLM

        llm = OllamaLLM(
            model=ollama_model,
            endpoint=ollama_endpoint,
            timeout_s=120,
            temperature=temperature,
            num_ctx=num_ctx,
        )

        if not llm.is_available():
            logger.warning("Ollama not available — appending snippet without merge.")
            combined = existing + "\n\n" + new_snip
            words = combined.split()
            return " ".join(words[:max_words])

        system = _MERGE_SYSTEM.format(max_words=max_words)
        prompt = _MERGE_USER.format(
            existing=existing,
            new_snippet=new_snip,
            max_words=max_words,
        )

        merged = llm.generate(prompt=prompt, system=system, temperature=temperature)
        merged = merged.strip()

        # Hard cap — if LLM ignored the word limit, truncate at sentence boundary
        words = merged.split()
        if len(words) > max_words + 30:
            logger.warning("Merged persona exceeded word limit — truncating.")
            merged = _truncate_at_sentence(merged, max_words)

        logger.info("Merged persona: %d words.", len(merged.split()))
        return merged

    except Exception as e:
        logger.warning("Persona merge failed: %s — concatenating.", e)
        combined = existing + "\n\n" + new_snip
        return _truncate_at_sentence(combined, max_words)


def _truncate_at_sentence(text: str, max_words: int) -> str:
    """Truncate text at the last sentence boundary before max_words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    # Find last sentence-ending punctuation
    last_end = max(
        truncated.rfind("."),
        truncated.rfind("?"),
        truncated.rfind("!"),
    )
    if last_end > len(truncated) // 2:
        return truncated[:last_end + 1].strip()
    return truncated.strip()


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_persona(persona_path: str) -> str:
    """Load existing persona from file. Returns '' if file doesn't exist."""
    if not os.path.exists(persona_path):
        return ""
    try:
        with open(persona_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.warning("Could not read persona file %s: %s", persona_path, e)
        return ""


def save_persona(persona_path: str, persona_text: str) -> bool:
    """Save persona text to file. Returns True on success."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(persona_path)), exist_ok=True)
        with open(persona_path, "w", encoding="utf-8") as f:
            f.write(persona_text.strip())
        logger.info("Saved persona to %s (%d words).", persona_path, len(persona_text.split()))
        return True
    except Exception as e:
        logger.warning("Could not save persona to %s: %s", persona_path, e)
        return False


def update_persona_file(
    persona_path: str,
    new_snippet: str,
    ollama_model: str = "llama3.1",
    ollama_endpoint: str = "http://localhost:11434",
    num_ctx: int = 8192,
    max_words: int = 350,
) -> str:
    """
    Convenience function: load → merge → save → return updated persona.
    Call this once per video after extraction.
    """
    existing = load_persona(persona_path)
    updated  = merge_persona(
        existing_persona=existing,
        new_snippet=new_snippet,
        ollama_model=ollama_model,
        ollama_endpoint=ollama_endpoint,
        num_ctx=num_ctx,
        max_words=max_words,
    )
    save_persona(persona_path, updated)
    return updated