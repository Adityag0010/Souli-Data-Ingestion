"""
souli_pipeline/youtube/segment_cleaner.py

Cleans one TopicSegment using llama3.1 via Ollama.

The cleaning goal is NOT summarisation.
It is rewriting messy spoken transcript into clean, readable prose that:
  - Preserves 100% of the teaching content
  - Removes filler, repetition, incomplete sentences
  - Maintains the coach's voice and style
  - Reads like a written transcript, not a cleaned-up notes version

Has retry logic — if LLM output is too short or malformed, retries once
with a stricter prompt.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a professional transcript editor working with wellness counseling videos.
Your job is to clean spoken transcripts into clear, readable prose.

STRICT RULES:
- Preserve EVERY teaching point, insight, and example. Do not summarise or shorten.
- Keep the coach's voice, tone, and sentence patterns intact.
- Remove: filler words (uh, um, like, you know), incomplete sentences, repeated phrases.
- Do NOT add new content. Do NOT explain or comment on what the coach said.
- Output only the cleaned transcript text. No headings, no bullet points, no preamble.
- Minimum output length: 60% of input word count.
"""

_USER_PROMPT = """\
Clean this spoken transcript segment. Preserve all content and the speaker's voice.

TRANSCRIPT:
\"\"\"
{text}
\"\"\"

Cleaned transcript:"""

_RETRY_SYSTEM_PROMPT = """\
You are a transcript editor. Your ONLY task is to lightly clean spoken text.
Remove filler words and obvious repetitions ONLY.
Keep everything else exactly as spoken. Do not summarise. Do not shorten.
Output only the cleaned text with no other words.
"""

_RETRY_USER_PROMPT = """\
Remove filler words and repetitions from this text. Keep all content.

TEXT:
\"\"\"
{text}
\"\"\"

Output:"""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _is_valid_output(original: str, cleaned: str, min_retention: float = 0.55) -> bool:
    """
    Check that cleaned output is not too short (LLM over-summarised).
    Returns True if output retains at least min_retention fraction of word count.
    """
    if not cleaned or not cleaned.strip():
        return False
    orig_words    = len(original.split())
    cleaned_words = len(cleaned.split())
    if orig_words == 0:
        return False
    retention = cleaned_words / orig_words
    if retention < min_retention:
        logger.warning(
            "Cleaning output too short (%.0f%% retention, min %.0f%%). Will retry.",
            retention * 100, min_retention * 100,
        )
        return False
    return True


def _postprocess(text: str) -> str:
    """Final cleanup of LLM output — remove any preamble lines it might add."""
    t = text.strip()
    # Remove lines like "Cleaned transcript:", "Here is the cleaned version:", etc.
    preamble_re = re.compile(
        r"^(cleaned transcript|here is|here's|output|result)[:\s].*\n?",
        re.IGNORECASE | re.MULTILINE,
    )
    t = preamble_re.sub("", t).strip()
    # Normalise whitespace
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r" {2,}", " ", t)
    return t.strip()


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------

def clean_segment(
    text: str,
    ollama_model: str = "llama3.1",
    ollama_endpoint: str = "http://localhost:11434",
    num_ctx: int = 8192,
    temperature: float = 0.3,
) -> str:
    """
    Clean a single transcript segment using llama3.1.

    Returns the cleaned text.
    Falls back to light regex cleaning if Ollama is unavailable.

    Args:
        text:             Raw transcript segment (300-600 words typically)
        ollama_model:     Ollama model name
        ollama_endpoint:  Ollama server URL
        num_ctx:          Context window (8192 recommended for improved pipeline)
        temperature:      Lower = more conservative / faithful cleaning
    """
    text = (text or "").strip()
    if not text:
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
            logger.warning("Ollama not available — using regex fallback for cleaning.")
            return _regex_clean_fallback(text)

        # First attempt
        prompt = _USER_PROMPT.format(text=text[:4000])  # cap input to stay within context
        cleaned = llm.generate(prompt=prompt, system=_SYSTEM_PROMPT, temperature=temperature)
        cleaned = _postprocess(cleaned)

        # Retry if output is too short
        if not _is_valid_output(text, cleaned):
            logger.info("Retrying segment cleaning with stricter prompt...")
            retry_prompt = _RETRY_USER_PROMPT.format(text=text[:4000])
            cleaned = llm.generate(
                prompt=retry_prompt,
                system=_RETRY_SYSTEM_PROMPT,
                temperature=0.1,
            )
            cleaned = _postprocess(cleaned)

            # If still bad, fall back to regex
            if not _is_valid_output(text, cleaned, min_retention=0.4):
                logger.warning("LLM cleaning still inadequate — using regex fallback.")
                return _regex_clean_fallback(text)

        return cleaned

    except Exception as e:
        logger.warning("Segment cleaning error: %s — using regex fallback.", e)
        return _regex_clean_fallback(text)


def _regex_clean_fallback(text: str) -> str:
    """
    Light regex-based cleaning when Ollama is unavailable.
    Removes obvious fillers and repeated phrases only.
    """
    import re
    t = text
    # Remove noise tokens
    t = re.sub(r"\[.*?\]|\(.*?\)", " ", t)
    # Remove filler words
    t = re.sub(r"\b(uh+|um+|hmm+|hm+|ah+|you know,?\s*)\b", " ", t, flags=re.IGNORECASE)
    # Remove simple word repetitions "the the", "and and"
    t = re.sub(r"\b(\w+)\s+\1\b", r"\1", t, flags=re.IGNORECASE)
    # Normalise whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------------------------------------------------------------------------
# Batch cleaning
# ---------------------------------------------------------------------------

def clean_all_segments(
    topic_segments,
    ollama_model: str = "llama3.1",
    ollama_endpoint: str = "http://localhost:11434",
    num_ctx: int = 8192,
    temperature: float = 0.3,
    log_every: int = 3,
) -> list:
    """
    Clean all TopicSegment objects. Returns list of dicts with cleaned text.

    Each returned dict contains everything from the original TopicSegment
    plus "cleaned_text" and "original_text" fields.
    """
    results = []
    total = len(topic_segments)

    for i, seg in enumerate(topic_segments, 1):
        if i % log_every == 0 or i == total:
            logger.info("Cleaning segment %d/%d (words: %d)...", i, total, seg.word_count)

        cleaned = clean_segment(
            text=seg.text,
            ollama_model=ollama_model,
            ollama_endpoint=ollama_endpoint,
            num_ctx=num_ctx,
            temperature=temperature,
        )

        results.append({
            "topic_index":    seg.topic_index,
            "start":          seg.start,
            "end":            seg.end,
            "original_text":  seg.text,
            "cleaned_text":   cleaned,
            "original_words": seg.word_count,
            "cleaned_words":  len(cleaned.split()),
        })

    logger.info("Cleaned %d/%d segments.", len(results), total)
    return results