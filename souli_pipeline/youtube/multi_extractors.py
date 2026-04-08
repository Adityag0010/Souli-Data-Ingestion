"""
souli_pipeline/youtube/multi_extractors.py

Five specialized extractors — each reads the same cleaned transcript
and pulls out a specific type of teachable content.

Each extractor function:
  - Takes: transcript (str), energy_node (str), ollama config
  - Returns: List[Dict] — each dict is one chunk ready for Qdrant ingestion

Shared output format per chunk:
    {
        "text":             str,   # the extracted content
        "chunk_type":       str,   # "healing" | "activities" | "stories" | "commitment" | "patterns"
        "energy_node":      str,   # passed in from caller
        "problem_keywords": str,   # comma-separated keywords to help retrieval
        "source_video":     str,   # filled by pipeline orchestrator
        "youtube_url":      str,   # filled by pipeline orchestrator
    }

The shared helpers _run_extractor_prompt() and _parse_extractor_output()
do the heavy lifting — each extractor just supplies its own system + user prompt.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid chunk types
# ---------------------------------------------------------------------------

CHUNK_TYPES = ["healing", "activities", "stories", "commitment", "patterns"]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _run_extractor_prompt(
    transcript: str,
    system_prompt: str,
    user_prompt: str,
    ollama_model: str,
    ollama_endpoint: str,
    timeout_s: int = 180,
) -> str:
    """
    Call Ollama with the given prompts and return raw LLM output string.
    Raises on Ollama failure — caller handles the exception.
    """
    from souli_pipeline.llm.ollama import OllamaLLM

    llm = OllamaLLM(
        model=ollama_model,
        endpoint=ollama_endpoint,
        timeout_s=timeout_s,
        temperature=0.2,   # low — we want faithful extraction, not creative rewriting
        num_ctx=8192,
    )

    if not llm.is_available():
        raise RuntimeError("Ollama not available")

    return llm.generate(prompt=user_prompt, system=system_prompt)


def _parse_extractor_output(
    raw: str,
    chunk_type: str,
    energy_node: str,
) -> List[Dict]:
    """
    Parse LLM JSON output into a list of standard chunk dicts.

    LLM is asked to return a JSON array like:
    [
        {"text": "...", "problem_keywords": "guilt, avoidance"},
        {"text": "...", "problem_keywords": "stuck, fear"}
    ]

    Falls back to treating the whole raw output as one chunk if JSON fails.
    """
    raw = raw.strip()
    # Strip markdown fences
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    chunks = []

    # Try parsing as JSON array
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            for item in data:
                text = str(item.get("text", "")).strip()
                keywords = str(item.get("problem_keywords", "")).strip()
                if text and len(text.split()) >= 5:  # ignore trivially short extracts
                    chunks.append({
                        "text":             text,
                        "chunk_type":       chunk_type,
                        "energy_node":      energy_node,
                        "problem_keywords": keywords,
                        "source_video":     "",   # filled by orchestrator
                        "youtube_url":      "",   # filled by orchestrator
                    })
            return chunks
        elif isinstance(data, dict):
            # Sometimes LLM wraps in {"items": [...]} — handle it
            items = data.get("items", data.get("chunks", data.get("results", [])))
            if isinstance(items, list):
                return _parse_extractor_output(json.dumps(items), chunk_type, energy_node)
    except json.JSONDecodeError:
        # Try to find JSON array inside noisy text
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            try:
                return _parse_extractor_output(raw[start:end], chunk_type, energy_node)
            except Exception:
                pass

    # Last resort — treat entire output as one chunk
    if raw and len(raw.split()) >= 10:
        logger.warning(
            "[EXTRACTOR:%s] JSON parse failed — wrapping entire output as single chunk.", chunk_type
        )
        chunks.append({
            "text":             raw[:2000],  # cap at 2000 chars
            "chunk_type":       chunk_type,
            "energy_node":      energy_node,
            "problem_keywords": "",
            "source_video":     "",
            "youtube_url":      "",
        })

    return chunks


# ---------------------------------------------------------------------------
# Extractor 1 — Healing Principles
# ---------------------------------------------------------------------------

_HEALING_SYSTEM = """\
You are a content analyst for Souli, an inner wellness platform.
Your job is to extract healing principles from a coach's transcript.

A healing principle is: a belief, truth, or inner shift the coach says
resolves the person's energy blockage. It answers "what truth sets you free?"

Examples of healing principles:
- "You don't need to justify yourself to anyone."
- "Energy must move to heal — stillness keeps you stuck."
- "You were born worthy — worthiness is not earned."

Rules:
- Extract ONLY complete, standalone principles. Not fragments.
- Keep the coach's voice and language — do not paraphrase into generic wellness speak.
- If the coach says the same thing multiple times, extract it ONCE.
- Minimum 15 words per principle, maximum 80 words.
- Return ONLY valid JSON array, no other text.
"""

_HEALING_USER = """\
Extract all healing principles from this transcript.

Transcript:
\"\"\"
{transcript}
\"\"\"

Return a JSON array where each item has:
- "text": the healing principle in the coach's words
- "problem_keywords": 3-5 comma-separated keywords describing what inner problem this resolves

Example format:
[
  {{"text": "You don't need to justify your existence to anyone. Your joy is your purpose.", "problem_keywords": "guilt, self-worth, approval seeking"}},
  {{"text": "...", "problem_keywords": "..."}}
]

If no healing principles are found, return an empty array: []
"""


def extract_healing_principles(
    transcript: str,
    energy_node: str,
    ollama_model: str = "llama3.1",
    ollama_endpoint: str = "http://localhost:11434",
) -> List[Dict]:
    """Extract healing principles — truths the coach says resolve this energy pattern."""
    try:
        user_prompt = _HEALING_USER.format(transcript=transcript[:5000])
        raw = _run_extractor_prompt(transcript, _HEALING_SYSTEM, user_prompt, ollama_model, ollama_endpoint)
        chunks = _parse_extractor_output(raw, "healing", energy_node)
        logger.info("[EXTRACTOR:healing] Extracted %d chunks.", len(chunks))
        return chunks
    except Exception as exc:
        logger.warning("[EXTRACTOR:healing] Failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Extractor 2 — Activities
# ---------------------------------------------------------------------------

_ACTIVITIES_SYSTEM = """\
You are a content analyst for Souli, an inner wellness platform.
Your job is to extract specific practices and activities from a coach's transcript.

An activity is: a concrete, named practice the coach recommends — something
the person can actually DO. It must have a name AND some instruction or description.

Examples of activities:
- "2-minute shaking practice: stand up, start shaking your hands and arms, let it travel to your whole body."
- "OM meditation: sit quietly, close your eyes, chant OM for 5 minutes."
- "Grounding body scan: lie down, breathe slowly, feel each part of your body touching the ground."

Rules:
- Only extract if the coach gives the name AND at least one sentence of instruction.
- Do NOT extract vague mentions like "try meditation" with no detail.
- Keep the coach's voice — do not invent instructions not in the transcript.
- Return ONLY valid JSON array, no other text.
"""

_ACTIVITIES_USER = """\
Extract all specific practices and activities from this transcript.

Transcript:
\"\"\"
{transcript}
\"\"\"

Return a JSON array where each item has:
- "text": the activity name + full instructions as the coach described
- "problem_keywords": 3-5 comma-separated keywords for the energy state this helps

Example format:
[
  {{"text": "Shaking practice: stand and shake your hands, arms, then whole body for 2 minutes to release blocked energy.", "problem_keywords": "blocked energy, stuck, body tension"}},
  {{"text": "...", "problem_keywords": "..."}}
]

If no specific activities are found, return an empty array: []
"""


def extract_activities(
    transcript: str,
    energy_node: str,
    ollama_model: str = "llama3.1",
    ollama_endpoint: str = "http://localhost:11434",
) -> List[Dict]:
    """Extract specific practices and activities the coach recommends."""
    try:
        user_prompt = _ACTIVITIES_USER.format(transcript=transcript[:5000])
        raw = _run_extractor_prompt(transcript, _ACTIVITIES_SYSTEM, user_prompt, ollama_model, ollama_endpoint)
        chunks = _parse_extractor_output(raw, "activities", energy_node)
        logger.info("[EXTRACTOR:activities] Extracted %d chunks.", len(chunks))
        return chunks
    except Exception as exc:
        logger.warning("[EXTRACTOR:activities] Failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Extractor 3 — Stories and Phrases
# ---------------------------------------------------------------------------

_STORIES_SYSTEM = """\
You are a content analyst for Souli, an inner wellness platform.
Your job is to extract stories, metaphors, and signature phrases from a coach's transcript.

What to extract:
1. Real client stories — "I had a client who..." anecdotes (keep them short, anonymised)
2. Metaphors — vivid comparisons the coach uses ("your energy is like a river that needs to flow")
3. Signature phrases — memorable lines the coach repeats or emphasises

Why these matter: they make Souli's responses sound like the real coach,
not a generic AI. These are the coach's voice captured exactly.

Rules:
- Keep the coach's EXACT words as much as possible — this is about voice, not summary.
- For stories: extract the key narrative in 2-5 sentences maximum.
- For metaphors and phrases: keep them as spoken, verbatim.
- Return ONLY valid JSON array, no other text.
"""

_STORIES_USER = """\
Extract stories, metaphors, and signature phrases from this transcript.

Transcript:
\"\"\"
{transcript}
\"\"\"

Return a JSON array where each item has:
- "text": the story, metaphor, or phrase in the coach's words (as verbatim as possible)
- "problem_keywords": 3-5 comma-separated keywords for the situation this applies to

Example format:
[
  {{"text": "Think of your energy like a river. When it flows, it heals everything around it. When it's dammed up, it becomes stagnant and starts to rot.", "problem_keywords": "blocked energy, flow, stuck"}},
  {{"text": "I had a client who hadn't cried in 10 years. She thought she was strong. She was just frozen.", "problem_keywords": "emotional closure, blocked energy, numbness"}}
]

If no stories or memorable phrases are found, return an empty array: []
"""


def extract_stories_and_phrases(
    transcript: str,
    energy_node: str,
    ollama_model: str = "llama3.1",
    ollama_endpoint: str = "http://localhost:11434",
) -> List[Dict]:
    """Extract stories, metaphors and signature coach phrases."""
    try:
        user_prompt = _STORIES_USER.format(transcript=transcript[:5000])
        raw = _run_extractor_prompt(transcript, _STORIES_SYSTEM, user_prompt, ollama_model, ollama_endpoint)
        chunks = _parse_extractor_output(raw, "stories", energy_node)
        logger.info("[EXTRACTOR:stories] Extracted %d chunks.", len(chunks))
        return chunks
    except Exception as exc:
        logger.warning("[EXTRACTOR:stories] Failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Extractor 4 — Commitment Prompts
# ---------------------------------------------------------------------------

_COMMITMENT_SYSTEM = """\
You are a content analyst for Souli, an inner wellness platform.
Your job is to extract commitment questions and readiness challenges from a coach's transcript.

A commitment prompt is: a question or statement the coach uses to test or
invite the person's readiness to change — it challenges them to be honest
with themselves about whether they truly want healing.

Examples of commitment prompts:
- "Ask yourself honestly: do you want to feel better, or do you want to stay safe?"
- "Are you ready to feel discomfort in order to heal?"
- "The real question is not what happened to you — it's what are you going to do now?"

Rules:
- Only extract prompts that are genuinely challenging or reflective.
- Do not extract gentle supportive statements — only questions that push for honest self-examination.
- Keep the coach's exact words.
- Return ONLY valid JSON array, no other text.
"""

_COMMITMENT_USER = """\
Extract all commitment questions and readiness challenges from this transcript.

Transcript:
\"\"\"
{transcript}
\"\"\"

Return a JSON array where each item has:
- "text": the commitment question or challenge in the coach's words
- "problem_keywords": 3-5 comma-separated keywords for the energy state this challenges

Example format:
[
  {{"text": "Are you ready to choose yourself — even if it disappoints others? Because that is what healing requires.", "problem_keywords": "depleted energy, people pleasing, self sacrifice"}},
  {{"text": "...", "problem_keywords": "..."}}
]

If no commitment prompts are found, return an empty array: []
"""


def extract_commitment_prompts(
    transcript: str,
    energy_node: str,
    ollama_model: str = "llama3.1",
    ollama_endpoint: str = "http://localhost:11434",
) -> List[Dict]:
    """Extract readiness challenges and commitment questions the coach poses."""
    try:
        user_prompt = _COMMITMENT_USER.format(transcript=transcript[:5000])
        raw = _run_extractor_prompt(transcript, _COMMITMENT_SYSTEM, user_prompt, ollama_model, ollama_endpoint)
        chunks = _parse_extractor_output(raw, "commitment", energy_node)
        logger.info("[EXTRACTOR:commitment] Extracted %d chunks.", len(chunks))
        return chunks
    except Exception as exc:
        logger.warning("[EXTRACTOR:commitment] Failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Extractor 5 — Problem Patterns
# ---------------------------------------------------------------------------

_PATTERNS_SYSTEM = """\
You are a content analyst for Souli, an inner wellness platform.
Your job is to extract detailed descriptions of problem patterns from a coach's transcript.

A problem pattern description is: how the coach describes and names a specific
inner energy state — what it looks like, how it shows up in daily life, what
drives it, and how to recognise it in yourself.

Examples of problem pattern descriptions:
- "Blocked energy looks like this: you stop reaching out to people, you start
  living in your own cocoon. You procrastinate not because you're lazy — because
  you're protecting yourself from more pain."
- "Depleted energy is when your battery is at zero. You keep going because
  you have to, but there is nothing left inside."

Rules:
- Extract descriptions that would help someone recognise this pattern in themselves.
- Keep the coach's vocabulary and framing — this trains Souli to speak his language.
- Minimum 30 words per pattern description.
- Return ONLY valid JSON array, no other text.
"""

_PATTERNS_USER = """\
Extract all detailed problem pattern descriptions from this transcript.

Transcript:
\"\"\"
{transcript}
\"\"\"

Return a JSON array where each item has:
- "text": the coach's description of the problem pattern
- "problem_keywords": 3-5 comma-separated keywords that identify this pattern

Example format:
[
  {{"text": "Blocked energy looks like living in your own world, not reaching out, not asking for help. It is not weakness — it is self-protection taken too far.", "problem_keywords": "blocked energy, withdrawal, self isolation, procrastination"}},
  {{"text": "...", "problem_keywords": "..."}}
]

If no pattern descriptions are found, return an empty array: []
"""


def extract_problem_patterns(
    transcript: str,
    energy_node: str,
    ollama_model: str = "llama3.1",
    ollama_endpoint: str = "http://localhost:11434",
) -> List[Dict]:
    """Extract how the coach describes and names this energy problem pattern."""
    try:
        user_prompt = _PATTERNS_USER.format(transcript=transcript[:5000])
        raw = _run_extractor_prompt(transcript, _PATTERNS_SYSTEM, user_prompt, ollama_model, ollama_endpoint)
        chunks = _parse_extractor_output(raw, "patterns", energy_node)
        logger.info("[EXTRACTOR:patterns] Extracted %d chunks.", len(chunks))
        return chunks
    except Exception as exc:
        logger.warning("[EXTRACTOR:patterns] Failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Convenience — run all extractors based on density report
# ---------------------------------------------------------------------------

def run_extractors_from_density(
    transcript: str,
    energy_node: str,
    density_report: Dict,
    ollama_model: str = "llama3.1",
    ollama_endpoint: str = "http://localhost:11434",
) -> Dict[str, List[Dict]]:
    """
    Run only the extractors that the density report says are worth running.

    Args:
        transcript:      Full cleaned transcript
        energy_node:     Detected energy node for this video
        density_report:  Output of detect_content_density()
        ollama_model:    Ollama model name
        ollama_endpoint: Ollama server URL

    Returns:
        Dict mapping chunk_type → list of extracted chunk dicts
        e.g. {"healing": [...], "stories": [...], "activities": [], ...}
    """
    results: Dict[str, List[Dict]] = {ct: [] for ct in CHUNK_TYPES}

    extractor_map = {
        "healing":    (density_report.get("healing_rich"),    extract_healing_principles),
        "activities": (density_report.get("activity_rich"),   extract_activities),
        "stories":    (density_report.get("story_rich"),      extract_stories_and_phrases),
        "commitment": (density_report.get("commitment_rich"), extract_commitment_prompts),
        "patterns":   (density_report.get("pattern_rich"),    extract_problem_patterns),
    }

    for chunk_type, (is_rich, extractor_fn) in extractor_map.items():
        if is_rich:
            logger.info("[EXTRACTORS] Running %s extractor...", chunk_type)
            results[chunk_type] = extractor_fn(
                transcript=transcript,
                energy_node=energy_node,
                ollama_model=ollama_model,
                ollama_endpoint=ollama_endpoint,
            )
        else:
            logger.info("[EXTRACTORS] Skipping %s extractor (not rich enough).", chunk_type)

    total = sum(len(v) for v in results.values())
    logger.info("[EXTRACTORS] Done. Total chunks extracted: %d across %d types.", total, len(CHUNK_TYPES))
    return results