"""
souli_pipeline/youtube/content_density_detector.py

Reads a full cleaned transcript and decides what types of extractable
content are present BEFORE running the 5 extractors.

This prevents wasting Ollama time running an "activities" extractor on a
video that is purely theoretical, or a "healing principles" extractor on
a video that is just a Q&A session.

Returns a density report dict:
    {
        "healing_rich":    bool,
        "activity_rich":   bool,
        "story_rich":      bool,
        "commitment_rich": bool,
        "pattern_rich":    bool,
        "dominant_node":   str,   # best-guess energy node for this video
    }

Usage:
    from souli_pipeline.youtube.content_density_detector import detect_content_density
    report = detect_content_density(transcript, energy_node="blocked_energy", ...)
    # report["healing_rich"] == True  → run healing extractor
    # report["activity_rich"] == False → skip activity extractor
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System + user prompts for LLM-based density detection
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a content analyst for the Souli inner wellness platform.
Your job is to read a coaching transcript and decide what types of
teachable content are present in significant quantity.

You must return ONLY valid JSON — no explanation, no preamble, no markdown.
"""

_USER_PROMPT = """\
Read the transcript below and decide which content types are present in
significant, extractable quantity (not just a passing mention).

Content types to check:
- healing_rich: Does the coach explain beliefs, truths, or principles about
  WHY a person feels stuck/depleted/scattered and WHAT inner shift resolves it?
  (e.g. "You don't need to justify yourself", "Energy must move to heal")

- activity_rich: Does the coach suggest specific practices, exercises,
  meditations, or physical activities by name with some instruction?
  (e.g. "Try 2 minutes of shaking", "Do OM meditation every morning")

- story_rich: Does the coach share real client stories, personal anecdotes,
  metaphors, or signature phrases that illustrate a point vividly?
  (e.g. "I had a client who...", "Think of your energy like a battery")

- commitment_rich: Does the coach ask reflection questions that test the
  person's readiness to change, or challenge their beliefs about themselves?
  (e.g. "Are you ready to feel discomfort to heal?", "Ask yourself — do you
  want comfort or do you want growth?")

- pattern_rich: Does the coach describe and name the problem pattern in
  detail — what it looks like, how it shows up, what drives it?
  (e.g. "Blocked energy looks like this: you withdraw, you avoid, you numb")

Set a type to true ONLY if there is enough content to extract at least
2-3 meaningful, standalone chunks of that type.

Also identify the dominant_node — the energy state this video is mainly about.
Pick ONE from: blocked_energy, depleted_energy, scattered_energy,
outofcontrol_energy, normal_energy.
If unclear, pick blocked_energy.

Transcript (first 6000 chars):
\"\"\"
{transcript}
\"\"\"

Return JSON with exactly these keys — boolean values only, no strings:
{{
    "healing_rich": true or false,
    "activity_rich": true or false,
    "story_rich": true or false,
    "commitment_rich": true or false,
    "pattern_rich": true or false,
    "dominant_node": "one_of_the_5_nodes"
}}
"""

# ---------------------------------------------------------------------------
# Keyword-based fallback (when Ollama is unavailable)
# ---------------------------------------------------------------------------

_HEALING_KW = [
    "you deserve", "born to be", "you don't need to", "truth is", "the key is",
    "real healing", "inner shift", "belief", "principle", "accept", "let go",
    "energy must", "soul is", "you are worthy", "you are enough",
]
_ACTIVITY_KW = [
    "meditation", "exercise", "practice", "try this", "do this",
    "dance", "breathing", "shaking", "grounding", "journaling", "yoga",
    "every morning", "every day", "minutes of", "step 1", "step 2",
]
_STORY_KW = [
    "i had a client", "one of my clients", "she told me", "he said",
    "real story", "for example", "imagine", "think of it like",
    "like a battery", "like a river", "metaphor", "i remember",
]
_COMMITMENT_KW = [
    "are you ready", "ask yourself", "ready to feel", "willing to",
    "commitment", "ready to choose", "do you want", "reflection",
    "honest with yourself", "are you willing",
]
_PATTERN_KW = [
    "this is what it looks like", "signs of", "typical sign", "withdrawal",
    "procrastination", "avoidance", "numbing", "this pattern", "you avoid",
    "you withdraw", "stuck in a loop", "this shows up as",
]

_NODE_KW: Dict[str, list] = {
    "blocked_energy":      ["stuck", "frozen", "numb", "blocked", "guilt", "shame", "procrastin", "avoid", "withdraw"],
    "depleted_energy":     ["exhausted", "drained", "tired", "burnout", "empty", "no energy", "victim", "giving up"],
    "scattered_energy":    ["overwhelm", "scattered", "too much", "unfocused", "anxious", "busy", "multitask"],
    "outofcontrol_energy": ["anger", "rage", "impulsive", "reactive", "explosive", "can't control", "obsessive"],
    "normal_energy":       ["growth", "ready to grow", "spiritual", "progression", "purpose", "fulfil"],
}


def _keyword_fallback_density(transcript: str) -> Dict:
    """Simple keyword count fallback when Ollama is unavailable."""
    t = transcript.lower()

    def is_rich(keywords: list, threshold: int = 3) -> bool:
        return sum(1 for kw in keywords if kw in t) >= threshold

    # Dominant node by keyword hit count
    node_scores = {
        node: sum(1 for kw in kws if kw in t)
        for node, kws in _NODE_KW.items()
    }
    dominant = max(node_scores, key=node_scores.get)
    # Default to blocked_energy if all scores are 0
    if node_scores[dominant] == 0:
        dominant = "blocked_energy"

    return {
        "healing_rich":    is_rich(_HEALING_KW),
        "activity_rich":   is_rich(_ACTIVITY_KW),
        "story_rich":      is_rich(_STORY_KW),
        "commitment_rich": is_rich(_COMMITMENT_KW),
        "pattern_rich":    is_rich(_PATTERN_KW),
        "dominant_node":   dominant,
    }


def _parse_llm_json(raw: str) -> Optional[Dict]:
    """Try to extract valid JSON from LLM output."""
    raw = raw.strip()
    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON block inside noisy output
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
    return None


def _validate_report(data: Dict) -> Dict:
    """
    Ensure all required keys exist and have correct types.
    Falls back to True for bool fields if missing (safer — better to extract too much).
    """
    valid_nodes = {
        "blocked_energy", "depleted_energy", "scattered_energy",
        "outofcontrol_energy", "normal_energy",
    }
    bool_keys = ["healing_rich", "activity_rich", "story_rich", "commitment_rich", "pattern_rich"]
    result = {}
    for k in bool_keys:
        val = data.get(k, True)  # default True = safe fallback (run extractor)
        result[k] = bool(val) if not isinstance(val, bool) else val

    node = str(data.get("dominant_node", "blocked_energy")).strip()
    result["dominant_node"] = node if node in valid_nodes else "blocked_energy"
    return result


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def detect_content_density(
    transcript: str,
    energy_node: Optional[str] = None,
    ollama_model: str = "llama3.1",
    ollama_endpoint: str = "http://localhost:11434",
    timeout_s: int = 60,
) -> Dict:
    """
    Analyse a cleaned transcript and return a density report.

    Args:
        transcript:       Full cleaned transcript text (will be capped at 6000 chars for LLM)
        energy_node:      Already-detected node for this video (used as hint, optional)
        ollama_model:     Ollama model to use (llama3.1 recommended)
        ollama_endpoint:  Ollama server URL
        timeout_s:        Seconds before giving up on LLM call

    Returns:
        Dict with keys: healing_rich, activity_rich, story_rich,
                        commitment_rich, pattern_rich, dominant_node
    """
    transcript = (transcript or "").strip()
    if not transcript:
        logger.warning("[DENSITY] Empty transcript — returning all-False density report.")
        return {
            "healing_rich": False, "activity_rich": False, "story_rich": False,
            "commitment_rich": False, "pattern_rich": False,
            "dominant_node": energy_node or "blocked_energy",
        }

    try:
        from souli_pipeline.llm.ollama import OllamaLLM

        llm = OllamaLLM(
            model=ollama_model,
            endpoint=ollama_endpoint,
            timeout_s=timeout_s,
            temperature=0.1,   # very low — we want deterministic classification
            num_ctx=8192,
        )

        if not llm.is_available():
            logger.warning("[DENSITY] Ollama unavailable — using keyword fallback.")
            report = _keyword_fallback_density(transcript)
            if energy_node:
                report["dominant_node"] = energy_node
            return report

        prompt = _USER_PROMPT.format(transcript=transcript[:6000])
        raw = llm.generate(prompt=prompt, system=_SYSTEM_PROMPT, format="json")
        data = _parse_llm_json(raw)

        if not data:
            logger.warning("[DENSITY] LLM returned unparseable output — using keyword fallback.")
            report = _keyword_fallback_density(transcript)
            if energy_node:
                report["dominant_node"] = energy_node
            return report

        report = _validate_report(data)

        # If caller already knows the energy node, honour it over LLM guess
        if energy_node:
            report["dominant_node"] = energy_node

        logger.info(
            "[DENSITY] Report: healing=%s activity=%s story=%s commitment=%s pattern=%s node=%s",
            report["healing_rich"], report["activity_rich"], report["story_rich"],
            report["commitment_rich"], report["pattern_rich"], report["dominant_node"],
        )
        return report

    except Exception as exc:
        logger.warning("[DENSITY] Detection failed (%s) — keyword fallback.", exc)
        report = _keyword_fallback_density(transcript)
        if energy_node:
            report["dominant_node"] = energy_node
        return report