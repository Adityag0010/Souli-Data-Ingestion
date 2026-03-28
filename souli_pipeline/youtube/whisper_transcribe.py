"""
souli_pipeline/youtube/whisper_transcribe.py

Dedicated Whisper transcription for the improved pipeline.
This is the PRIMARY transcript source (not a fallback).

Differences from whisper_fallback.py:
  - Returns richer segment dicts (text, start, end, avg_logprob, no_speech_prob)
  - Filters low-confidence segments automatically
  - Cleans [Music], [Applause] and similar noise tokens
  - Saves audio file to out_dir instead of cwd
  - Never raises — returns [] with a warning on failure
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from typing import Dict, List

logger = logging.getLogger(__name__)

# Noise tokens injected by Whisper for non-speech audio events
_NOISE_TOKEN_RE = re.compile(r"\[.*?\]|\(.*?\)", re.IGNORECASE)
# Filler words to strip
_FILLERS = re.compile(
    r"\b(uh+|um+|hmm+|hm+|ah+|er+|like,?\s+I\s+said)\b", re.IGNORECASE
)


def _clean_segment_text(text: str) -> str:
    """Remove noise tokens, filler words, and normalise whitespace."""
    t = _NOISE_TOKEN_RE.sub(" ", text)
    t = _FILLERS.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _download_audio(url: str, out_path: str) -> bool:
    """
    Download best audio track from YouTube URL to out_path.
    Returns True on success.
    """
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "-f", "bestaudio/best",
                "--no-playlist",
                "-o", out_path,
                url,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            logger.warning("yt-dlp failed: %s", result.stderr[:300])
            return False
        return os.path.exists(out_path)
    except Exception as e:
        logger.warning("Audio download error: %s", e)
        return False


def transcribe_url(
    url: str,
    out_dir: str,
    whisper_model: str = "medium",
    language: str = "en",
    min_words: int = 3,
    min_confidence: float = -1.0,
) -> List[Dict]:
    """
    Download audio from YouTube URL and transcribe with faster-whisper.

    Returns list of segment dicts:
        {
            "start": float,       # seconds
            "end":   float,       # seconds
            "text":  str,         # cleaned text
            "confidence": float,  # avg_logprob (higher = more confident)
        }

    Returns [] on any failure (never raises).

    Args:
        url:              YouTube video URL
        out_dir:          Directory to save the audio file
        whisper_model:    faster-whisper model size ("base", "medium", "large-v2")
        language:         Language hint — "en" for English, None for auto-detect
        min_words:        Drop segments shorter than this many words
        min_confidence:   Drop segments with avg_logprob below this (-1.0 = keep all)
    """
    os.makedirs(out_dir, exist_ok=True)
    audio_path = os.path.join(out_dir, "audio.m4a")

    # Remove stale audio if present
    if os.path.exists(audio_path):
        os.remove(audio_path)

    logger.info("Downloading audio from: %s", url)
    if not _download_audio(url, audio_path):
        logger.warning("Could not download audio for %s — returning empty.", url)
        return []

    logger.info("Transcribing with faster-whisper (%s)...", whisper_model)
    try:
        from faster_whisper import WhisperModel

        # device="auto" uses GPU if available, otherwise CPU
        model = WhisperModel(whisper_model, device="auto", compute_type="auto")

        kwargs: Dict = {"vad_filter": True}
        if language:
            kwargs["language"] = language

        segments_iter, info = model.transcribe(audio_path, **kwargs)
        segments_list = list(segments_iter)

        logger.info(
            "Whisper detected language '%s' (prob %.2f), duration %.1fs",
            info.language,
            info.language_probability,
            info.duration,
        )

    except Exception as e:
        logger.warning("Whisper transcription failed: %s", e)
        return []

    # Build clean output
    out: List[Dict] = []
    for seg in segments_list:
        raw_text = (seg.text or "").strip()
        if not raw_text:
            continue

        # Confidence filter
        confidence = float(getattr(seg, "avg_logprob", 0.0))
        if confidence < min_confidence:
            logger.debug("Dropped low-confidence segment (%.2f): %s", confidence, raw_text[:60])
            continue

        # Skip pure noise segments (entire text was [Music] etc.)
        no_speech = float(getattr(seg, "no_speech_prob", 0.0))
        if no_speech > 0.85:
            logger.debug("Dropped no-speech segment: %s", raw_text[:60])
            continue

        cleaned = _clean_segment_text(raw_text)
        if not cleaned:
            continue

        word_count = len(cleaned.split())
        if word_count < min_words:
            continue

        out.append(
            {
                "start":      round(float(seg.start), 3),
                "end":        round(float(seg.end), 3),
                "text":       cleaned,
                "confidence": round(confidence, 4),
            }
        )

    logger.info("Whisper produced %d clean segments.", len(out))
    return out