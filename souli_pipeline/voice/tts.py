"""
Text-to-Speech (TTS) adapter.

Supports:
  - edge_tts  : Microsoft Edge TTS (free, no API key, async)
  - piper     : Local Piper TTS (fully offline)
  - kokoro    : Kokoro TTS (local, high quality)

Default: edge_tts with Indian English voice (en-IN-NeerjaNeural).
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Edge TTS (Microsoft, free, online required)
# ---------------------------------------------------------------------------

class EdgeTTS:
    """
    Uses edge-tts (pip install edge-tts).
    Voices: en-IN-NeerjaNeural (Indian English female),
            en-US-JennyNeural, hi-IN-SwaraNeural (Hindi)
    """

    def __init__(self, voice: str = "en-IN-NeerjaNeural", rate: str = "+0%"):
        self.voice = voice
        self.rate = rate

    async def synthesize_async(self, text: str) -> bytes:
        """Synthesize text → WAV bytes."""
        import edge_tts

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
            await communicate.save(tmp_path)
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def synthesize(self, text: str) -> bytes:
        """Synchronous wrapper for synthesize_async."""
        return asyncio.run(self.synthesize_async(text))

    async def stream_async(self, text: str) -> AsyncGenerator[bytes, None]:
        """Stream audio chunks as they're generated."""
        import edge_tts

        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]


# ---------------------------------------------------------------------------
# Piper TTS (fully local, requires piper binary)
# ---------------------------------------------------------------------------

class PiperTTS:
    """
    Local TTS via Piper (https://github.com/rhasspy/piper).
    Requires: piper binary on PATH + voice model downloaded.
    """

    def __init__(self, model_path: str, piper_binary: str = "piper"):
        self.model_path = model_path
        self.piper_binary = piper_binary

    def synthesize(self, text: str) -> bytes:
        import subprocess

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd = [
                self.piper_binary,
                "--model", self.model_path,
                "--output_file", tmp_path,
            ]
            proc = subprocess.run(
                cmd,
                input=text.encode(),
                capture_output=True,
                timeout=30,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"Piper error: {proc.stderr.decode()}")
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_tts(provider: str = "edge_tts", voice: str = "en-IN-NeerjaNeural", **kwargs):
    if provider == "edge_tts":
        return EdgeTTS(voice=voice)
    if provider == "piper":
        model_path = kwargs.get("model_path", "")
        if not model_path:
            raise ValueError("piper TTS requires model_path kwarg")
        return PiperTTS(model_path=model_path)
    raise ValueError(f"Unknown TTS provider: {provider}")
