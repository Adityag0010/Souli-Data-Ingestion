"""
Speech-to-Text (STT) adapter.

Supports:
  - whisper  : local faster-whisper (no API key needed)
  - deepgram : Deepgram cloud STT (requires DEEPGRAM_API_KEY env var)

Default: whisper (fully local).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Whisper STT (local, via faster-whisper)
# ---------------------------------------------------------------------------

class WhisperSTT:
    """
    Transcribes raw PCM audio bytes using faster-whisper (local, no API key).
    """

    def __init__(self, model_name: str = "base"):
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self._model_name, device="cpu", compute_type="int8")
            logger.info("Loaded faster-whisper model: %s", self._model_name)
        return self._model

    def transcribe_file(self, audio_path: str, language: Optional[str] = None) -> str:
        """Transcribe an audio file. Returns transcript text."""
        model = self._load()
        kwargs = {}
        if language:
            kwargs["language"] = language
        segments, _ = model.transcribe(audio_path, beam_size=5, **kwargs)
        return " ".join(seg.text.strip() for seg in segments)

    def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """
        Transcribe raw PCM audio bytes (16-bit, mono).
        Saves to a temp file then transcribes.
        """
        import tempfile
        import wave

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)

        try:
            return self.transcribe_file(tmp_path)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Deepgram STT (cloud, requires API key)
# ---------------------------------------------------------------------------

class DeepgramSTT:
    """
    Transcribes audio via Deepgram cloud API.
    Requires: pip install deepgram-sdk
    Set env var DEEPGRAM_API_KEY.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "nova-2"):
        self.api_key = api_key or os.environ.get("DEEPGRAM_API_KEY", "")
        self.model = model
        if not self.api_key:
            logger.warning("DEEPGRAM_API_KEY not set — Deepgram STT will fail.")

    def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        from deepgram import DeepgramClient, PrerecordedOptions, FileSource
        dg = DeepgramClient(self.api_key)
        payload: FileSource = {"buffer": audio_bytes, "mimetype": "audio/wav"}
        opts = PrerecordedOptions(model=self.model, smart_format=True, language="en")
        response = dg.listen.prerecorded.v("1").transcribe_file(payload, opts)
        return response.results.channels[0].alternatives[0].transcript


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_stt(provider: str = "whisper", whisper_model: str = "base"):
    if provider == "whisper":
        return WhisperSTT(model_name=whisper_model)
    if provider == "deepgram":
        return DeepgramSTT()
    raise ValueError(f"Unknown STT provider: {provider}")
