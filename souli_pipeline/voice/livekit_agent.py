"""
Souli Voice Agent — LiveKit-based real-time voice pipeline.

Flow:
    User speaks → STT (Whisper/Deepgram)
              → ConversationEngine (Ollama llama3.1 + Qdrant RAG)
              → TTS (Edge TTS / Piper)
              → Audio played back to user via LiveKit room

Requirements:
    pip install livekit livekit-agents edge-tts faster-whisper

Usage:
    agent = SouliVoiceAgent(cfg, gold_path="outputs/.../gold.xlsx")
    asyncio.run(agent.run())
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class SouliVoiceAgent:
    """
    LiveKit voice agent for Souli.
    Connects to a LiveKit room and handles real-time voice conversation.
    """

    def __init__(
        self,
        cfg,
        gold_path: Optional[str] = None,
        excel_path: Optional[str] = None,
    ):
        self.cfg = cfg
        self.gold_path = gold_path
        self.excel_path = excel_path
        self._engine = None
        self._stt = None
        self._tts = None

    def _get_engine(self):
        if self._engine is None:
            from ..conversation.engine import ConversationEngine
            self._engine = ConversationEngine.from_config(
                self.cfg,
                gold_path=self.gold_path,
                excel_path=self.excel_path,
            )
        return self._engine

    def _get_stt(self):
        if self._stt is None:
            from .stt import make_stt
            v = self.cfg.voice
            self._stt = make_stt(provider=v.stt_provider, whisper_model=v.whisper_model)
        return self._stt

    def _get_tts(self):
        if self._tts is None:
            from .tts import make_tts
            v = self.cfg.voice
            self._tts = make_tts(provider=v.tts_provider, voice=v.tts_voice)
        return self._tts

    # ------------------------------------------------------------------
    # LiveKit agent entry point (new livekit-agents SDK style)
    # ------------------------------------------------------------------

    async def run(self):
        """
        Main entry point. Connects to LiveKit room and starts the voice loop.
        Requires livekit-agents >= 0.8
        """
        try:
            from livekit.agents import WorkerOptions, cli
            from livekit.agents import AutoSubscribe, JobContext
            from livekit import rtc
        except ImportError:
            logger.error(
                "livekit-agents not installed. "
                "Run: pip install livekit livekit-agents"
            )
            raise

        v = self.cfg.voice

        async def entrypoint(ctx: JobContext):
            logger.info("Agent joined room: %s", ctx.room.name)
            engine = self._get_engine()
            stt = self._get_stt()
            tts = self._get_tts()

            # Send greeting
            greeting = engine.greeting()
            logger.info("Greeting: %s", greeting)
            audio_bytes = tts.synthesize(greeting)
            await _publish_audio(ctx.room, audio_bytes)

            # Subscribe to audio track from participants
            await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

            @ctx.room.on("track_subscribed")
            def on_track(track, publication, participant):
                if track.kind == rtc.TrackKind.KIND_AUDIO:
                    asyncio.ensure_future(
                        _handle_audio_track(track, engine, stt, tts, ctx.room)
                    )

        worker_opts = WorkerOptions(
            entrypoint_fnc=entrypoint,
            api_key=v.livekit_api_key,
            api_secret=v.livekit_api_secret,
            ws_url=v.livekit_url,
        )
        cli.run_app(worker_opts)

    # ------------------------------------------------------------------
    # Standalone mode (without LiveKit room — for testing voice locally)
    # ------------------------------------------------------------------

    async def run_local_voice(self):
        """
        Local test mode: uses microphone input via sounddevice + plays back audio.
        No LiveKit server needed.
        """
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            logger.error("sounddevice not installed. Run: pip install sounddevice")
            raise

        engine = self._get_engine()
        stt = self._get_stt()
        tts = self._get_tts()

        print("\n🎙️  Souli Voice (local mode). Press Ctrl+C to stop.\n")
        greeting = engine.greeting()
        print(f"Souli: {greeting}\n")
        audio_bytes = tts.synthesize(greeting)
        _play_audio_bytes(audio_bytes)

        while True:
            print("Listening... (speak now, 5 seconds)")
            sample_rate = 16000
            duration = 5
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
            )
            sd.wait()
            audio_bytes_raw = recording.tobytes()

            user_text = stt.transcribe_bytes(audio_bytes_raw, sample_rate=sample_rate)
            if not user_text.strip():
                print("(No speech detected, try again)\n")
                continue

            print(f"You: {user_text}")
            response = engine.turn(user_text)
            print(f"Souli: {response}\n")
            audio_bytes = tts.synthesize(response)
            _play_audio_bytes(audio_bytes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _handle_audio_track(track, engine, stt, tts, room):
    """Continuously read audio frames from a LiveKit track and respond."""
    from livekit import rtc

    audio_buffer = bytearray()
    sample_rate = 16000
    silence_threshold = 500  # frames of silence before processing
    silence_count = 0
    is_speaking = False

    async for frame_event in track:
        if not isinstance(frame_event, rtc.AudioFrame):
            continue

        frame: rtc.AudioFrame = frame_event
        pcm_data = bytes(frame.data)
        audio_buffer.extend(pcm_data)

        # Simple VAD: check RMS of frame
        import struct
        samples = struct.unpack(f"{len(pcm_data)//2}h", pcm_data)
        rms = (sum(s * s for s in samples) / max(1, len(samples))) ** 0.5

        if rms > 300:
            is_speaking = True
            silence_count = 0
        elif is_speaking:
            silence_count += 1
            if silence_count >= silence_threshold:
                # End of utterance — transcribe and respond
                user_text = stt.transcribe_bytes(bytes(audio_buffer), sample_rate=sample_rate)
                audio_buffer.clear()
                is_speaking = False
                silence_count = 0

                if user_text.strip():
                    logger.info("User said: %s", user_text)
                    response = engine.turn(user_text)
                    logger.info("Souli: %s", response)
                    audio_bytes = tts.synthesize(response)
                    await _publish_audio(room, audio_bytes)


async def _publish_audio(room, audio_bytes: bytes):
    """Encode and publish audio bytes to the LiveKit room."""
    try:
        from livekit import rtc
        import io

        # Publish as audio track
        source = rtc.AudioSource(sample_rate=24000, num_channels=1)
        track = rtc.LocalAudioTrack.create_audio_track("souli-voice", source)
        opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await room.local_participant.publish_track(track, opts)

        # Feed audio bytes to source (simplified — assumes MP3 decoded)
        # In production, decode MP3 → PCM using pydub or av
        await source.capture_frame(
            rtc.AudioFrame(
                data=audio_bytes,
                sample_rate=24000,
                num_channels=1,
                samples_per_channel=len(audio_bytes) // 2,
            )
        )
    except Exception as exc:
        logger.warning("Could not publish audio to LiveKit: %s", exc)


def _play_audio_bytes(audio_bytes: bytes):
    """Play audio bytes locally using system player (for local mode)."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # Try afplay (macOS), then mpg123 (Linux)
        if os.path.exists("/usr/bin/afplay"):
            subprocess.run(["afplay", tmp_path], check=False)
        else:
            subprocess.run(["mpg123", "-q", tmp_path], check=False)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
