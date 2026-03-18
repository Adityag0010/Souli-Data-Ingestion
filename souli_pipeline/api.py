"""
Souli REST API — FastAPI server for mobile app integration
=========================================================

Endpoints:
  POST /chat              — text message → text reply
  POST /voice             — audio file upload → text reply + audio bytes
  POST /session/reset     — reset conversation state for a session
  GET  /session/{id}/state — get current phase, energy_node, turn count
  GET  /health            — check if API + Ollama + Qdrant are all up

In Docker (GCP):
  "gunicorn", "souli_pipeline.api:app",
          "--worker-class", "uvicorn.workers.UvicornWorker",
          "--workers", "3",
          "--bind", "0.0.0.0:8000",
          "--timeout", "120"
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config paths (same env vars as streamlit_app.py) ─────────────────────────

CONFIG_PATH = os.environ.get(
    "SOULI_CONFIG_PATH",
    str(Path(__file__).parent.parent / "configs" / "pipeline.gcp.yaml"),
)
GOLD_PATH = os.environ.get("SOULI_GOLD_PATH", None)
_default_excel = str(Path(__file__).parent / "data" / "Souli_EnergyFramework_PW (1).xlsx")
EXCEL_PATH = os.environ.get(
    "SOULI_EXCEL_PATH",
    _default_excel if os.path.exists(_default_excel) else None,
)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Souli API",
    description="REST API for Souli wellness companion — connects chat and voice to your mobile app",
    version="1.0.0",
)

# Allow any origin so the mobile app (React Native / Flutter) can call freely.
# Tighten this to your domain in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session store ─────────────────────────────────────────────────────────────
# Maps session_id (string) → ConversationEngine instance
# Each user/device should use a unique session_id (UUID from the mobile app).
# Simple in-memory store — good for single-server GCP VM.
# If you scale to multiple replicas later, replace with Redis.

_sessions: Dict[str, object] = {}


def _get_or_create_engine(session_id: str):
    """Return existing engine for session or create a fresh one."""
    if session_id not in _sessions:
        logger.info("Creating new engine for session: %s", session_id)
        from souli_pipeline.config_loader import load_config
        from souli_pipeline.conversation.engine import ConversationEngine

        cfg = load_config(CONFIG_PATH)

        # Auto-find newest gold.xlsx if not set via env var
        gold_path = GOLD_PATH
        if not gold_path and os.path.exists("outputs"):
            runs = sorted(
                [r for r in os.listdir("outputs") if os.path.isdir(os.path.join("outputs", r))],
                reverse=True,
            )
            for r in runs:
                gp = os.path.join("outputs", r, "energy", "gold.xlsx")
                if os.path.exists(gp):
                    gold_path = gp
                    break

        _sessions[session_id] = ConversationEngine.from_config(
            cfg,
            gold_path=gold_path,
            excel_path=EXCEL_PATH,
        )

    return _sessions[session_id]


# ── Cached STT / TTS (loaded once per process) ───────────────────────────────

_stt = None
_tts = None


def _get_stt():
    global _stt
    if _stt is None:
        from souli_pipeline.voice.stt import WhisperSTT
        _stt = WhisperSTT(model_name=os.environ.get("WHISPER_MODEL", "base"))
    return _stt


def _get_tts():
    global _tts
    if _tts is None:
        from souli_pipeline.voice.tts import EdgeTTS
        _tts = EdgeTTS(voice=os.environ.get("TTS_VOICE", "en-IN-NeerjaNeural"))
    return _tts


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "user-abc-123",
                "message": "I've been feeling really overwhelmed lately",
            }
        }


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    phase: str
    energy_node: Optional[str]
    turn_count: int


class SessionState(BaseModel):
    session_id: str
    phase: str
    energy_node: Optional[str]
    turn_count: int
    intent: Optional[str]
    user_name: Optional[str]


class ResetResponse(BaseModel):
    session_id: str
    status: str
    greeting: str


class HealthResponse(BaseModel):
    status: str
    ollama: str
    qdrant: str
    config_loaded: bool


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

# ── 1. Text Chat ──────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, summary="Send a text message to Souli")
def chat(req: ChatRequest):
    """
    Send a text message and get Souli's response.

    - **session_id**: unique identifier for this user's conversation (e.g. device UUID)
    - **message**: what the user typed

    Returns the reply text plus current conversation state metadata.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    engine = _get_or_create_engine(req.session_id)

    try:
        reply = engine.turn(req.message)
    except Exception as exc:
        logger.error("Engine error for session %s: %s", req.session_id, exc)
        raise HTTPException(status_code=500, detail=f"Engine error: {exc}")

    diag = engine.diagnosis_summary
    return ChatResponse(
        session_id=req.session_id,
        reply=reply,
        phase=engine.state.phase,
        energy_node=diag.get("energy_node"),
        turn_count=engine.state.turn_count,
    )




def _safe_header(text: str) -> str:
    """Strip non-latin-1 chars so HTTP headers don't blow up on em-dashes, smart quotes etc."""
    return text.encode("latin-1", errors="replace").decode("latin-1")

# ── 2. Voice Chat ─────────────────────────────────────────────────────────────

@app.post(
    "/voice",
    summary="Send a voice recording, get back text + audio",
    responses={
        200: {
            "content": {"audio/mpeg": {}},
            "description": "MP3 audio of Souli's spoken reply",
        }
    },
)
async def voice(
    session_id: str = Form(..., description="Unique session ID for this user"),
    audio: UploadFile = File(..., description="Audio recording (.wav, .mp3, .webm, .m4a)"),
):
    """
    Upload a voice recording. Returns:
    - `X-Transcript` header — what Souli heard you say
    - `X-Reply` header — Souli's text reply
    - `X-Phase` header — current conversation phase
    - Response body — MP3 audio of Souli's spoken reply

    The mobile app should:
    1. POST the audio file with form-data
    2. Read the headers for text display
    3. Play the response body as audio
    """
    # Save uploaded audio to a temp file
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        # STT: transcribe audio
        stt = _get_stt()
        transcript = stt.transcribe_file(tmp_path)
    except Exception as exc:
        os.unlink(tmp_path)
        logger.error("STT error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not transcript.strip():
        raise HTTPException(status_code=422, detail="Could not transcribe audio — please try again")

    # Conversation engine turn
    engine = _get_or_create_engine(session_id)
    try:
        reply = engine.turn(transcript)
    except Exception as exc:
        logger.error("Engine error for session %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail=f"Engine error: {exc}")

    # TTS: synthesize reply to audio
    try:
        tts = _get_tts()
        audio_bytes = tts.synthesize(reply)
    except Exception as exc:
        logger.error("TTS error: %s", exc)
        # Don't fail the whole request — return the text in headers even if TTS breaks
        audio_bytes = b""

    diag = engine.diagnosis_summary
    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={
            "X-Transcript": _safe_header(transcript),
            "X-Reply": _safe_header(reply),
            "X-Phase": engine.state.phase,
            "X-Energy-Node": _safe_header(diag.get("energy_node") or ""),
            "X-Turn-Count": str(engine.state.turn_count),
            "Access-Control-Expose-Headers": (
                "X-Transcript, X-Reply, X-Phase, X-Energy-Node, X-Turn-Count"
            ),
        },
    )


# ── 3. Reset Session ──────────────────────────────────────────────────────────

@app.post("/session/reset", response_model=ResetResponse, summary="Start a fresh conversation")
def reset_session(session_id: str = Form(...)):
    """
    Reset the conversation state for a session.
    Call this when the user taps "New Session" in the mobile app.
    Returns the greeting message so the app can display it immediately.
    """
    if session_id in _sessions:
        _sessions[session_id].reset()
        engine = _sessions[session_id]
    else:
        engine = _get_or_create_engine(session_id)

    greeting = engine.greeting()
    return ResetResponse(
        session_id=session_id,
        status="reset",
        greeting=greeting,
    )


# ── 4. Session State ──────────────────────────────────────────────────────────

@app.get("/session/{session_id}/state", response_model=SessionState, summary="Get current session state")
def get_session_state(session_id: str):
    """
    Get the current conversation state for a session.
    Useful for the mobile app to show progress indicators or debug.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    engine = _sessions[session_id]
    diag = engine.diagnosis_summary
    return SessionState(
        session_id=session_id,
        phase=engine.state.phase,
        energy_node=diag.get("energy_node"),
        turn_count=engine.state.turn_count,
        intent=engine.state.intent,
        user_name=engine.state.user_name,
    )


# ── 5. Health Check ───────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, summary="Check API + service health")
def health():
    """
    Check the health of the API and its dependencies.
    The mobile app or load balancer can poll this endpoint.
    Returns 200 if all services are up, 503 if something is down.
    """
    ollama_status = "unknown"
    qdrant_status = "unknown"
    config_loaded = False

    # Check config
    try:
        from souli_pipeline.config_loader import load_config
        load_config(CONFIG_PATH)
        config_loaded = True
    except Exception as exc:
        logger.warning("Config load failed: %s", exc)

    # Check Ollama
    try:
        from souli_pipeline.llm.ollama import OllamaLLM
        ollama_endpoint = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")
        llm = OllamaLLM(endpoint=ollama_endpoint)
        ollama_status = "ok" if llm.is_available() else "unreachable"
    except Exception:
        ollama_status = "error"

    # Check Qdrant
    try:
        from qdrant_client import QdrantClient
        from souli_pipeline.config_loader import load_config
        cfg = load_config(CONFIG_PATH)
        qc = QdrantClient(
            host=cfg.retrieval.qdrant_host,
            port=cfg.retrieval.qdrant_port,
            timeout=3,
        )
        qc.get_collections()
        qdrant_status = "ok"
    except Exception:
        qdrant_status = "unreachable"

    overall = "ok" if (ollama_status == "ok" and qdrant_status == "ok") else "degraded"

    response_status = 200 if overall == "ok" else 503
    return Response(
        content=HealthResponse(
            status=overall,
            ollama=ollama_status,
            qdrant=qdrant_status,
            config_loaded=config_loaded,
        ).model_dump_json(),
        media_type="application/json",
        status_code=response_status,
    )


# ── 6. Greeting (convenience for first-open in mobile app) ───────────────────

@app.post("/session/greeting", response_model=ChatResponse, summary="Get opening greeting for a new session")
def greeting(session_id: str = Form(...)):
    """
    Get Souli's opening greeting for a brand new session.
    Call this when the user opens the app for the first time or after a reset.
    """
    engine = _get_or_create_engine(session_id)
    greet_text = engine.greeting()
    return ChatResponse(
        session_id=session_id,
        reply=greet_text,
        phase=engine.state.phase,
        energy_node=None,
        turn_count=0,
    )
