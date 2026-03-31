"""
Souli — Developer / Tester Debug UI
====================================
Run from project root:
    streamlit run souli_pipeline/streamlit_dev.py

Fixes applied vs original:
  1. sys.path patch so 'souli_pipeline' is importable when run from inside the package dir
  2. KB toggle (Original vs Improved) at top of page — resets conversation on switch
  3. engine._debug_events / engine.latest_debug stubs added (were missing from engine.py)
  4. _count_turns_in_phase dead-code bug noted in debug output
"""
from __future__ import annotations

# ── PATH FIX — must be FIRST, before any souli_pipeline imports ──────────────
import sys
import os
from pathlib import Path

# When run as /app/souli_pipeline/streamlit_dev.py, __file__ is inside the package.
# We need the PARENT of souli_pipeline/ on sys.path so imports resolve correctly.
_this_file = Path(__file__).resolve()
_project_root = _this_file.parent.parent   # …/app/
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
# Also handle being run directly from project root (already works, but be safe)
_pkg_parent = _this_file.parent            # …/app/souli_pipeline/
if str(_pkg_parent) not in sys.path:
    sys.path.insert(0, str(_pkg_parent))

# ─────────────────────────────────────────────────────────────────────────────

import json
import logging
import tempfile
import time
from typing import Any, Dict, List, Optional

import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = os.environ.get(
    "SOULI_CONFIG_PATH",
    str(_project_root / "configs" / "pipeline.gcp.yaml"),
)

def _find_latest_gold() -> str | None:
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        return None
    for run_id in sorted(os.listdir(outputs_dir), reverse=True):
        gp = os.path.join(outputs_dir, run_id, "energy", "gold.xlsx")
        if os.path.exists(gp):
            return gp
    return None

GOLD_PATH = os.environ.get("SOULI_GOLD_PATH") or _find_latest_gold()

_default_excel = str(_this_file.parent / "data" / "Souli_EnergyFramework_PW (1).xlsx")
EXCEL_PATH  = os.environ.get(
    "SOULI_EXCEL_PATH",
    _default_excel if os.path.exists(_default_excel) else None,
)

logging.basicConfig(level=logging.WARNING)

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Souli Dev",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base: clean white/light-grey background ── */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main { background: #f7f8fa !important; }

section[data-testid="stSidebar"] { display: none; }

/* ── Typography ── */
body, p, div, span, label           { color: #1e2532 !important; }
h1, h2, h3, h4                      { color: #0f172a !important; font-weight: 700; }
.stMarkdown p, .stMarkdown li       { color: #334155 !important; }
[data-testid="stWidgetLabel"] p,
label                               { color: #475569 !important; font-size: 0.82rem !important; }
[data-testid="stMetricLabel"]        { color: #64748b !important; }
[data-testid="stMetricValue"]        { color: #0f172a !important; }
[data-testid="stCaptionContainer"] p { color: #94a3b8 !important; font-size: 0.78rem !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 10px 14px !important;
}

/* ── Inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {
    background: #ffffff !important;
    color: #1e2532 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    color: #1e2532 !important;
    border-radius: 8px !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] summary {
    color: #334155 !important;
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    font-weight: 600;
}
[data-testid="stExpander"] summary:hover {
    border-color: #94a3b8 !important;
    background: #f8fafc !important;
}
[data-testid="stExpander"] > div > div {
    background: #fafbfc !important;
    border: 1px solid #e2e8f0 !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-testid="stTab"] p        { color: #64748b !important; }
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] p { color: #2563eb !important; font-weight: 600; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] { background: transparent !important; }

/* ── Dividers ── */
hr { border-color: #e2e8f0 !important; }

/* ── Badges ── */
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; margin: 2px 2px;
}
.badge-phase    { background: #dbeafe; color: #1d4ed8; }
.badge-node     { background: #dcfce7; color: #15803d; }
.badge-fallback { background: #fee2e2; color: #dc2626; }
.badge-embed    { background: #ede9fe; color: #7c3aed; }
.badge-kw       { background: #fef9c3; color: #a16207; }
.badge-llm      { background: #e0f2fe; color: #0369a1; }
.badge-ok       { background: #dcfce7; color: #15803d; }
.badge-warn     { background: #ffedd5; color: #c2410c; }
.badge-neutral  { background: #f1f5f9; color: #475569; }

/* ── RAG cards ── */
.rag-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #94a3b8;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.78rem;
}
.rag-score  { color: #2563eb; font-weight: 700; }
.rag-node   { color: #15803d; font-size: 0.7rem; background: #dcfce7; padding: 1px 6px; border-radius: 10px; }
.rag-source { color: #94a3b8; font-size: 0.68rem; }
.rag-text   { color: #334155; line-height: 1.6; margin-top: 6px; }

/* ── Info boxes ── */
.info-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.8rem;
    color: #334155;
}

/* ── Mono ── */
.mono { font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.74rem; color: #2563eb; }

/* ── Section headers in debug panel ── */
.dbg-section-header {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #94a3b8 !important;
    margin: 16px 0 5px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #e2e8f0;
}

/* ── Divider ── */
.dbg-divider { border: none; border-top: 1px solid #e2e8f0; margin: 12px 0; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# KB TOGGLE STATE  (lives in session_state so it persists across reruns)
# ═════════════════════════════════════════════════════════════════════════════

if "kb_mode" not in st.session_state:
    st.session_state.kb_mode = "original"   # "original" | "improved"

def _active_collection() -> str:
    return (
        "souli_chunks_improved"
        if st.session_state.kb_mode == "improved"
        else "souli_chunks"
    )

def _kb_label() -> str:
    return (
        "🚀 Improved Pipeline  (souli_chunks_improved)"
        if st.session_state.kb_mode == "improved"
        else "📦 Original Pipeline  (souli_chunks)"
    )


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading config...")
def _load_config():
    from souli_pipeline.config_loader import load_config
    return load_config(CONFIG_PATH)

import os
os.environ.setdefault("QDRANT_HOST", "localhost")

def get_engine():
    """Return engine for current KB mode. Creates a separate engine per mode."""
    key = f"engine_{st.session_state.kb_mode}"
    if key not in st.session_state:
        from souli_pipeline.conversation.engine import ConversationEngine
        cfg = _load_config()
        engine = ConversationEngine.from_config(cfg, gold_path=GOLD_PATH, excel_path=EXCEL_PATH)

        # Override which Qdrant collection this engine queries
        engine.qdrant_collection = _active_collection()

        # ── Attach debug event storage if engine doesn't have it ──────────────
        # engine.py doesn't define _debug_events / latest_debug yet,
        # so we add them here to avoid AttributeError.
        if not hasattr(engine, "_debug_events"):
            engine._debug_events = []
        if not hasattr(engine, "latest_debug"):
            engine.latest_debug = None

        st.session_state[key] = engine
    return st.session_state[key]


@st.cache_resource(show_spinner="Loading Whisper STT...")
def get_stt():
    from souli_pipeline.voice.stt import WhisperSTT
    return WhisperSTT(model_name="base")


@st.cache_resource(show_spinner="Loading Edge TTS...")
def get_tts():
    from souli_pipeline.voice.tts import EdgeTTS
    return EdgeTTS(voice="en-US-ChristopherNeural", rate="-5%", pitch="-10Hz")

def _reset_all():
    """Wipe engine + conversation — called when KB toggle switches."""
    for mode in ("original", "improved"):
        key = f"engine_{mode}"
        if key in st.session_state:
            try:
                st.session_state[key].reset()
            except Exception:
                pass
            del st.session_state[key]
    for k in ("messages", "voice_messages"):
        st.session_state.pop(k, None)


def init_session():
    engine = get_engine()
    if "messages" not in st.session_state:
        greeting = engine.greeting()
        st.session_state.messages = [{"role": "assistant", "content": greeting}]
    if "voice_messages" not in st.session_state:
        greeting = st.session_state.messages[0]["content"]
        st.session_state.voice_messages = [{"role": "assistant", "content": greeting}]


def _messages():
    return st.session_state["messages"]

def _voice_messages():
    return st.session_state["voice_messages"]


# ── Badge helpers ─────────────────────────────────────────────────────────────

_NODE_COLORS = {
    "blocked_energy":      ("#fee2e2", "#dc2626"),
    "depleted_energy":     ("#ffedd5", "#ea580c"),
    "scattered_energy":    ("#fef9c3", "#ca8a04"),
    "outofcontrol_energy": ("#ede9fe", "#7c3aed"),
    "normal_energy":       ("#dcfce7", "#16a34a"),
}
_PHASE_LABELS = {
    "greeting":     "Greeting",
    "intake":       "Intake",
    "sharing":      "Sharing",
    "deepening":    "Deepening",
    "summary":      "Summary",
    "intent_check": "Intent Check",
    "venting":      "Venting",
    "solution":     "Solution",
}

def phase_badge(phase: str) -> str:
    label = _PHASE_LABELS.get(phase, phase)
    return f'<span class="badge badge-phase">{label}</span>'

def node_badge(node: Optional[str]) -> str:
    if not node:
        return '<span class="badge badge-neutral">not detected</span>'
    bg, border = _NODE_COLORS.get(node, ("#f1f5f9", "#64748b"))
    label = node.replace("_", " ").title()
    return (f'<span class="badge" style="background:{bg};'
            f'color:{border};border-left:3px solid {border};">{label}</span>')

def conf_badge(conf: str) -> str:
    """Coloured confidence badge — light background so text is always readable."""
    _MAP = {
        "high_confidence":   ("#dcfce7", "#15803d", "✅ High Confidence"),
        "tagger_confirmed":  ("#d1fae5", "#065f46", "🤖 Tagger Confirmed"),
        "tagger_only":       ("#ecfdf5", "#047857", "🤖 Tagger Only"),
        "embedding_match":   ("#dbeafe", "#1d4ed8", "🔢 Embedding Match"),
        "keyword_fallback":  ("#fee2e2", "#b91c1c", "⚠️ Keyword Fallback"),
        "unknown":           ("#f1f5f9", "#475569", "❓ Unknown"),
    }
    bg, fg, label = _MAP.get(conf, ("#f1f5f9", "#475569", conf or "—"))
    return (
        f'<span class="badge" style="background:{bg};color:{fg};'
        f'border:1px solid {fg}55;font-weight:700;padding:3px 10px;border-radius:12px;">'
        f'{label}</span>'
    )


# ── Engine turn ───────────────────────────────────────────────────────────────

def run_turn(user_input: str):
    """Run one engine turn and return (response_text, debug_event).
    
    Extended version: also captures the exact prompt sent to Ollama,
    the diagnosis breakdown (keyword/embedding/tagger), and solution
    framework content when in solution phase.
    """
    engine = get_engine()
 
    # ── Capture RAG chunks ────────────────────────────────────────────────
    rag_captured: list = []
    _orig_rag = engine._rag_retrieve
 
    def _capturing_rag(query, energy_node):
        chunks = _orig_rag(query, energy_node)
        rag_captured.extend(chunks)
        return chunks
 
    engine._rag_retrieve = _capturing_rag
 
    # ── Capture the exact prompt sent to Ollama ───────────────────────────
    # We monkey-patch counselor's _build_chat_messages to intercept the call.
    prompt_captured = {"system": None, "messages": None, "type": "counselor"}
 
    try:
        import souli_pipeline.conversation.counselor as _counselor_mod
 
        _orig_generate = _counselor_mod.generate_counselor_response
        _orig_solution = _counselor_mod.generate_solution_response
 
        def _capturing_counselor(history, user_message, rag_chunks, **kwargs):
            # Reconstruct exactly what gets built inside generate_counselor_response
            from souli_pipeline.conversation.counselor import (
                _build_chat_messages, _build_counselor_system,
            )
            msgs = _build_chat_messages(
                history, user_message, rag_chunks,
                energy_node=kwargs.get("energy_node"),
            )
            sys_p = _build_counselor_system(
                user_name=kwargs.get("user_name"),
                phase=kwargs.get("phase"),
                asked_topics=kwargs.get("asked_topics"),
            )
            prompt_captured["system"] = sys_p
            prompt_captured["messages"] = msgs
            prompt_captured["type"] = "counselor"
            return _orig_generate(history, user_message, rag_chunks, **kwargs)
 
        def _capturing_solution(energy_node, framework_solution, user_context, **kwargs):
            from souli_pipeline.conversation.counselor import (
                _build_solution_prompt, _SOLUTION_SYSTEM,
            )
            p = _build_solution_prompt(energy_node, framework_solution, user_context)
            prompt_captured["system"] = _SOLUTION_SYSTEM
            prompt_captured["messages"] = [{"role": "user", "content": p}]
            prompt_captured["type"] = "solution"
            prompt_captured["framework_solution"] = framework_solution
            prompt_captured["energy_node"] = energy_node
            prompt_captured["user_context"] = user_context[:400]
            return _orig_solution(energy_node, framework_solution, user_context, **kwargs)
 
        _counselor_mod.generate_counselor_response = _capturing_counselor
        _counselor_mod.generate_solution_response  = _capturing_solution
        patched_counselor = True
    except Exception:
        patched_counselor = False
 
    # ── Run the turn ──────────────────────────────────────────────────────
    phase_before = engine.state.phase
    t_start = time.perf_counter()
 
    full_response = ""
    source = "llm"
    try:
        for chunk in engine.turn_stream(user_input):
            full_response += chunk
    except Exception:
        try:
            full_response = engine.turn(user_input)
        except Exception as e:
            full_response = f"[Engine error: {e}]"
            source = "fallback"
 
    elapsed_ms = int((time.perf_counter() - t_start) * 1000)
 
    # ── Restore patches ───────────────────────────────────────────────────
    engine._rag_retrieve = _orig_rag
    if patched_counselor:
        _counselor_mod.generate_counselor_response = _orig_generate
        _counselor_mod.generate_solution_response  = _orig_solution
 
    phase_after = engine.state.phase
    diag = engine.diagnosis_summary
 
    # ── Pull the triple-diagnosis breakdown if available ──────────────────
    diag_detail = getattr(engine.state, "_last_diagnosis_detail", None) or {}
 
    # ── Build debug event ─────────────────────────────────────────────────
    debug_ev = {
        "turn":         engine.state.turn_count,
        "user_text":    user_input,
        "phase_before": phase_before,
        "phase_after":  phase_after,
        "kb_mode":      st.session_state.kb_mode,
        "collection":   _active_collection(),
 
        # ── Diagnosis (now includes full triple-hybrid breakdown) ──────────
        "diagnosis": {
            "ran":          True,
            "energy_node":  diag.get("energy_node"),
            "confidence":   diag.get("confidence", "unknown"),
            # New fields from triple hybrid:
            "detail":       diag_detail,
            # Is this a fallback? True if confidence is keyword_fallback
            "is_fallback":  diag.get("confidence", "") == "keyword_fallback",
        },
 
        # ── RAG ────────────────────────────────────────────────────────────
        "rag": {
            "ran":                  True,
            "query":                user_input,
            "energy_node_filter":   diag.get("energy_node"),
            "results_count":        len(rag_captured),
            "results":              rag_captured[:5],   # full chunks with text
        },
 
        # ── LLM / Prompt ───────────────────────────────────────────────────
        "llm": {
            "ran":                  True,
            "model":                engine.chat_model,
            "used_fallback":        source == "fallback",
            "phase":                phase_before,
            "history_length":       len(engine.state.messages),
            "rag_chunks_injected":  len(rag_captured),
            "latency_ms":           elapsed_ms,
            # New: exact prompt captured
            "prompt_type":          prompt_captured.get("type", "counselor"),
            "prompt_system":        prompt_captured.get("system"),
            "prompt_messages":      prompt_captured.get("messages"),
        },
 
        # ── Solution phase extra data ──────────────────────────────────────
        "solution": {
            "active":              phase_after == "solution" or phase_before == "solution",
            "framework_solution":  prompt_captured.get("framework_solution"),
            "energy_node":         prompt_captured.get("energy_node"),
            "user_context":        prompt_captured.get("user_context"),
        } if prompt_captured.get("type") == "solution" else {"active": False},
 
        # ── Full state snapshot ────────────────────────────────────────────
        "state_after": {
            "phase":                engine.state.phase,
            "energy_node":          engine.state.energy_node,
            "turn_count":           engine.state.turn_count,
            "intent":               engine.state.intent,
            "user_name":            engine.state.user_name,
            "summary_attempted":    engine.state.summary_attempted,
        },
    }

    if not hasattr(engine, "_debug_events"):
        engine._debug_events = []
    engine._debug_events.append(debug_ev)
    engine.latest_debug = debug_ev
 
    return full_response, debug_ev


# ═════════════════════════════════════════════════════════════════════════════
# DEBUG PANEL RENDERERS
# ═════════════════════════════════════════════════════════════════════════════

def render_phase_flow():
    engine = get_engine()
    events = getattr(engine, "_debug_events", [])
    if not events:
        st.markdown('<span style="color:#94a3b8;font-size:0.78rem;">No turns yet.</span>',
                    unsafe_allow_html=True)
        return
    parts = []
    for ev in events:
        pb = ev.get("phase_before", "?")
        pa = ev.get("phase_after", "?")
        if pb == pa:
            parts.append(f'<span class="badge badge-phase" style="font-size:0.65rem;">{_PHASE_LABELS.get(pb, pb)}</span>')
        else:
            parts.append(
                f'<span class="badge badge-phase" style="font-size:0.65rem;">{_PHASE_LABELS.get(pb, pb)}</span>'
                f'<span style="color:#16a34a;margin:0 3px;">→</span>'
                f'<span class="badge badge-warn" style="font-size:0.65rem;">{_PHASE_LABELS.get(pa, pa)}</span>'
            )
    html = '<div style="display:flex;flex-wrap:wrap;gap:4px;align-items:center;">' + "".join(parts) + "</div>"
    st.markdown(html, unsafe_allow_html=True)

def render_turn_debug(ev: Dict[str, Any]):
    """
    Render the full debug panel for one conversation turn.
    Sections:
      1. Phase Transition
      2. KB Used
      3. User Input
      4. 🧠 Diagnosis  ← ENHANCED: shows keyword / embedding / tagger breakdown
      5. 🗄️ Qdrant RAG ← ENHANCED: shows actual chunk text + scores
      6. 🤖 LLM Call   ← ENHANCED: shows full prompt inspector
      7. 💊 Solution   ← NEW: shows framework content when in solution phase
      8. ⚠️ Fallback   ← NEW: big warning banner when keyword_fallback active
    """
    if not ev:
        return
    # ── 1. Phase Transition ───────────────────────────────────────────────
    st.markdown('<div class="dbg-section-header">Phase Transition</div>', unsafe_allow_html=True)
    pb, pa = ev.get("phase_before", "?"), ev.get("phase_after", "?")
    if pb == pa:
        st.markdown(phase_badge(pb) + ' <span style="color:#94a3b8;">no change</span>', unsafe_allow_html=True)
    else:
        st.markdown(
            phase_badge(pb) + ' <span style="color:#16a34a;font-size:1rem;">→</span> ' + phase_badge(pa),
            unsafe_allow_html=True,
        )
 
    # ── 2. KB Mode ────────────────────────────────────────────────────────
    kb = ev.get("kb_mode", "?")
    coll = ev.get("collection", "?")
    color = "#2563eb" if kb == "improved" else "#ca8a04"
    st.markdown(
        f'<div class="dbg-section-header">Knowledge Base Used</div>'
        f'<span class="badge" style="background:#eff6ff;color:{color};border-left:3px solid {color};">'
        f'{"🚀 Improved" if kb == "improved" else "📦 Original"}  ·  {coll}'
        f'</span>',
        unsafe_allow_html=True,
    )
 
    # ── 3. User Input ─────────────────────────────────────────────────────
    st.markdown('<div class="dbg-section-header">User Input</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-box mono">{ev.get("user_text","")[:400]}</div>', unsafe_allow_html=True)
 
    # ── 4. 🧠 ENHANCED Diagnosis ──────────────────────────────────────────
    diag = ev.get("diagnosis", {})
    detail = diag.get("detail", {})
    is_fallback = diag.get("is_fallback", False)
 
    # ── Fallback warning banner ────────────────────────────────────────────
    if is_fallback:
        st.markdown(
            """<div style="background:#fef2f2;border:2px solid #ef4444;border-radius:8px;
            padding:10px 14px;margin:6px 0;">
            <span style="color:#b91c1c;font-weight:700;font-size:0.88rem;">
            ⚠️ KEYWORD FALLBACK ACTIVE
            </span><br>
            <span style="color:#7f1d1d;font-size:0.8rem;">
            Neither gold embedding nor Qwen tagger produced a confident result.
            The energy node was guessed from keyword matching only.
            Response quality may be poor.
            </span></div>""",
            unsafe_allow_html=True,
        )
 
    with st.expander("🧠 Diagnosis — full breakdown", expanded=True):
 
        # ── Final result row ───────────────────────────────────────────────
        col_node, col_secondary, col_conf = st.columns(3)
 
        with col_node:
            st.markdown("**Primary Node**")
            st.markdown(node_badge(diag.get("energy_node")), unsafe_allow_html=True)
 
        with col_secondary:
            st.markdown("**Also Possible**")
            sec = diag.get("detail", {}).get("final", {}).get("secondary_node") or \
                  ev.get("state_after", {}).get("secondary_node")
            if sec:
                _NODE_COLORS = {
                    "blocked_energy": "#e74c3c", "depleted_energy": "#e67e22",
                    "scattered_energy": "#d4a017", "outofcontrol_energy": "#9b59b6",
                    "normal_energy": "#27ae60",
                }
                c = _NODE_COLORS.get(sec, "#64748b")
                label = sec.replace("_energy","").replace("_"," ").title()
                st.markdown(
                    f'<span style="background:#f8fafc;color:{c};border:1px solid {c}88;'
                    f'padding:3px 10px;border-radius:12px;font-size:0.75rem;font-weight:600;">'
                    f'~ {label}</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<span style="color:#94a3b8;font-size:0.78rem;">—</span>', unsafe_allow_html=True)
 
        with col_conf:
            st.markdown("**Confidence**")
            st.markdown(conf_badge(diag.get("confidence", "unknown")), unsafe_allow_html=True)
 
        # ── Node reasoning (only shown at summary phase) ───────────────────
        reasoning = ev.get("state_after", {}).get("node_reasoning") or \
                    diag.get("detail", {}).get("node_reasoning")
        if reasoning:
            st.markdown(
                f'<div style="background:#f0fdf4;border-left:3px solid #16a34a;'
                f'border-radius:6px;padding:8px 12px;margin:6px 0;">'
                f'<span style="color:#166534;font-size:0.75rem;font-weight:600;">WHY THIS NODE</span><br>'
                f'<span style="color:#14532d;font-size:0.85rem;">{reasoning}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
 
        # ── Rolling context info ───────────────────────────────────────────
        n_msgs = detail.get("rolling_context_messages", 0)
        if n_msgs:
            st.caption(f"Diagnosis based on rolling context from {n_msgs} problem message(s)")
 
        st.markdown("---")
        st.markdown(
            '<span style="color:#374151;font-size:0.8rem;font-weight:600;">'
            'Method breakdown — how each signal voted:</span>',
            unsafe_allow_html=True,
        )
 
        # ── Three method cards — LIGHT background so text is readable ─────
        c1, c2, c3 = st.columns(3)
 
        # Keyword card
        kw = detail.get("keyword", {})
        with c1:
            st.markdown(
                f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
                f'border-radius:8px;padding:10px 12px;">'
                f'<div style="color:#64748b;font-size:0.72rem;font-weight:600;'
                f'text-transform:uppercase;letter-spacing:0.5px;">1️⃣ Keyword</div>'
                f'<div style="color:#94a3b8;font-size:0.72rem;margin-top:2px;">Always runs</div>'
                f'<div style="color:#1e293b;font-weight:700;font-size:0.85rem;margin-top:6px;">'
                f'{kw.get("node", "—").replace("_energy","").replace("_"," ").title()}'
                f'</div></div>',
                unsafe_allow_html=True,
            )
 
        # Embedding card
        emb = detail.get("embedding", {})
        emb_available = emb.get("available", False)
        emb_hit = emb.get("confidence") == "embedding_match"
        emb_sim = emb.get("similarity")
        emb_node = emb.get("node", "—")
        if not emb_available:
            emb_status_color, emb_status = "#94a3b8", "No gold.xlsx loaded"
        elif emb_hit:
            emb_status_color, emb_status = "#16a34a", f"Hit · sim={emb_sim:.3f}" if emb_sim else "Hit"
        else:
            emb_status_color, emb_status = "#d97706", f"Below threshold · {emb_sim:.3f}" if emb_sim else "Below threshold"
 
        with c2:
            st.markdown(
                f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
                f'border-radius:8px;padding:10px 12px;">'
                f'<div style="color:#64748b;font-size:0.72rem;font-weight:600;'
                f'text-transform:uppercase;letter-spacing:0.5px;">2️⃣ Gold Embedding</div>'
                f'<div style="color:{emb_status_color};font-size:0.72rem;margin-top:2px;">{emb_status}</div>'
                f'<div style="color:#1e293b;font-weight:700;font-size:0.85rem;margin-top:6px;">'
                f'{"—" if not emb_hit else emb_node.replace("_energy","").replace("_"," ").title()}'
                f'</div></div>',
                unsafe_allow_html=True,
            )
 
        # Tagger card
        tgr = detail.get("tagger", {})
        tgr_available = tgr.get("available", False)
        tgr_node = tgr.get("node", "—")
        tgr_fallback = tgr.get("used_fallback", False)
        tgr_reason = tgr.get("reason", "")
        if not tgr_available:
            tgr_status_color, tgr_status = "#94a3b8", "Ollama offline"
        elif tgr_fallback:
            tgr_status_color, tgr_status = "#d97706", "Tagger fell back to keyword"
        else:
            tgr_status_color, tgr_status = "#16a34a", "Qwen used ✓"
 
        tgr_reason_html = f'<div style="color:#64748b;font-size:0.7rem;margin-top:2px;">{tgr_reason[:55]}…</div>' if tgr_reason and not tgr_fallback else ""

        with c3:
            st.markdown(
                f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
                f'border-radius:8px;padding:10px 12px;">'
                f'<div style="color:#64748b;font-size:0.72rem;font-weight:600;'
                f'text-transform:uppercase;letter-spacing:0.5px;">3️⃣ Qwen Tagger</div>'
                f'<div style="color:{tgr_status_color};font-size:0.72rem;margin-top:2px;">{tgr_status}</div>'
                f'<div style="color:#1e293b;font-weight:700;font-size:0.85rem;margin-top:6px;">'
                f'{tgr_node.replace("_energy","").replace("_"," ").title() if tgr_available else "—"}'
                f'</div>'
                f'{tgr_reason_html}'
                f'</div>',
                unsafe_allow_html=True,
            )
 
        # ── Score bar chart ────────────────────────────────────────────────
        scores = detail.get("scores", {})
        if scores:
            st.markdown(
                '<div style="margin-top:12px;">'
                '<span style="color:#374151;font-size:0.75rem;font-weight:600;">Node scores (higher = stronger signal)</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            max_score = max(scores.values()) if scores else 1
            _NODE_COLORS_SCORE = {
                "blocked_energy": "#ef4444", "depleted_energy": "#f97316",
                "scattered_energy": "#eab308", "outofcontrol_energy": "#a855f7",
                "normal_energy": "#22c55e",
            }
            for node_name, score in sorted(scores.items(), key=lambda x: -x[1]):
                pct = int((score / max_score) * 100)
                color = _NODE_COLORS_SCORE.get(node_name, "#94a3b8")
                label = node_name.replace("_energy","").replace("_"," ").title()
                is_primary = (node_name == diag.get("energy_node"))
                border = f"border:2px solid {color};" if is_primary else ""
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0;">'
                    f'<div style="width:110px;color:#374151;font-size:0.72rem;'
                    f'font-weight:{"700" if is_primary else "400"};">{label}</div>'
                    f'<div style="flex:1;background:#f1f5f9;border-radius:4px;height:14px;{border}">'
                    f'<div style="width:{pct}%;background:{color};height:100%;border-radius:4px;"></div>'
                    f'</div>'
                    f'<div style="color:{color};font-size:0.7rem;font-weight:700;width:28px;">{score:.1f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    # ── 5. 🗄️ ENHANCED RAG Chunks ─────────────────────────────────────────
    rag = ev.get("rag", {})
    rag_count = rag.get("results_count", 0)
    results = rag.get("results", [])
 
    with st.expander(
        f"🗄️ Qdrant — {rag_count} chunks  [filter: {rag.get('energy_node_filter','none')}]",
        expanded=True,
    ):
        if not results:
            st.markdown(
                '<div class="info-box" style="color:#dc2626;border-left:3px solid #fca5a5;">'
                '⚠️ No chunks retrieved. The LLM prompt has ZERO teaching context.<br>'
                'This is a major cause of poor/generic responses.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            for i, r in enumerate(results, 1):
                score = r.get("score", 0)
                chunk_node = r.get("energy_node", "")
                diag_node = rag.get("energy_node_filter", "")
                # Flag if chunk node doesn't match diagnosed node — context bleed
                node_mismatch = chunk_node and diag_node and chunk_node != diag_node
                score_color = "#16a34a" if score > 0.7 else "#d97706" if score > 0.45 else "#dc2626"
                mismatch_html = (
                    f'<span style="color:#ef4444;font-size:0.7rem;"> ⚠️ node mismatch!</span>'
                    if node_mismatch else ""
                )
                st.markdown(
                    f'<div class="rag-card">'
                    f'<span class="rag-node">[{chunk_node}]</span>{mismatch_html}  '
                    f'<span class="rag-score" style="color:{score_color};">score: {score:.4f}</span>'
                    f'<span class="rag-source">  {r.get("source_video","")}</span>'
                    f'<div class="rag-text">{r.get("text","")[:350]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
 
    # ── 6. 🤖 ENHANCED LLM / Full Prompt Inspector ────────────────────────
    llm = ev.get("llm", {})
    fallback_flag = llm.get("used_fallback", False)
 
    with st.expander(
        "🤖 LLM Call" + (" — ⚠️ FALLBACK" if fallback_flag else ""),
        expanded=True,
    ):
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<span class="badge badge-llm">{llm.get("model","?")}</span>', unsafe_allow_html=True)
        c2.metric("History msgs", llm.get("history_length", 0))
        c3.metric("Latency", f'{llm.get("latency_ms", 0)} ms')
 
        if fallback_flag:
            st.error("⚠️ Ollama was unavailable — response came from hardcoded fallback strings, NOT from the LLM.")
 
        st.markdown(f"RAG chunks injected: **{llm.get('rag_chunks_injected', 0)}**")
 
        # ── Full Prompt Inspector ──────────────────────────────────────────
        prompt_type = llm.get("prompt_type", "counselor")
        sys_p = llm.get("prompt_system")
        msgs_p = llm.get("prompt_messages")
 
        st.markdown("---")
        st.markdown(
            f"**📋 Prompt Inspector** "
            f'<span style="color:#64748b;font-size:0.75rem;">(type: {prompt_type})</span>',
            unsafe_allow_html=True,
        )
 
        if sys_p:
            with st.expander("System Prompt", expanded=False):
                st.code(sys_p, language="text")
        else:
            st.caption("System prompt not captured (turn may have used fallback path)")
 
        if msgs_p:
            with st.expander(f"Messages array ({len(msgs_p)} msgs)", expanded=False):
                for i, m in enumerate(msgs_p):
                    role = m.get("role", "?")
                    content = m.get("content", "")
                    role_color = "#2563eb" if role == "user" else "#16a34a" if role == "assistant" else "#9333ea"
                    # Highlight RAG injection message
                    is_rag_injection = "[CONTEXT" in content
                    border = "2px solid #d97706" if is_rag_injection else "1px solid #334155"
                    label = f"[{i}] {role}" + (" ← RAG injection" if is_rag_injection else "")
                    st.markdown(
                        f'<div style="border:{border};border-radius:6px;padding:8px;margin:4px 0;">'
                        f'<div style="color:{role_color};font-size:0.72rem;font-weight:700;">{label}</div>'
                        f'<div style="color:#cbd5e1;font-size:0.8rem;margin-top:4px;white-space:pre-wrap;">'
                        f'{content[:600]}'
                        f'{"..." if len(content)>600 else ""}'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("Messages not captured.")
 
    # ── 7. 💊 Solution Phase Inspector (only shown when in solution phase) ─
    sol = ev.get("solution", {})
    if sol.get("active"):
        fw = sol.get("framework_solution") or {}
        sol_node = sol.get("energy_node", "")
        user_ctx = sol.get("user_context", "")
 
        with st.expander("💊 Solution Phase — framework content used", expanded=True):
            if not fw:
                st.warning(
                    "No framework solution found for this node. "
                    "The engine fell back to generic LLM response without structured practices."
                )
            else:
                st.markdown(f"**Node:** `{sol_node}`")
                st.markdown(f"**User context passed to LLM:** _{user_ctx}_")
                st.markdown("---")
 
                healing = fw.get("primary_healing_principles", "")
                practices = fw.get("primary_practices ( 7 min quick relief)", "")
                deeper = fw.get("deeper_meditations_program ( 7 day quick recovery)", "")
                caution = fw.get("Caution", "")
 
                if healing:
                    st.markdown("**Healing Principles** (injected into prompt):")
                    st.info(healing[:500])
                if practices:
                    st.markdown("**Quick Relief Practices — 7 min** (injected):")
                    st.success(practices[:400])
                if deeper:
                    st.markdown("**7-Day Recovery Program** (injected):")
                    st.info(deeper[:400])
                if caution:
                    st.markdown("**Caution** (injected):")
                    st.warning(caution[:200])
 
                if not healing and not practices:
                    st.error(
                        "Framework entry exists but has empty healing + practices fields. "
                        "Check your Energy Framework Excel — this node may have missing data."
                    )
 


def render_qdrant_inspector(cfg):
    st.markdown("### 🔍 Qdrant Inspector")
    st.caption(f"Currently querying: **{_active_collection()}** (follows KB toggle above)")

    col1, col2 = st.columns([3, 1])
    with col1:
        query_text = st.text_area("Query text", height=80, placeholder="Type any text to search...")
    with col2:
        node_options = ["(no filter)"] + (cfg.energy.nodes_allowed if cfg else [
            "blocked_energy", "depleted_energy", "scattered_energy",
            "outofcontrol_energy", "normal_energy",
        ])
        node_filter = st.selectbox("Energy node filter", node_options)
        top_k = st.slider("Top K", 1, 15, 5)

    r = cfg.retrieval if cfg else None
    col_a, col_b = st.columns(2)
    with col_a:
        qdrant_host = st.text_input("Qdrant host", value=r.qdrant_host if r else "localhost")
    with col_b:
        qdrant_port = st.number_input("Port", value=r.qdrant_port if r else 6333, step=1)

    st.markdown(
        f'<div style="display:inline-block;background:#eff6ff;border:1px solid #bfdbfe;'
        f'border-radius:8px;padding:5px 14px;font-size:0.75rem;color:#7eb8f7;margin-bottom:10px;">'
        f'🗄️ Querying: <b>{_active_collection()}</b></div>',
        unsafe_allow_html=True,
    )

    if st.button("🔍 Run Query", type="primary", use_container_width=True):
        if not query_text.strip():
            st.warning("Enter some query text first.")
            return
        with st.spinner("Querying Qdrant..."):
            try:
                emb_model = r.embedding_model if r else "sentence-transformers/all-MiniLM-L6-v2"
                node = None if node_filter == "(no filter)" else node_filter
                t0 = time.time()
                if st.session_state.kb_mode == "improved":
                    from souli_pipeline.retrieval.qdrant_store_improved import query_improved_chunks
                    results = query_improved_chunks(
                        user_text=query_text, collection=_active_collection(),
                        energy_node=node, top_k=top_k, embedding_model=emb_model,
                        host=qdrant_host, port=int(qdrant_port),
                    )
                else:
                    from souli_pipeline.retrieval.qdrant_store import query_chunks
                    results = query_chunks(
                        user_text=query_text, collection=_active_collection(),
                        energy_node=node, top_k=top_k, embedding_model=emb_model,
                        host=qdrant_host, port=int(qdrant_port), score_threshold=0.0,
                    )
                latency = (time.time() - t0) * 1000
                st.success(f"Retrieved {len(results)} chunks in {latency:.0f} ms")
                for i, r_item in enumerate(results, 1):
                    score = r_item.get("score", 0)
                    score_color = "#16a34a" if score > 0.7 else "#d97706" if score > 0.45 else "#dc2626"
                    with st.expander(
                        f"#{i} — score: {score:.4f}  [{r_item.get('energy_node','')}]  {r_item.get('source_video','')[:50]}",
                        expanded=(i <= 3),
                    ):
                        st.markdown(
                            f'<span class="rag-score" style="color:{score_color};">Score: {score:.4f}</span>  '
                            f'<span class="rag-node">[{r_item.get("energy_node","")}]</span>  '
                            f'<span class="rag-source">{r_item.get("source_video","")}</span>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(r_item.get("text", ""))
                        st.caption(f"URL: {r_item.get('youtube_url','—')}")
            except Exception as exc:
                st.error(f"Qdrant error: {exc}")


def render_session_state_tab():
    engine = get_engine()
    s = engine.state
    st.markdown("### 🗃️ Full ConversationState")
    st.json({
        "phase": s.phase, "turn_count": s.turn_count, "user_name": s.user_name,
        "energy_node": s.energy_node, "node_confidence": s.node_confidence,
        "intent": s.intent, "summary_attempted": s.summary_attempted,
        "summary_confirmed": s.summary_confirmed, "rich_opening": s.rich_opening,
        "short_answer_count": s.short_answer_count,
        "user_text_buffer_words": len(s.user_text_buffer.split()),
        "messages_count": len(s.messages),
        "active_collection": _active_collection(),
    })
    st.markdown("### 💬 Message History")
    for i, msg in enumerate(s.messages):
        role = msg["role"]
        color = "#2563eb" if role == "user" else "#16a34a"
        icon = "👤" if role == "user" else "🌿"
        with st.expander(f"{icon} [{i}] {role}", expanded=False):
            st.markdown(f'<div class="info-box" style="color:{color};">{msg["content"]}</div>', unsafe_allow_html=True)


def render_turn_history_tab():
    engine = get_engine()
    events = getattr(engine, "_debug_events", [])
    if not events:
        st.markdown('<span style="color:#94a3b8;">No turns yet.</span>', unsafe_allow_html=True)
        return
    st.markdown(f"### {len(events)} turns recorded")
    for ev in reversed(events):
        turn_n = ev.get("turn", "?")
        pb = _PHASE_LABELS.get(ev.get("phase_before", ""), ev.get("phase_before", "?"))
        pa = _PHASE_LABELS.get(ev.get("phase_after", ""), ev.get("phase_after", "?"))
        node = ev.get("state_after", {}).get("energy_node") or ev.get("diagnosis", {}).get("energy_node")
        rag_n = ev.get("rag", {}).get("results_count", 0)
        fallback = ev.get("llm", {}).get("used_fallback", False)
        kb = ev.get("kb_mode", "?")
        user_snippet = ev.get("user_text", "")[:60]
        label = f"Turn #{turn_n} | {pb}" + (f" → {pa}" if pb != pa else "") + f" | {node or '?'} | RAG: {rag_n} | KB: {kb}" + (" | ⚠ FALLBACK" if fallback else "")
        with st.expander(label, expanded=False):
            st.caption(f'User: "{user_snippet}"')
            render_turn_debug(ev)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

init_session()

# ── TOP BAR ───────────────────────────────────────────────────────────────────
top_left, top_right = st.columns([1, 2])
with top_left:
    st.markdown("## 🔬 Souli Dev")
    st.caption("Developer debug interface")
with top_right:
    engine_ref = get_engine()
    diag = engine_ref.diagnosis_summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Turn", diag.get("turn_count", 0))
    c2.metric("Phase", _PHASE_LABELS.get(diag.get("phase", ""), "—"))
    c3.metric("Node", (diag.get("energy_node") or "—").replace("_energy", "").replace("_", " ").title())
    c4.metric("Confidence", diag.get("confidence", "—"))

# ═════════════════════════════════════════════════════════════════════════════
# KB TOGGLE BAR — always visible, above everything
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 🗄️ Knowledge Base")
st.caption("Switch which Qdrant collection the conversation engine queries for RAG. Each mode keeps its own independent conversation history.")

kb_col1, kb_col2, kb_col3 = st.columns([2, 2, 3])

with kb_col1:
    orig_active = st.session_state.kb_mode == "original"
    orig_style = "primary" if orig_active else "secondary"
    if st.button(
        f"{'✅ ' if orig_active else ''}📦 Original Pipeline\nsouli_chunks",
        key="btn_original",
        type=orig_style,
        use_container_width=True,
        disabled=orig_active,
    ):
        st.session_state.kb_mode = "original"
        _reset_all()
        st.rerun()

with kb_col2:
    impr_active = st.session_state.kb_mode == "improved"
    impr_style = "primary" if impr_active else "secondary"
    if st.button(
        f"{'✅ ' if impr_active else ''}🚀 Improved Pipeline\nsouli_chunks_improved",
        key="btn_improved",
        type=impr_style,
        use_container_width=True,
        disabled=impr_active,
    ):
        st.session_state.kb_mode = "improved"
        _reset_all()
        st.rerun()

with kb_col3:
    mode_color = "#2563eb" if st.session_state.kb_mode == "improved" else "#ca8a04"
    mode_icon  = "🚀" if st.session_state.kb_mode == "improved" else "📦"
    st.markdown(
        f'<div style="background:#f0f7ff;border:1px solid {mode_color};border-radius:10px;'
        f'padding:12px 16px;margin-top:4px;">'
        f'<span style="color:{mode_color};font-weight:700;font-size:0.9rem;">'
        f'{mode_icon} Active: {_active_collection()}</span><br>'
        f'<span style="color:#64748b;font-size:0.75rem;">Switching resets conversation — fresh start with the new KB</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── TWO-COLUMN LAYOUT ─────────────────────────────────────────────────────────
left_col, right_col = st.columns([4, 5], gap="medium")

# ── LEFT: DEBUG PANEL ─────────────────────────────────────────────────────────
with left_col:
    cfg_obj = _load_config()

    st.markdown('<div class="dbg-section-header">Phase Flow (all turns)</div>', unsafe_allow_html=True)
    render_phase_flow()
    st.markdown('<hr class="dbg-divider"/>', unsafe_allow_html=True)

    tab_current, tab_history, tab_qdrant, tab_session = st.tabs([
        "📍 Current Turn",
        "🕑 Turn History",
        "🗄️ Qdrant Inspector",
        "🗃️ Session State",
    ])

    with tab_current:
        latest = getattr(get_engine(), "latest_debug", None)
        if not latest:
            st.markdown('<div style="color:#94a3b8;padding:20px 0;">No turns yet. Send a message to start.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="dbg-section-header">Turn #{latest.get("turn","?")} details</div>', unsafe_allow_html=True)
            render_turn_debug(latest)

    with tab_history:
        render_turn_history_tab()

    with tab_qdrant:
        render_qdrant_inspector(cfg_obj)

    with tab_session:
        render_session_state_tab()

# ── RIGHT: CHAT ───────────────────────────────────────────────────────────────
with right_col:
    st.markdown(f"## 🌿 Souli  <span style='font-size:0.75rem;color:#64748b;'>· {_kb_label()}</span>", unsafe_allow_html=True)
    st.caption("Your inner wellness companion  ·  [dev mode]")

    ctrl_l, ctrl_r = st.columns([3, 1])
    with ctrl_r:
        if st.button("↺ Reset", use_container_width=True, help="Reset current conversation (same KB)"):
            key = f"engine_{st.session_state.kb_mode}"
            if key in st.session_state:
                try:
                    st.session_state[key].reset()
                except Exception:
                    pass
                del st.session_state[key]
            for k in ("messages", "voice_messages"):
                st.session_state.pop(k, None)
            st.rerun()
    with ctrl_l:
        st.caption(f"Config: `{CONFIG_PATH}`  |  Gold: `{GOLD_PATH or 'none'}`")

    chat_tab, voice_tab = st.tabs(["💬 Text Chat", "🎤 Voice Chat"])


    # Secondary node + reasoning tag (shown in chat UI top area)
    diag_now = get_engine().diagnosis_summary
    sec_node = diag_now.get("secondary_node")
    reasoning = diag_now.get("node_reasoning")

    if sec_node or reasoning:
        _NC = {
            "blocked_energy": "#ef4444",
            "depleted_energy": "#f97316",
            "scattered_energy": "#eab308",
            "outofcontrol_energy": "#a855f7",
            "normal_energy": "#22c55e",
        }

        parts = []

        if sec_node:
            c = _NC.get(sec_node, "#64748b")
            lbl = sec_node.replace("_energy", "").replace("_", " ").title()

            parts.append(
                f"<span style='background:#f8fafc;color:{c};border:1px solid {c}88;"
                f"padding:2px 8px;border-radius:10px;font-size:0.72rem;font-weight:600;'>"
                f"Also possible: {lbl}</span>"
            )

        if reasoning:
            parts.append(
                f"<span style='color:#64748b;font-size:0.62rem;font-style:italic;'>"
                f"{reasoning}</span>"
            )

        st.markdown(
            "<div style='display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:6px;'>"
            + " ".join(parts)
            + "</div>",
            unsafe_allow_html=True,
        )   

    # ── Text Chat ─────────────────────────────────────────────────────────────
    with chat_tab:
        msg_box = st.container(height=460, border=False)
        with msg_box:
            for msg in _messages():
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        if user_input := st.chat_input("Share what's on your mind...", key="text_input"):
            _messages().append({"role": "user", "content": user_input})
            with st.spinner("Souli is with you…"):
                full_response, _ = run_turn(user_input)
            _messages().append({"role": "assistant", "content": full_response})
            st.rerun()

    # ── Voice Chat ────────────────────────────────────────────────────────────
    with voice_tab:
        voice_box = st.container(height=380, border=False)
        with voice_box:
            for msg in _voice_messages():
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    if msg["role"] == "assistant" and "audio" in msg:
                        st.audio(msg["audio"], format="audio/mp3")

        audio_input = st.audio_input("🎙️ Press to record", key="voice_input")
        if audio_input is not None:
            with st.spinner("Transcribing..."):
                stt = get_stt()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_input.read())
                    tmp_path = tmp.name
                try:
                    transcript = stt.transcribe_file(tmp_path)
                finally:
                    os.unlink(tmp_path)
            if transcript.strip():
                _voice_messages().append({"role": "user", "content": transcript})
                with st.spinner("Souli is thinking..."):
                    response, _ = run_turn(transcript)
                with st.spinner("Generating voice..."):
                    tts = get_tts()
                    audio_bytes = tts.synthesize(response)
                _voice_messages().append({"role": "assistant", "content": response, "audio": audio_bytes})
                st.rerun()
            else:
                st.warning("Could not transcribe. Try again.")

        if voice_text := st.chat_input("Or type here...", key="voice_text_input"):
            _voice_messages().append({"role": "user", "content": voice_text})
            with st.spinner("Souli is thinking..."):
                response, _ = run_turn(voice_text)
            with st.spinner("Generating voice..."):
                tts = get_tts()
                audio_bytes = tts.synthesize(response)
            _voice_messages().append({"role": "assistant", "content": response, "audio": audio_bytes})
            st.rerun()