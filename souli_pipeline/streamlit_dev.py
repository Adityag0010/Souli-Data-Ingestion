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
GOLD_PATH   = os.environ.get("SOULI_GOLD_PATH", None)
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
    return EdgeTTS(voice="en-IN-NeerjaNeural")


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

def conf_badge(confidence: str) -> str:
    if confidence == "embedding_match":
        return '<span class="badge badge-embed">embedding match</span>'
    if "keyword" in confidence:
        return '<span class="badge badge-kw">keyword fallback</span>'
    return f'<span class="badge badge-neutral">{confidence}</span>'


# ── Engine turn ───────────────────────────────────────────────────────────────

def run_turn(user_input: str) -> tuple[str, dict]:
    """Run one engine turn. Returns (response_text, debug_dict)."""
    engine = get_engine()

    # Capture RAG chunks via monkey-patch
    rag_captured: list = []
    _orig_rag = engine._rag_retrieve

    def _capturing_rag(query, energy_node):
        t0 = time.perf_counter()
        chunks = _orig_rag(query, energy_node)
        rag_captured.extend(chunks)
        return chunks

    engine._rag_retrieve = _capturing_rag

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
    engine._rag_retrieve = _orig_rag

    phase_after = engine.state.phase
    diag = engine.diagnosis_summary

    # Build a lightweight debug event
    debug_ev = {
        "turn":         engine.state.turn_count,
        "user_text":    user_input,
        "phase_before": phase_before,
        "phase_after":  phase_after,
        "kb_mode":      st.session_state.kb_mode,
        "collection":   _active_collection(),
        "diagnosis": {
            "ran":          True,
            "energy_node":  diag.get("energy_node"),
            "confidence":   diag.get("confidence", "unknown"),
        },
        "rag": {
            "ran":            True,
            "query":          user_input,
            "energy_node_filter": diag.get("energy_node"),
            "results_count":  len(rag_captured),
            "results":        rag_captured[:5],
        },
        "llm": {
            "ran":            True,
            "model":          engine.chat_model,
            "used_fallback":  source == "fallback",
            "phase":          phase_before,
            "history_length": len(engine.state.messages),
            "rag_chunks_injected": len(rag_captured),
            "latency_ms":     elapsed_ms,
        },
        "state_after": {
            "phase":            engine.state.phase,
            "energy_node":      engine.state.energy_node,
            "turn_count":       engine.state.turn_count,
            "intent":           engine.state.intent,
            "user_name":        engine.state.user_name,
            "summary_attempted":engine.state.summary_attempted,
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
    if not ev:
        return

    # Phase transition
    st.markdown('<div class="dbg-section-header">Phase Transition</div>', unsafe_allow_html=True)
    pb, pa = ev.get("phase_before", "?"), ev.get("phase_after", "?")
    if pb == pa:
        st.markdown(phase_badge(pb) + ' <span style="color:#94a3b8;">no change</span>', unsafe_allow_html=True)
    else:
        st.markdown(phase_badge(pb) + ' <span style="color:#16a34a;font-size:1rem;">→</span> ' + phase_badge(pa), unsafe_allow_html=True)

    # KB mode used for this turn
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

    # User text
    st.markdown('<div class="dbg-section-header">User Input</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-box mono">{ev.get("user_text","")[:400]}</div>', unsafe_allow_html=True)

    # Diagnosis
    diag = ev.get("diagnosis", {})
    with st.expander("🧠 Diagnosis", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Energy Node**")
            st.markdown(node_badge(diag.get("energy_node")), unsafe_allow_html=True)
        with col2:
            st.markdown("**Confidence**")
            st.markdown(conf_badge(diag.get("confidence", "unknown")), unsafe_allow_html=True)

    # RAG
    rag = ev.get("rag", {})
    rag_count = rag.get("results_count", 0)
    with st.expander(f"🗄️ Qdrant — {rag_count} chunks  [{rag.get('energy_node_filter','no filter')}]", expanded=True):
        results = rag.get("results", [])
        if not results:
            st.markdown('<div class="info-box" style="color:#dc2626;border-left:3px solid #fca5a5;">⚠️ No chunks retrieved.</div>', unsafe_allow_html=True)
        else:
            for i, r in enumerate(results, 1):
                score = r.get("score", 0)
                score_color = "#16a34a" if score > 0.7 else "#d97706" if score > 0.45 else "#dc2626"
                st.markdown(
                    f'<div class="rag-card">'
                    f'<span class="rag-node">[{r.get("energy_node","")}]</span>  '
                    f'<span class="rag-score" style="color:{score_color};">score: {score:.4f}</span>'
                    f'<span class="rag-source">  {r.get("source_video","")}</span>'
                    f'<div class="rag-text">{r.get("text","")[:300]}…</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # LLM
    llm = ev.get("llm", {})
    with st.expander("🤖 LLM Call", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<span class="badge badge-llm">{llm.get("model","?")}</span>', unsafe_allow_html=True)
        if llm.get("used_fallback"):
            c2.markdown('<span class="badge badge-fallback">⚠ Fallback</span>', unsafe_allow_html=True)
        else:
            c2.markdown('<span class="badge badge-ok">✓ LLM OK</span>', unsafe_allow_html=True)
        lat = llm.get("latency_ms")
        if lat:
            color = "#16a34a" if lat < 2000 else "#d97706" if lat < 6000 else "#dc2626"
            c3.markdown(f'<span style="color:{color};font-weight:700;">{lat:.0f} ms</span>', unsafe_allow_html=True)

    # Full state
    with st.expander("📋 Full State After Turn", expanded=False):
        st.json(ev.get("state_after", {}))


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