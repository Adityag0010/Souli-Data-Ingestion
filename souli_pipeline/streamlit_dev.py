"""
Souli — Developer / Tester Debug UI
====================================
Run:  streamlit run souli_pipeline/streamlit_dev.py

Two-column layout:
  LEFT  (40%)  — live debug panel: phase flow, diagnosis, full Qdrant
                 results, LLM call details, turn history, session state,
                 and a standalone Qdrant inspector
  RIGHT (60%)  — normal chat UI (text + voice tabs)

Drop the instrumented engine.py alongside this file as:
  souli_pipeline/conversation/engine.py
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = os.environ.get(
    "SOULI_CONFIG_PATH",
    str(Path(__file__).parent.parent / "configs" / "pipeline.gcp.yaml"),
)
GOLD_PATH  = os.environ.get("SOULI_GOLD_PATH", None)
_default_excel = str(Path(__file__).parent / "data" / "Souli_EnergyFramework_PW (1).xlsx")
EXCEL_PATH = os.environ.get(
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
/* ── overall ── */
[data-testid="stAppViewContainer"] { background: #0d0f14; }
[data-testid="stMain"] { background: #0d0f14; }
section[data-testid="stSidebar"] { display: none; }

/* ── typography ── */
body, p, div, span { color: #d0d6e0; }
h1, h2, h3, h4 { color: #e8ecf4; }

/* ── left panel container ── */
.debug-panel {
    background: #111318;
    border-right: 1px solid #1e2230;
    min-height: 100vh;
    padding: 0 0 40px 0;
}

/* ── section headers inside left panel ── */
.dbg-section-header {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #4a5568;
    margin: 18px 0 6px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #1e2230;
}

/* ── badges ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.73rem;
    font-weight: 600;
    margin: 2px 2px;
}
.badge-phase    { background: #1a2744; color: #7eb8f7; }
.badge-node     { background: #132a1e; color: #56c785; }
.badge-fallback { background: #2a1218; color: #f87171; }
.badge-embed    { background: #1a2030; color: #818cf8; }
.badge-kw       { background: #2a2010; color: #fbbf24; }
.badge-llm      { background: #0e2230; color: #38bdf8; }
.badge-ok       { background: #102a1a; color: #4ade80; }
.badge-warn     { background: #2a1a08; color: #fb923c; }
.badge-neutral  { background: #1e2230; color: #94a3b8; }

/* ── rag chunk card ── */
.rag-card {
    background: #161b26;
    border: 1px solid #1e2a3a;
    border-radius: 6px;
    padding: 10px 12px;
    margin: 6px 0;
    font-size: 0.78rem;
}
.rag-score   { color: #38bdf8; font-weight: 700; }
.rag-node    { color: #56c785; font-size: 0.7rem; }
.rag-source  { color: #94a3b8; font-size: 0.68rem; }
.rag-text    { color: #c8d0dc; line-height: 1.5; margin-top: 5px; }

/* ── phase timeline ── */
.phase-step {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 0.72rem;
}
.phase-arrow { color: #4a5568; margin: 0 3px; }

/* ── turn row in history ── */
.turn-row {
    background: #141820;
    border: 1px solid #1e2230;
    border-radius: 5px;
    padding: 7px 10px;
    margin: 4px 0;
    font-size: 0.76rem;
    cursor: pointer;
}
.turn-row:hover { border-color: #334466; }

/* ── chat bubble tweaks ── */
[data-testid="stChatMessage"] { background: transparent !important; }

/* ── info boxes ── */
.info-box {
    background: #141820;
    border: 1px solid #1e2230;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.8rem;
}

/* ── mono code ── */
.mono { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #7dd3fc; }

/* ── divider ── */
.dbg-divider { border: none; border-top: 1px solid #1e2230; margin: 12px 0; }
</style>
""", unsafe_allow_html=True)


# ── Helpers: cached resources ─────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading config...")
def _load_config():
    from souli_pipeline.config_loader import load_config
    return load_config(CONFIG_PATH)


def get_engine():
    if "engine" not in st.session_state:
        from souli_pipeline.conversation.engine import ConversationEngine
        cfg = _load_config()
        st.session_state.engine = ConversationEngine.from_config(
            cfg, gold_path=GOLD_PATH, excel_path=EXCEL_PATH
        )
    return st.session_state.engine


@st.cache_resource(show_spinner="Loading Whisper STT...")
def get_stt():
    from souli_pipeline.voice.stt import WhisperSTT
    return WhisperSTT(model_name="base")


@st.cache_resource(show_spinner="Loading Edge TTS...")
def get_tts():
    from souli_pipeline.voice.tts import EdgeTTS
    return EdgeTTS(voice="en-IN-NeerjaNeural")


def init_session():
    engine = get_engine()
    if "messages" not in st.session_state:
        greeting = engine.greeting()
        st.session_state.messages = [{"role": "assistant", "content": greeting}]
    if "voice_messages" not in st.session_state:
        greeting = st.session_state.messages[0]["content"]
        st.session_state.voice_messages = [{"role": "assistant", "content": greeting}]
    if "selected_turn_idx" not in st.session_state:
        st.session_state.selected_turn_idx = None


# ── Helpers: badge rendering ──────────────────────────────────────────────────

_NODE_COLORS = {
    "blocked_energy":      ("#132a1e", "#e74c3c"),
    "depleted_energy":     ("#2a1a08", "#e67e22"),
    "scattered_energy":    ("#2a2008", "#f1c40f"),
    "outofcontrol_energy": ("#1e1030", "#9b59b6"),
    "normal_energy":       ("#0e2a1a", "#27ae60"),
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
    bg, border = _NODE_COLORS.get(node, ("#1e2230", "#94a3b8"))
    label = node.replace("_", " ").title()
    return (f'<span class="badge" style="background:{bg};'
            f'color:{border};border-left:3px solid {border};">{label}</span>')


def conf_badge(confidence: str) -> str:
    if confidence == "embedding_match":
        return '<span class="badge badge-embed">embedding match</span>'
    if "keyword" in confidence:
        return '<span class="badge badge-kw">keyword fallback</span>'
    return f'<span class="badge badge-neutral">{confidence}</span>'


# ── Engine turn runner ────────────────────────────────────────────────────────

def run_turn(user_input: str) -> tuple[str, bool]:
    """Run one engine turn. Returns (response_text, used_streaming)."""
    engine = get_engine()
    full_response = ""
    used_stream = True
    try:
        for chunk in engine.turn_stream(user_input):
            full_response += chunk
    except Exception:
        full_response = engine.turn(user_input)
        used_stream = False
    return full_response, used_stream


# ── Debug panel renderers ─────────────────────────────────────────────────────

def render_phase_flow():
    """Render a compact phase timeline across all turns."""
    engine = get_engine()
    events = engine._debug_events
    if not events:
        st.markdown('<span style="color:#4a5568;font-size:0.78rem;">No turns yet.</span>',
                    unsafe_allow_html=True)
        return

    # Build phase transition chain
    parts = []
    for ev in events:
        pb = ev.get("phase_before", "?")
        pa = ev.get("phase_after", "?")
        if pb == pa:
            parts.append(f'<span class="badge badge-phase" style="font-size:0.65rem;">{_PHASE_LABELS.get(pb, pb)}</span>')
        else:
            parts.append(
                f'<span class="badge badge-phase" style="font-size:0.65rem;">{_PHASE_LABELS.get(pb, pb)}</span>'
                f'<span class="phase-arrow">→</span>'
                f'<span class="badge badge-warn" style="font-size:0.65rem;">{_PHASE_LABELS.get(pa, pa)}</span>'
            )

    html = '<div style="display:flex;flex-wrap:wrap;gap:4px;align-items:center;">'
    for i, p in enumerate(parts):
        html += p
        if i < len(parts) - 1:
            html += '<span class="phase-arrow" style="color:#2a3a55;">┆</span>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_turn_debug(ev: Dict[str, Any], collapsed: bool = False):
    """Render all debug sections for one turn event."""
    if not ev:
        return

    # ── Phase transition ──────────────────────────────────────────────
    st.markdown('<div class="dbg-section-header">Phase Transition</div>', unsafe_allow_html=True)
    pb = ev.get("phase_before", "?")
    pa = ev.get("phase_after", "?")
    if pb == pa:
        st.markdown(phase_badge(pb) + ' <span style="color:#4a5568;">no change</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown(phase_badge(pb) + ' <span style="color:#56c785;font-size:1rem;">→</span> ' + phase_badge(pa),
                    unsafe_allow_html=True)

    # ── User text ─────────────────────────────────────────────────────
    st.markdown('<div class="dbg-section-header">User Input</div>', unsafe_allow_html=True)
    user_text = ev.get("user_text", "")
    st.markdown(
        f'<div class="info-box mono">{user_text[:400]}</div>',
        unsafe_allow_html=True
    )

    # ── Diagnosis ─────────────────────────────────────────────────────
    diag = ev.get("diagnosis", {})
    with st.expander("🧠 Diagnosis", expanded=True):
        if not diag.get("ran"):
            st.markdown('<span style="color:#4a5568;">Not run this turn.</span>',
                        unsafe_allow_html=True)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Energy Node**")
                st.markdown(node_badge(diag.get("energy_node")), unsafe_allow_html=True)
            with col2:
                st.markdown("**Confidence**")
                st.markdown(conf_badge(diag.get("confidence", "unknown")), unsafe_allow_html=True)

            sim = diag.get("similarity")
            if sim is not None:
                st.markdown(f"**Similarity Score:** `{sim:.4f}`")

            method = diag.get("method", "")
            st.markdown(f"**Method:** `{method}`")

            matched = diag.get("matched_problem")
            if matched:
                st.markdown("**Matched Gold Problem Statement:**")
                st.markdown(
                    f'<div class="info-box" style="border-color:#1a3a55;">'
                    f'<span style="color:#7eb8f7;">{matched}</span></div>',
                    unsafe_allow_html=True
                )

            fw = diag.get("framework_row_preview", {})
            if fw:
                with st.expander("Framework row (gold match)", expanded=False):
                    for k, v in fw.items():
                        if v and v != "nan":
                            st.markdown(f"**{k}:** {v}")

            snippet = diag.get("input_snippet", "")
            if snippet:
                with st.expander("Input text sent to diagnose()", expanded=False):
                    st.markdown(
                        f'<div class="info-box mono" style="font-size:0.7rem;">'
                        f'{snippet[-600:]}</div>',
                        unsafe_allow_html=True
                    )

            err = diag.get("error")
            if err:
                st.error(f"Diagnosis error: {err}")

    # ── Qdrant / RAG ──────────────────────────────────────────────────
    rag = ev.get("rag", {})
    rag_count = rag.get("results_count", len(rag.get("results", [])))
    rag_label = f"🗄️ Qdrant — {rag_count} chunks retrieved"
    with st.expander(rag_label, expanded=True):
        if not rag.get("ran"):
            st.markdown('<span style="color:#4a5568;">RAG not run this turn.</span>',
                        unsafe_allow_html=True)
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Query**")
                q = rag.get("query", "")
                st.markdown(
                    f'<span class="mono" style="font-size:0.7rem;">{q[:80]}...</span>'
                    if len(q) > 80 else
                    f'<span class="mono" style="font-size:0.7rem;">{q}</span>',
                    unsafe_allow_html=True
                )
            with c2:
                st.markdown("**Filter**")
                st.markdown(node_badge(rag.get("energy_node_filter")), unsafe_allow_html=True)
            with c3:
                st.markdown("**Latency**")
                lat = rag.get("latency_ms")
                if lat is not None:
                    color = "#4ade80" if lat < 200 else "#fb923c" if lat < 800 else "#f87171"
                    st.markdown(f'<span style="color:{color};font-weight:700;">{lat:.0f} ms</span>',
                                unsafe_allow_html=True)

            err = rag.get("error")
            if err:
                st.warning(f"Qdrant error: {err}")

            results = rag.get("results", [])
            if not results:
                st.markdown(
                    '<div class="info-box" style="color:#f87171;">'
                    '⚠️ No chunks retrieved. Qdrant may be empty or not running.</div>',
                    unsafe_allow_html=True
                )
            else:
                for i, r in enumerate(results, 1):
                    score = r.get("score", 0)
                    score_color = (
                        "#4ade80" if score > 0.7 else
                        "#fbbf24" if score > 0.45 else
                        "#f87171"
                    )
                    node = r.get("energy_node", "")
                    source = r.get("source_video", "")
                    text = r.get("text", "")

                    st.markdown(
                        f'<div class="rag-card">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'<span><b style="color:#94a3b8;">#{i}</b> '
                        f'<span class="rag-node">[{node}]</span></span>'
                        f'<span class="rag-score" style="color:{score_color};">score: {score:.4f}</span>'
                        f'</div>'
                        f'<div class="rag-source">{source}</div>'
                        f'<div class="rag-text">{text[:350]}{"..." if len(text) > 350 else ""}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    # ── LLM call ──────────────────────────────────────────────────────
    llm = ev.get("llm", {})
    with st.expander("🤖 LLM Call", expanded=True):
        if not llm.get("ran"):
            st.markdown('<span style="color:#4a5568;">LLM not called this turn.</span>',
                        unsafe_allow_html=True)
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Model**")
                st.markdown(f'<span class="badge badge-llm">{llm.get("model","?")}</span>',
                            unsafe_allow_html=True)
            with c2:
                fallback = llm.get("used_fallback", False)
                st.markdown("**Status**")
                if fallback:
                    st.markdown('<span class="badge badge-fallback">⚠ Fallback</span>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge badge-ok">✓ LLM OK</span>',
                                unsafe_allow_html=True)
            with c3:
                lat = llm.get("latency_ms")
                st.markdown("**Latency**")
                if lat is not None:
                    color = "#4ade80" if lat < 2000 else "#fb923c" if lat < 6000 else "#f87171"
                    st.markdown(f'<span style="color:{color};font-weight:700;">{lat:.0f} ms</span>',
                                unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"**History msgs:** `{llm.get('history_length', 0)}`")
            with c2:
                st.markdown(f"**RAG injected:** `{llm.get('rag_chunks_injected', 0)}`")
            with c3:
                st.markdown(f"**Phase:** `{llm.get('phase', '?')}`")

            topics = llm.get("asked_topics", [])
            if topics:
                st.markdown(f"**Topics already asked:** `{', '.join(topics)}`")

            fallback_reason = llm.get("fallback_reason", "")
            if fallback_reason:
                st.error(f"Fallback reason: {fallback_reason}")

            sys_prompt = llm.get("system_prompt", "")
            if sys_prompt:
                with st.expander("System prompt sent", expanded=False):
                    st.markdown(
                        f'<div class="info-box mono" style="white-space:pre-wrap;">'
                        f'{sys_prompt}</div>',
                        unsafe_allow_html=True
                    )

    # ── State after turn ──────────────────────────────────────────────
    state = ev.get("state_after", {})
    with st.expander("📋 Full State After Turn", expanded=False):
        st.json(state)


def render_qdrant_inspector(cfg):
    """Standalone Qdrant query tool — test retrieval independent of chat."""
    st.markdown("### 🔍 Qdrant Inspector")
    st.caption("Run queries directly against your vector store, independent of the conversation.")

    col1, col2 = st.columns([3, 1])
    with col1:
        query_text = st.text_area("Query text", height=80,
                                  placeholder="Type any text to search for similar chunks...")
    with col2:
        node_options = ["(no filter)"] + (cfg.energy.nodes_allowed if cfg else [
            "blocked_energy", "depleted_energy", "scattered_energy",
            "outofcontrol_energy", "normal_energy",
        ])
        node_filter = st.selectbox("Energy node filter", node_options)
        top_k = st.slider("Top K", 1, 15, 5)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        r = cfg.retrieval if cfg else None
        host = st.text_input("Qdrant host", value=r.qdrant_host if r else "localhost")
    with col_b:
        port = st.number_input("Port", value=r.qdrant_port if r else 6333, step=1)
    with col_c:
        collection = st.text_input("Collection", value=r.qdrant_collection if r else "souli_chunks")

    if st.button("🔍 Run Query", type="primary", use_container_width=True):
        if not query_text.strip():
            st.warning("Enter some query text first.")
            return

        with st.spinner("Querying Qdrant..."):
            try:
                from souli_pipeline.retrieval.qdrant_store import query_chunks
                emb_model = r.embedding_model if r else "sentence-transformers/all-MiniLM-L6-v2"
                node = None if node_filter == "(no filter)" else node_filter

                t0 = time.time()
                results = query_chunks(
                    user_text=query_text,
                    collection=collection,
                    energy_node=node,
                    top_k=top_k,
                    embedding_model=emb_model,
                    host=host,
                    port=int(port),
                    score_threshold=0.0,   # show everything for inspection
                )
                latency = (time.time() - t0) * 1000

                st.success(f"Retrieved {len(results)} chunks in {latency:.0f} ms")

                if not results:
                    st.info("No results. The collection may be empty or the query doesn't match anything.")
                    return

                for i, r_item in enumerate(results, 1):
                    score = r_item.get("score", 0)
                    score_color = (
                        "#4ade80" if score > 0.7 else
                        "#fbbf24" if score > 0.45 else
                        "#f87171"
                    )
                    with st.expander(
                        f"#{i} — score: {score:.4f}  [{r_item.get('energy_node', '')}]  "
                        f"{r_item.get('source_video', '')[:50]}",
                        expanded=(i <= 3),
                    ):
                        st.markdown(
                            f'<span class="rag-score" style="color:{score_color};">'
                            f'Score: {score:.4f}</span>  '
                            f'<span class="rag-node">[{r_item.get("energy_node","")}]</span>  '
                            f'<span class="rag-source">{r_item.get("source_video","")}</span>',
                            unsafe_allow_html=True
                        )
                        st.markdown(r_item.get("text", ""))
                        st.caption(f"URL: {r_item.get('youtube_url','—')}")

            except Exception as exc:
                st.error(f"Qdrant error: {exc}")
                st.info("Make sure Qdrant is running and the collection has been ingested.")


def render_session_state_tab():
    """Full session state dump and message history."""
    engine = get_engine()
    s = engine.state

    st.markdown("### 🗃️ Full ConversationState")
    st.json({
        "phase": s.phase,
        "turn_count": s.turn_count,
        "user_name": s.user_name,
        "energy_node": s.energy_node,
        "node_confidence": s.node_confidence,
        "intent": s.intent,
        "summary_attempted": s.summary_attempted,
        "summary_confirmed": s.summary_confirmed,
        "rich_opening": s.rich_opening,
        "short_answer_count": s.short_answer_count,
        "used_probe_indices": s.used_probe_indices,
        "used_sharing_probe_indices": s.used_sharing_probe_indices,
        "user_text_buffer_words": len(s.user_text_buffer.split()),
        "messages_count": len(s.messages),
    })

    st.markdown("### 💬 Message History (LLM context)")
    for i, msg in enumerate(s.messages):
        role = msg["role"]
        color = "#7eb8f7" if role == "user" else "#56c785"
        icon = "👤" if role == "user" else "🌿"
        with st.expander(f"{icon} [{i}] {role}", expanded=False):
            st.markdown(
                f'<div class="info-box" style="color:{color};">{msg["content"]}</div>',
                unsafe_allow_html=True
            )

    st.markdown("### 📝 User Text Buffer")
    st.markdown(
        f'<div class="info-box mono" style="font-size:0.72rem;white-space:pre-wrap;">'
        f'{s.user_text_buffer.strip()}</div>',
        unsafe_allow_html=True
    )


def render_turn_history_tab():
    """Compact table of all turns with click-to-expand detail."""
    engine = get_engine()
    events = engine._debug_events

    if not events:
        st.markdown('<span style="color:#4a5568;">No turns yet — start chatting.</span>',
                    unsafe_allow_html=True)
        return

    st.markdown(f"### {len(events)} turns recorded")

    for ev in reversed(events):
        turn_n = ev.get("turn", "?")
        pb = _PHASE_LABELS.get(ev.get("phase_before", ""), ev.get("phase_before", "?"))
        pa = _PHASE_LABELS.get(ev.get("phase_after", ""), ev.get("phase_after", "?"))
        node = ev.get("state_after", {}).get("energy_node") or ev.get("diagnosis", {}).get("energy_node")
        rag_n = ev.get("rag", {}).get("results_count", 0)
        fallback = ev.get("llm", {}).get("used_fallback", False)
        conf = ev.get("diagnosis", {}).get("confidence", "—")
        user_snippet = ev.get("user_text", "")[:60] + ("..." if len(ev.get("user_text","")) > 60 else "")

        phase_changed = pb != pa
        label = (
            f"Turn #{turn_n} | {pb}"
            + (f" → **{pa}**" if phase_changed else "")
            + f" | {node or '?'} | RAG: {rag_n}"
            + (" | ⚠ FALLBACK" if fallback else "")
        )

        with st.expander(label, expanded=False):
            st.caption(f'User: "{user_snippet}"')
            st.caption(f"Confidence: {conf}")
            render_turn_debug(ev)


# ── Main layout ───────────────────────────────────────────────────────────────

init_session()

# Top bar
top_left, top_right = st.columns([1, 2])
with top_left:
    st.markdown("## 🔬 Souli Dev")
    st.caption("Internal developer & tester debug interface")
with top_right:
    engine_ref = get_engine()
    diag = engine_ref.diagnosis_summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Turn", diag.get("turn_count", 0))
    c2.metric("Phase", _PHASE_LABELS.get(diag.get("phase", ""), "—"))
    c3.metric("Node", (diag.get("energy_node") or "—").replace("_energy", "").replace("_", " ").title())
    c4.metric("Confidence", diag.get("confidence", "—"))

st.divider()

# Two-column layout
left_col, right_col = st.columns([4, 5], gap="medium")

# ═══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — DEBUG PANEL
# ═══════════════════════════════════════════════════════════════════════════════
with left_col:
    cfg_obj = _load_config()

    # Phase flow timeline (always visible)
    st.markdown('<div class="dbg-section-header">Phase Flow (all turns)</div>',
                unsafe_allow_html=True)
    render_phase_flow()
    st.markdown('<hr class="dbg-divider"/>', unsafe_allow_html=True)

    # Tabs for the four inspection modes
    tab_current, tab_history, tab_qdrant, tab_session = st.tabs([
        "📍 Current Turn",
        "🕑 Turn History",
        "🗄️ Qdrant Inspector",
        "🗃️ Session State",
    ])

    with tab_current:
        engine_ref = get_engine()
        latest = engine_ref.latest_debug
        if not latest:
            st.markdown(
                '<div style="color:#4a5568;padding:20px 0;">'
                'No turns yet. Send a message to start seeing debug info.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="dbg-section-header">Turn #{latest.get("turn","?")} details</div>',
                unsafe_allow_html=True
            )
            render_turn_debug(latest)

    with tab_history:
        render_turn_history_tab()

    with tab_qdrant:
        render_qdrant_inspector(cfg_obj)

    with tab_session:
        render_session_state_tab()

# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — CHAT
# ═══════════════════════════════════════════════════════════════════════════════
with right_col:
    st.markdown("## 🌿 Souli")
    st.caption("Your inner wellness companion  ·  [dev mode]")

    # Reset + config info on one row
    ctrl_l, ctrl_r = st.columns([3, 1])
    with ctrl_r:
        if st.button("↺ Reset", use_container_width=True, help="Clear conversation and start fresh"):
            for key in ["messages", "voice_messages", "engine"]:
                st.session_state.pop(key, None)
            st.rerun()
    with ctrl_l:
        st.caption(f"Config: `{CONFIG_PATH}`  |  Gold: `{GOLD_PATH or 'none'}`")

    chat_tab, voice_tab = st.tabs(["💬 Text Chat", "🎤 Voice Chat"])

    # ── Text Chat ─────────────────────────────────────────────────────────────
    with chat_tab:
        # Render existing messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if user_input := st.chat_input("Share what's on your mind...", key="text_input"):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("_Souli is with you..._")
                full_response, _ = run_turn(user_input)
                placeholder.write(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun()   # rerun so debug panel updates

    # ── Voice Chat ────────────────────────────────────────────────────────────
    with voice_tab:
        for msg in st.session_state.voice_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg["role"] == "assistant" and "audio" in msg:
                    st.audio(msg["audio"], format="audio/mp3")

        audio_input = st.audio_input("Press to record", key="voice_input")
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
                st.session_state.voice_messages.append({"role": "user", "content": transcript})
                with st.spinner("Souli is thinking..."):
                    response, _ = run_turn(transcript)
                with st.spinner("Generating voice..."):
                    tts = get_tts()
                    audio_bytes = tts.synthesize(response)
                st.session_state.voice_messages.append({
                    "role": "assistant", "content": response, "audio": audio_bytes
                })
                st.rerun()
            else:
                st.warning("Could not transcribe. Try again.")

        if voice_text := st.chat_input("Or type here...", key="voice_text_input"):
            st.session_state.voice_messages.append({"role": "user", "content": voice_text})
            with st.spinner("Souli is thinking..."):
                response, _ = run_turn(voice_text)
            with st.spinner("Generating voice..."):
                tts = get_tts()
                audio_bytes = tts.synthesize(response)
            st.session_state.voice_messages.append({
                "role": "assistant", "content": response, "audio": audio_bytes
            })
            st.rerun()