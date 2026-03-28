"""
Souli Dev Testing Dashboard
============================
Surfaces every known silent failure mode in the pipeline so nothing is invisible during testing.
"""

import streamlit as st
import pandas as pd
import os
import sys
import json
import time
import struct
import traceback
import tempfile
import importlib
from io import StringIO
from typing import Any


OLLAMA_ENDPOINT = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")
QDRANT_HOST     = os.environ.get("QDRANT_HOST",     "localhost")
QDRANT_PORT     = int(os.environ.get("QDRANT_PORT", "6333"))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ok(msg):    st.success(f"✅ {msg}")
def _fail(msg):  st.error(f"❌ {msg}")
def _warn(msg):  st.warning(f"⚠️ {msg}")
def _info(msg):  st.info(f"ℹ️ {msg}")

def _badge(label, color):
    st.markdown(
        f'<span style="background:{color};color:#fff;padding:3px 10px;'
        f'border-radius:12px;font-size:0.78rem;font-weight:600;">{label}</span>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — System Health
# ─────────────────────────────────────────────────────────────────────────────

def _check_system_health():
    st.subheader("1 · System Health")
    st.caption("Checks every service the pipeline depends on — surfaces silent fallbacks explicitly.")

    col1, col2, col3 = st.columns(3)

    # ── Ollama ────────────────────────────────────────────────────────────────
    with col1:
        st.markdown("**Ollama**")
        try:
            import requests
            # r = requests.get("http://localhost:11434/api/tags", timeout=4)
            r = requests.get(f"{OLLAMA_ENDPOINT}/api/tags", timeout=4)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                _ok("Ollama is running")
                st.caption(f"Models: `{'`, `'.join(models) if models else 'none pulled'}`")

                # Check required models
                required = {"llama3.1", "qwen2.5:1.5b"}
                missing = required - set(models)
                if missing:
                    _warn(f"Missing required models: `{', '.join(missing)}`")
                    st.caption("Run: `ollama pull llama3.1` and `ollama pull qwen2.5:1.5b`")
                else:
                    _ok("All required models present")
            else:
                _fail(f"Ollama responded with HTTP {r.status_code}")
        except Exception as e:
            _fail(f"Ollama unreachable: {e}")
            _warn("ConversationEngine will use `fallback_response()` silently — you won't know from the UI")

    # ── Qdrant ────────────────────────────────────────────────────────────────
    with col2:
        st.markdown("**Qdrant**")
        try:
            from qdrant_client import QdrantClient
            # client = QdrantClient(host="localhost", port=6333, timeout=3)
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3)
            collections = [c.name for c in client.get_collections().collections]
            _ok("Qdrant server is reachable")
            st.caption(f"Collections: `{'`, `'.join(collections) if collections else 'none'}`")

            if "souli_chunks" not in collections:
                _warn("`souli_chunks` collection missing — RAG will return 0 results")
                st.caption("Run `souli ingest` to populate it")
            else:
                info = client.get_collection("souli_chunks")
                count = info.points_count
                if count == 0:
                    _warn("`souli_chunks` exists but is EMPTY — RAG returning nothing")
                else:
                    _ok(f"`souli_chunks` has {count:,} vectors")

        except Exception as e:
            _fail(f"Qdrant server unreachable: {e}")
            st.markdown(
                """
                <div style="background:#3a1a1a;padding:10px;border-radius:6px;
                border-left:4px solid #e74c3c;margin-top:6px;">
                <b style="color:#e74c3c;">⚠ Silent Failure Active</b><br>
                <span style="color:#ddd;font-size:0.82rem;">
                qdrant_store.py will silently create an <b>in-memory Qdrant</b>
                that starts empty. All RAG queries return 0 chunks.
                The counselor falls back to generic responses.
                <b>There is no error shown to the user.</b>
                </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── CLI sys import bug ────────────────────────────────────────────────────
    with col3:
        st.markdown("**CLI `sys` Import**")
        try:
            import ast
            cli_path = os.path.join(
                os.path.dirname(__file__), "..", "souli_pipeline", "cli.py"
            )
            if os.path.exists(cli_path):
                src = open(cli_path).read()
                tree = ast.parse(src)
                top_imports = [
                    node.names[0].name
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Import) and node.col_offset == 0
                ]
                if "sys" in top_imports:
                    _ok("`sys` is imported at module level")
                else:
                    _fail("`sys` is NOT imported at top of cli.py")
                    st.markdown(
                        """
                        <div style="background:#3a1a1a;padding:8px;border-radius:6px;
                        border-left:4px solid #e74c3c;margin-top:4px;">
                        <code style="color:#f39c12;">souli voice</code>
                        <span style="color:#ddd;font-size:0.82rem;">
                        will crash with <b>NameError: name 'sys' is not defined</b>
                        at runtime. Your smoke test won't catch this.
                        </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                _warn("cli.py not found at expected path — skipping check")
        except Exception as e:
            _warn(f"Could not parse cli.py: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — CSV Column Name Validator
# ─────────────────────────────────────────────────────────────────────────────

def _check_csv_columns():
    st.subheader("2 · CSV Column Name Validator")
    st.caption(
        "The CLI loader (`videos_csv.py`) and the UI loader (`data_ingestion.py`) "
        "expect **different column names** for the YouTube URL. Upload a CSV to see exactly what breaks."
    )

    # Show the mismatch statically first
    with st.expander("📋 Known column name mismatch (always present)", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**`videos_csv.py` (CLI loader) accepts:**")
            st.code("youtube_url\nurl\nvideo_url\nlink", language="text")
            st.caption("Used by: `souli run videos`, `souli run all`")

        with col2:
            st.markdown("**`data_ingestion.py` (UI loader) accepts:**")
            st.code("yt_links\nyoutube_url", language="text")
            st.caption("Used by: Streamlit Data Ingestion page")

        st.error(
            "❌ `yt_links` is accepted by the UI but **NOT** by the CLI loader. "
            "Your own example CSV (`data/videos.csv`) uses `yt_links` — "
            "it will **fail silently** when run via `souli run videos`."
        )

    # Live CSV upload check
    uploaded = st.file_uploader("Upload your CSV to validate against both loaders", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        cols = set(df.columns.str.strip())
        st.write(f"**Detected columns:** `{list(cols)}`")

        cli_cols   = {"youtube_url", "url", "video_url", "link"}
        ui_cols    = {"yt_links", "youtube_url"}

        cli_match  = bool(cols & cli_cols)
        ui_match   = bool(cols & ui_cols)

        results = {
            "CLI loader (`souli run videos`)":    ("✅ Will work" if cli_match else "❌ Will FAIL — no valid URL column", cli_match),
            "UI loader (Data Ingestion page)":    ("✅ Will work" if ui_match  else "❌ Will FAIL — no valid URL column", ui_match),
        }
        for loader, (msg, ok) in results.items():
            if ok:
                st.success(f"**{loader}**: {msg}")
            else:
                st.error(f"**{loader}**: {msg}")

        if cli_match and not ui_match:
            st.warning("This CSV works in CLI but BREAKS the Streamlit UI ingestion page.")
        if ui_match and not cli_match:
            st.warning("This CSV works in the Streamlit UI but BREAKS the CLI pipeline.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — RAG Pipeline Inspector
# ─────────────────────────────────────────────────────────────────────────────

def _check_rag_pipeline():
    st.subheader("3 · RAG Pipeline Inspector")
    st.caption(
        "Tests an actual Qdrant query and shows exactly what gets injected into the LLM prompt. "
        "Also flags the fake-assistant-message injection pattern."
    )

    query = st.text_input(
        "Test query",
        value="I feel exhausted and nobody values my work",
        key="rag_query",
    )
    node_filter = st.selectbox(
        "Energy node filter (optional)",
        ["(none)", "blocked_energy", "depleted_energy", "scattered_energy",
         "outofcontrol_energy", "normal_energy"],
        key="rag_node",
    )

    if st.button("Run RAG Query", key="rag_btn"):
        node = None if node_filter == "(none)" else node_filter

        # ── Check if Qdrant is actually connected or in-memory ────────────────
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(host="localhost", port=6333, timeout=3)
            client.get_collections()
            qdrant_live = True
        except Exception:
            qdrant_live = False

        if not qdrant_live:
            st.error(
                "Qdrant server is DOWN. `qdrant_store.py` is silently using "
                "in-memory mode. The query below returns 0 results. "
                "The counselor will give generic responses with zero context."
            )

        # ── Run the actual query ──────────────────────────────────────────────
        try:
            from souli_pipeline.retrieval.qdrant_store import query_chunks
            t0 = time.perf_counter()
            chunks = query_chunks(
                user_text=query,
                collection="souli_chunks",
                energy_node=node,
                top_k=3,
            )
            elapsed = time.perf_counter() - t0

            st.caption(f"Query time: {elapsed*1000:.0f} ms · Results: {len(chunks)}")

            if not chunks:
                st.error(
                    "0 chunks returned. The LLM prompt will have NO teaching context. "
                    "Responses will be generic Ollama output, not Souli-specific counseling."
                )
            else:
                for i, c in enumerate(chunks, 1):
                    with st.expander(
                        f"Chunk {i} · score={c.get('score', '?')} · node={c.get('energy_node', '?')}",
                        expanded=i == 1,
                    ):
                        st.write(c.get("text", ""))
                        st.caption(f"Source: {c.get('source_video', 'unknown')}")

        except Exception as e:
            st.error(f"RAG query failed: {e}")
            st.code(traceback.format_exc())

        # ── Show the RAG injection flaw ───────────────────────────────────────
        st.divider()
        st.markdown("**⚠️ RAG Injection Audit**")
        st.warning(
            "RAG context is injected as a **fake assistant message** "
            "(`role: assistant`) in `counselor.py`. "
            "The LLM believes it *said* the teaching content itself. "
            "It will confidently attribute retrieved words to its own prior speech."
        )

        if chunks if 'chunks' in dir() else []:
            injected_msg = {
                "role": "assistant",
                "content": "\n".join(
                    [f"[Relevant teaching from Souli counselor videos:]"]
                    + [f"{i}. {c.get('text','')[:200]}" for i, c in enumerate(chunks[:3], 1)]
                ),
            }
            st.caption("This is what gets injected into the message history:")
            st.json(injected_msg)
            st.caption(
                "✅ Fix: inject as a `system` message or a clearly-labelled `user` turn "
                "like `[CONTEXT]: ...` so the model knows it didn't produce these words."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Conversation State Inspector
# ─────────────────────────────────────────────────────────────────────────────

def _check_conversation_state():
    st.subheader("4 · Conversation Engine State Inspector")
    st.caption(
        "Runs a multi-turn conversation and shows the full internal state after each turn. "
        "Highlights whether responses come from Ollama or the silent fallback."
    )

    # Load engine
    if "dev_engine" not in st.session_state:
        with st.spinner("Loading ConversationEngine..."):
            try:
                config_path = (
                    "configs/pipeline.gcp.yaml"
                    if os.path.exists("configs/pipeline.gcp.yaml")
                    else "configs/pipeline.yaml"
                )
                from souli_pipeline.config_loader import load_config
                from souli_pipeline.conversation.engine import ConversationEngine

                cfg = load_config(config_path)
                st.session_state.dev_engine = ConversationEngine.from_config(cfg)
                st.session_state.dev_turns = []
            except Exception as e:
                st.error(f"Failed to load engine: {e}")
                st.code(traceback.format_exc())
                return

    engine = st.session_state.dev_engine

    # Input
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input("User message", key="dev_chat_input")
    with col2:
        send = st.button("Send", key="dev_send")
        reset = st.button("Reset", key="dev_reset")

    if reset:
        engine.reset()
        st.session_state.dev_turns = []
        st.rerun()

    if send and user_input:
        t0 = time.perf_counter()

        # Detect fallback by checking if Ollama is up
        try:
            import requests
            ollama_up = requests.get(
                f"{OLLAMA_ENDPOINT}/api/tags", timeout=3
            ).status_code == 200
        except Exception:
            ollama_up = False

        # Capture RAG chunks
        rag_captured = []
        _orig = engine._rag_retrieve

        def _capture(q, n):
            chunks = _orig(q, n)
            rag_captured.extend(chunks)
            return chunks

        engine._rag_retrieve = _capture
        try:
            response = engine.turn(user_input)
        except Exception as e:
            response = f"[ENGINE ERROR]: {e}"
        finally:
            engine._rag_retrieve = _orig

        elapsed = time.perf_counter() - t0
        diag = engine.diagnosis_summary

        # Detect dead-code phase counting bug
        dead_code_triggered = diag.get("phase") == "sharing"

        st.session_state.dev_turns.append({
            "user":       user_input,
            "response":   response,
            "state":      dict(diag),
            "rag_count":  len(rag_captured),
            "rag_chunks": rag_captured[:3],
            "ollama_up":  ollama_up,
            "source":     "ollama" if ollama_up else "fallback",
            "elapsed_ms": round(elapsed * 1000),
            "phase_bug":  dead_code_triggered,
        })

    # Render turns
    for i, turn in enumerate(reversed(st.session_state.get("dev_turns", [])), 1):
        with st.expander(f"Turn {len(st.session_state.dev_turns) - i + 1}: {turn['user'][:60]}", expanded=i == 1):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Response**")
                st.write(turn["response"])

            with col2:
                st.markdown("**Internal State**")
                state = turn["state"]

                source_color = "#27ae60" if turn["source"] == "ollama" else "#e74c3c"
                st.markdown(
                    f'<span style="background:{source_color};color:#fff;'
                    f'padding:2px 8px;border-radius:10px;font-size:0.78rem;">'
                    f'{"🧠 Ollama" if turn["source"] == "ollama" else "⚠️ FALLBACK — Ollama offline"}'
                    f'</span>',
                    unsafe_allow_html=True,
                )
                st.json({
                    "phase":        state.get("phase"),
                    "energy_node":  state.get("energy_node"),
                    "confidence":   state.get("confidence"),
                    "intent":       state.get("intent"),
                    "turn_count":   state.get("turn_count"),
                    "rag_chunks":   turn["rag_count"],
                    "elapsed_ms":   turn["elapsed_ms"],
                })

                if turn["rag_count"] == 0:
                    st.error("RAG returned 0 chunks — response has no teaching context")

                # Dead code bug warning
                if turn.get("phase_bug"):
                    st.warning(
                        "⚠️ `_count_turns_in_phase` dead code active: "
                        "first loop result is discarded. Phase tracking may be incorrect."
                    )

                if turn["rag_chunks"]:
                    st.markdown("**RAG Chunks Used**")
                    for c in turn["rag_chunks"]:
                        st.caption(f"[{c.get('score', '?'):.3f}] {c.get('text','')[:100]}…")


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Audio Pipeline Tester
# ─────────────────────────────────────────────────────────────────────────────

def _check_audio_pipeline():
    st.subheader("5 · Audio Pipeline Tester")
    st.caption(
        "Tests TTS output and checks whether LiveKit is being fed raw MP3 bytes "
        "instead of decoded PCM — which produces silence or noise."
    )

    test_text = st.text_input(
        "TTS test text",
        value="Hello, I am Souli, your inner wellness companion.",
        key="tts_text",
    )

    if st.button("Run TTS Test", key="tts_btn"):
        try:
            from souli_pipeline.voice.tts import EdgeTTS

            with st.spinner("Synthesizing audio..."):
                tts = EdgeTTS(voice="en-IN-NeerjaNeural")
                audio_bytes = tts.synthesize(test_text)

            st.success(f"TTS produced {len(audio_bytes):,} bytes")
            st.audio(audio_bytes, format="audio/mp3")

            # ── Check if bytes look like MP3 (ID3 tag or sync frame) ──────────
            is_mp3 = (
                audio_bytes[:3] == b"ID3"              # ID3 tag
                or audio_bytes[:2] == b"\xff\xfb"      # MP3 sync frame
                or audio_bytes[:2] == b"\xff\xf3"
                or audio_bytes[:2] == b"\xff\xf2"
            )

            # ── Check if bytes look like WAV (RIFF header) ────────────────────
            is_wav = audio_bytes[:4] == b"RIFF"

            st.divider()
            st.markdown("**LiveKit Compatibility Audit**")

            if is_mp3:
                st.error(
                    "❌ TTS output is raw **MP3** bytes. "
                    "`livekit_agent.py` feeds these directly into `rtc.AudioFrame(data=audio_bytes)` "
                    "as if they are PCM samples. "
                    "**This produces silence or noise in the LiveKit room.** "
                    "The comment in the code even says 'In production, decode MP3 → PCM' "
                    "— it was never fixed."
                )
                st.code(
                    """# Current broken code in livekit_agent.py:
await source.capture_frame(
    rtc.AudioFrame(
        data=audio_bytes,   # ← raw MP3 bytes, NOT PCM
        sample_rate=24000,
        num_channels=1,
        samples_per_channel=len(audio_bytes) // 2,
    )
)

# Fix: decode MP3 to PCM first
from pydub import AudioSegment
import io
seg = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
seg = seg.set_frame_rate(24000).set_channels(1).set_sample_width(2)
pcm_bytes = seg.raw_data
await source.capture_frame(
    rtc.AudioFrame(
        data=pcm_bytes,
        sample_rate=24000,
        num_channels=1,
        samples_per_channel=len(pcm_bytes) // 2,
    )
)""",
                    language="python",
                )
            elif is_wav:
                st.success("TTS output is WAV — compatible with PCM AudioFrame (after stripping header)")
            else:
                st.warning(
                    f"Unknown audio format (first 4 bytes: {audio_bytes[:4].hex()}). "
                    "Verify before passing to LiveKit."
                )

            # Show byte signature
            st.caption(
                f"First 16 bytes: `{audio_bytes[:16].hex()}` "
                f"| Format detected: {'MP3' if is_mp3 else 'WAV' if is_wav else 'Unknown'}"
            )

        except Exception as e:
            st.error(f"TTS failed: {e}")
            st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Energy Diagnosis Confidence
# ─────────────────────────────────────────────────────────────────────────────

def _check_energy_diagnosis():
    st.subheader("6 · Energy Diagnosis Confidence Tester")
    st.caption(
        "Tests the energy node classifier and shows whether it used embedding matching "
        "or keyword fallback — so you know how accurate the diagnosis actually is."
    )

    test_inputs = {
        "Custom input": "",
        "Exhausted & undervalued": "I am so tired all the time, nobody appreciates what I do",
        "Angry & reactive": "I keep exploding at people and I can't control my anger",
        "Stuck & withdrawn": "I just don't want to do anything anymore, I feel numb",
        "Overwhelmed & scattered": "There is too much happening, I can't focus on anything",
    }

    preset = st.selectbox("Test preset", list(test_inputs.keys()), key="diag_preset")
    text = st.text_area(
        "Input text",
        value=test_inputs[preset] if preset != "Custom input" else "",
        key="diag_text",
        height=80,
    )

    if st.button("Diagnose", key="diag_btn") and text.strip():
        col1, col2 = st.columns(2)

        # ── Keyword fallback ──────────────────────────────────────────────────
        with col1:
            st.markdown("**Keyword Fallback Result**")
            from souli_pipeline.energy.normalize import infer_node
            kw_node = infer_node(text, "")
            st.json({"energy_node": kw_node, "method": "keyword_heuristic"})
            st.caption(
                "This is always used when Qdrant/gold.xlsx is unavailable. "
                "No similarity score — just keyword matching."
            )

        # ── Embedding-based diagnosis ─────────────────────────────────────────
        with col2:
            st.markdown("**Embedding-based Diagnosis**")
            try:
                from souli_pipeline.retrieval.embedding import embed_one, available

                if not available():
                    _warn("`sentence-transformers` not installed — embedding diagnosis unavailable")
                    st.caption("Install: `pip install sentence-transformers`")
                else:
                    emb = embed_one(text)
                    if emb:
                        st.success(f"Embedding produced ({len(emb)}-dim vector)")
                        st.caption(f"First 5 dims: {[round(v, 4) for v in emb[:5]]}")

                        # Try full gold-based diagnosis if gold exists
                        gold_path = None
                        if os.path.exists("outputs"):
                            for run in sorted(os.listdir("outputs"), reverse=True):
                                gp = os.path.join("outputs", run, "energy", "gold.xlsx")
                                if os.path.exists(gp):
                                    gold_path = gp
                                    break

                        if gold_path:
                            from souli_pipeline.retrieval.match import (
                                load_gold, diagnose,
                            )
                            nodes = [
                                "blocked_energy", "depleted_energy", "scattered_energy",
                                "outofcontrol_energy", "normal_energy",
                            ]
                            gold_df = load_gold(gold_path, nodes)
                            result = diagnose(text, gold_df, nodes)
                            st.json({
                                "energy_node":      result.get("energy_node"),
                                "confidence":       result.get("confidence"),
                                "similarity":       result.get("similarity"),
                                "matched_problem":  result.get("matched_problem", "")[:80],
                            })
                            if result.get("confidence") == "keyword_fallback":
                                _warn(
                                    "Similarity below threshold — fell back to keyword heuristic. "
                                    "Consider adding more gold examples for this type of problem."
                                )
                        else:
                            _warn("No gold.xlsx found — run `souli run energy` first")
                    else:
                        _fail("Embedding returned None")
            except Exception as e:
                st.error(f"Embedding diagnosis failed: {e}")
                st.code(traceback.format_exc())

        # ── normalize_node dead mapping audit ─────────────────────────────────
        st.divider()
        st.markdown("**`normalize_node` Dead Mapping Audit**")
        phantom_entries = ["depletedenergy", "outofcontrol", "outofcontrolenergy"]
        st.warning(
            "The `normalize_node` function contains mappings like `depletedenergy` "
            "that can never appear after the cleaning steps already applied "
            "(spaces stripped, `/` replaced). These are debug artifacts that were never removed."
        )
        from souli_pipeline.energy.normalize import normalize_node
        nodes = ["blocked_energy", "depleted_energy", "scattered_energy",
                 "outofcontrol_energy", "normal_energy"]
        rows = []
        for phantom in phantom_entries:
            result = normalize_node(phantom, nodes)
            rows.append({"Phantom Input": phantom, "Result": result, "Reachable?": "❓ Never"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Working Directory Side-Effect Check
# ─────────────────────────────────────────────────────────────────────────────

def _check_filesystem_sideeffects():
    st.subheader("7 · Filesystem Side-Effect Check")
    st.caption(
        "Checks for the `download_captions()` bug where VTT files are deleted from "
        "the current working directory before every download."
    )

    cwd = os.getcwd()
    vtt_files = [f for f in os.listdir(cwd) if f.endswith(".vtt")]

    if vtt_files:
        st.error(
            f"Found {len(vtt_files)} `.vtt` files in the current directory: "
            f"`{', '.join(vtt_files)}`. "
            "The next call to `download_captions()` will **silently delete all of them** "
            "regardless of which video you're processing."
        )
        for f in vtt_files:
            st.code(f"Would delete: {os.path.join(cwd, f)}")
    else:
        st.success("No `.vtt` files in current directory — safe from the deletion bug right now.")

    with st.expander("See the problematic code"):
        st.code(
            """# captions.py — runs on EVERY download_captions() call
for f in os.listdir():          # scans CURRENT WORKING DIRECTORY
    if f.endswith(".vtt"):
        try: os.remove(f)       # silently deletes it
        except: pass            # swallows ALL errors including PermissionError""",
            language="python",
        )
        st.markdown(
            "**Fix:** Use `tempfile.TemporaryDirectory()` so files are isolated "
            "per-call and cleaned up automatically."
        )

    st.divider()
    st.markdown("**Broad `except: pass` Counter**")
    st.caption("Counts silent exception swallowers across the pipeline source.")

    src_dir = os.path.join(os.path.dirname(__file__), "..", "souli_pipeline")
    if os.path.exists(src_dir):
        import ast
        bare_excepts = []
        for root, _, files in os.walk(src_dir):
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    tree = ast.parse(open(fpath).read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ExceptHandler):
                            if node.type is None:  # bare except:
                                bare_excepts.append(
                                    f"{os.path.relpath(fpath, src_dir)}:{node.lineno}"
                                )
                except Exception:
                    pass

        if bare_excepts:
            st.error(f"Found {len(bare_excepts)} bare `except:` blocks:")
            st.dataframe(
                pd.DataFrame(bare_excepts, columns=["Location"]),
                use_container_width=True,
            )
        else:
            st.success("No bare `except:` blocks found.")
    else:
        _warn("souli_pipeline source not found at expected path")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def show():
    st.header("🔬 Dev Testing Dashboard")
    st.markdown(
        """
        <div style="background:#1a2a1a;border-left:4px solid #6fcf97;
        padding:12px 16px;border-radius:6px;margin-bottom:20px;">
        <b style="color:#6fcf97;">Purpose:</b>
        <span style="color:#ccc;">
        Every silent failure mode in the pipeline is surfaced here.
        Run these checks before testing a new deployment.
        A green board here means failures will be loud — not swallowed.
        </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sections = {
        "1 · System Health":                _check_system_health,
        "2 · CSV Column Validator":         _check_csv_columns,
        "3 · RAG Pipeline Inspector":       _check_rag_pipeline,
        "4 · Conversation State Inspector": _check_conversation_state,
        "5 · Audio Pipeline Tester":        _check_audio_pipeline,
        "6 · Energy Diagnosis Confidence":  _check_energy_diagnosis,
        "7 · Filesystem Side-Effects":      _check_filesystem_sideeffects,
    }

    selected = st.multiselect(
        "Run checks",
        list(sections.keys()),
        default=list(sections.keys()),
        key="dev_sections",
    )

    st.divider()

    for name, fn in sections.items():
        if name in selected:
            try:
                fn()
            except Exception as e:
                st.error(f"Check `{name}` crashed: {e}")
                st.code(traceback.format_exc())
            st.divider()