"""
pages/multi_data_ingestion_improved.py

Multi-Collection Ingestion UI
Runs the full multi-ingestion pipeline:
  Whisper → Topic Segment → Clean → Persona → Energy Tag
  → General Ingest (souli_chunks_improved)
  → Density Detection
  → 5 Typed Extractors
  → 5 Typed Collection Ingest

Shows step-by-step results after each video.
Mirrors the structure of data_ingestion_improved.py.
"""
import json
import os

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Helpers (same pattern as data_ingestion_improved.py)
# ---------------------------------------------------------------------------

def _read_excel_safe(path: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path)
    except Exception:
        return pd.DataFrame()


def _read_json_safe(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _read_text_safe(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def _validate_csv(df: pd.DataFrame) -> bool:
    required = {"yt_links", "youtube_url", "url"}
    return bool(set(df.columns) & required)


def _get_url(row) -> str:
    return row.get("yt_links") or row.get("youtube_url") or row.get("url") or ""


def _create_example_csv() -> str:
    df = pd.DataFrame({
        "url":   ["https://youtu.be/VIDEO_ID_1", "https://youtu.be/VIDEO_ID_2"],
        "name":  ["session_01_blocked",           "session_02_depleted"],
        "title": ["Blocked Energy Session 1",     "Depleted Energy Session 2"],
    })
    return df.to_csv(index=False)


def _download_excel_btn(path: str, label: str, key: str):
    try:
        with open(path, "rb") as f:
            st.download_button(
                label=f"⬇ {label}",
                data=f.read(),
                file_name=os.path.basename(path),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=key,
            )
    except Exception:
        pass


def _retention_badge(orig: int, cleaned: int) -> str:
    if orig == 0:
        return "N/A"
    pct = int(cleaned / orig * 100)
    if pct >= 75:
        return f"✅ {pct}%"
    elif pct >= 55:
        return f"⚠️ {pct}%"
    return f"❌ {pct}%"


# ---------------------------------------------------------------------------
# Step renderers
# ---------------------------------------------------------------------------

def _render_step1(whisper_path: str, vid: int):
    st.markdown("#### Step 1 — Whisper Transcription")
    df = _read_excel_safe(whisper_path)
    if df.empty:
        st.warning("whisper_segments.xlsx not found or empty")
        return
    col1, col2 = st.columns(2)
    col1.metric("Raw segments", len(df))
    if "end" in df.columns and "start" in df.columns:
        duration = df["end"].max() - df["start"].min()
        col2.metric("Duration (min)", f"{duration / 60:.1f}")
    st.dataframe(df.head(5), use_container_width=True)
    _download_excel_btn(whisper_path, "whisper_segments.xlsx", f"dl_whisper_{vid}")


def _render_step2(paragraphs_path: str, topics_path: str, vid: int):
    st.markdown("#### Step 2 — Topic Segmentation")
    df_p = _read_excel_safe(paragraphs_path)
    df_t = _read_excel_safe(topics_path)
    col1, col2 = st.columns(2)
    col1.metric("Paragraphs", len(df_p))
    col2.metric("Topic segments", len(df_t))
    if not df_t.empty:
        st.dataframe(
            df_t[["topic_index", "start", "end", "word_count"]].head(10)
            if "topic_index" in df_t.columns else df_t.head(10),
            use_container_width=True,
        )
    _download_excel_btn(topics_path, "topic_segments.xlsx", f"dl_topics_{vid}")


def _render_step3(cleaned_path: str, vid: int):
    st.markdown("#### Step 3 — LLM Cleaning")
    df = _read_excel_safe(cleaned_path)
    if df.empty:
        st.warning("cleaned_chunks.xlsx not found or empty")
        return
    col1, col2, col3 = st.columns(3)
    col1.metric("Chunks", len(df))
    if "original_words" in df.columns and "cleaned_words" in df.columns:
        total_orig    = int(df["original_words"].sum())
        total_cleaned = int(df["cleaned_words"].sum())
        col2.metric("Original words", f"{total_orig:,}")
        col3.metric("Retention", _retention_badge(total_orig, total_cleaned))
    st.dataframe(
        df[["topic_index", "original_words", "cleaned_words", "cleaned_text"]].head(5)
        if "topic_index" in df.columns else df.head(5),
        use_container_width=True,
    )
    _download_excel_btn(cleaned_path, "cleaned_chunks.xlsx", f"dl_cleaned_{vid}")


def _render_step4_persona(persona_path: str, vid: int):
    st.markdown("#### Step 3b — Persona Extraction")
    text = _read_text_safe(persona_path)
    if not text:
        st.info("Persona extraction skipped or no snippet extracted.")
        return
    st.text_area(
        "Persona snippet",
        value=text,
        height=120,
        disabled=True,
        key=f"persona_{vid}_{len(text)}",
    )


def _render_step4_energy(tagged_path: str, vid: int):
    st.markdown("#### Step 4 — Energy Node Tagging")
    df = _read_excel_safe(tagged_path)
    if df.empty:
        st.warning("cleaned_chunks_tagged.xlsx not found")
        return
    if "energy_node" in df.columns:
        node_counts = df["energy_node"].value_counts().reset_index()
        node_counts.columns = ["energy_node", "count"]
        col1, col2 = st.columns([1, 2])
        col1.metric("Tagged chunks", len(df))
        col2.dataframe(node_counts, use_container_width=True, hide_index=True)
    _download_excel_btn(tagged_path, "cleaned_chunks_tagged.xlsx", f"dl_tagged_{vid}")


def _render_step5_general(ingested_count: str, collection: str, skipped: bool):
    st.markdown(f"#### Step 5 — General Ingest → `{collection}`")
    if skipped:
        st.warning("Qdrant ingest was skipped (skip_ingest=True)")
        return
    if ingested_count:
        col1, col2 = st.columns(2)
        col1.metric("Points ingested", ingested_count)
        col2.metric("Collection", collection)
        st.success(f"✅ `{collection}` populated")
    else:
        st.warning("General ingest count not available")


def _render_step6_density(density_path: str, vid: int):
    st.markdown("#### Step 6 — Content Density Detection")
    report = _read_json_safe(density_path)
    if not report:
        st.warning("density_report.json not found")
        return

    node = report.get("dominant_node", "unknown")
    st.info(f"**Dominant energy node detected:** `{node}`")

    cols = st.columns(5)
    types = ["healing", "activity", "story", "commitment", "pattern"]
    keys  = ["healing_rich", "activity_rich", "story_rich", "commitment_rich", "pattern_rich"]
    for col, label, key in zip(cols, types, keys):
        val = report.get(key, False)
        col.metric(label, "✅ Rich" if val else "⬜ Sparse")

    with st.expander("Full density report JSON", expanded=False):
        st.json(report)


def _render_step7_extractors(out_dir: str, vid: int):
    st.markdown("#### Step 7 — Typed Extractors")

    chunk_types = {
        "healing":    ("extracted_healing.xlsx",    "🔵 Healing Principles"),
        "activities": ("extracted_activities.xlsx", "🟢 Activities"),
        "stories":    ("extracted_stories.xlsx",    "🟠 Stories & Phrases"),
        "commitment": ("extracted_commitment.xlsx", "🟣 Commitment Prompts"),
        "patterns":   ("extracted_patterns.xlsx",   "🔴 Problem Patterns"),
    }

    total_extracted = 0
    summary_rows = []

    for chunk_type, (filename, label) in chunk_types.items():
        path = os.path.join(out_dir, filename)
        df   = _read_excel_safe(path)
        count = len(df)
        total_extracted += count
        summary_rows.append({"Type": label, "Chunks": count, "Status": "✅" if count > 0 else "⬜ skipped"})

    # Summary table
    df_summary = pd.DataFrame(summary_rows)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    st.metric("Total chunks extracted", total_extracted)

    # Per-type expandable preview + download
    for chunk_type, (filename, label) in chunk_types.items():
        path = os.path.join(out_dir, filename)
        df   = _read_excel_safe(path)
        if df.empty:
            continue
        with st.expander(f"{label} — {len(df)} chunks", expanded=False):
            cols_to_show = ["text", "problem_keywords"] if "problem_keywords" in df.columns else ["text"]
            st.dataframe(df[cols_to_show].head(5), use_container_width=True)
            _download_excel_btn(path, filename, f"dl_{chunk_type}_{vid}")


def _render_step8_typed_ingest(ingest_summary: dict, skipped: bool):
    st.markdown("#### Step 8 — Typed Collection Ingest")

    if skipped:
        st.warning("Qdrant ingest was skipped (skip_ingest=True)")
        return

    if not ingest_summary:
        st.warning("ingest_summary.json not found")
        return

    collection_labels = {
        "general":    "souli_chunks_improved",
        "healing":    "souli_healing",
        "activities": "souli_activities",
        "stories":    "souli_stories",
        "commitment": "souli_commitment",
        "patterns":   "souli_patterns",
    }

    rows = []
    for key, collection_name in collection_labels.items():
        count = ingest_summary.get(key, 0)
        rows.append({
            "Collection":      collection_name,
            "Chunks ingested": count,
            "Status":          "✅" if count > 0 else "⬜ 0",
        })

    df_ing = pd.DataFrame(rows)
    st.dataframe(df_ing, use_container_width=True, hide_index=True)

    total = sum(ingest_summary.values())
    st.success(f"✅ **{total} total chunks** across all 6 collections")


# ---------------------------------------------------------------------------
# Full video result card
# ---------------------------------------------------------------------------

def _render_video_results(
    video_name: str,
    url: str,
    out_dir: str,
    outputs: dict,
    general_collection: str,
    skip_ingest: bool,
    vid: int,
):
    with st.expander(f"📹 {video_name} — {url}", expanded=True):

        _render_step1(outputs.get("whisper_segments", ""), vid=vid)
        st.divider()

        _render_step2(
            outputs.get("paragraphs", ""),
            outputs.get("topic_segments", ""),
            vid=vid,
        )
        st.divider()

        _render_step3(outputs.get("cleaned_chunks", ""), vid=vid)
        st.divider()

        _render_step4_persona(outputs.get("persona_snippet", ""), vid=vid)
        _render_step4_energy(outputs.get("cleaned_chunks_tagged", ""), vid=vid)
        st.divider()

        _render_step5_general(
            ingested_count=outputs.get("general_ingested_count", ""),
            collection=general_collection,
            skipped=skip_ingest,
        )
        st.divider()

        _render_step6_density(outputs.get("density_report", ""), vid=vid)
        st.divider()

        _render_step7_extractors(out_dir=out_dir, vid=vid)
        st.divider()

        ingest_summary = _read_json_safe(outputs.get("ingest_summary", ""))
        _render_step8_typed_ingest(ingest_summary=ingest_summary, skipped=skip_ingest)


# ---------------------------------------------------------------------------
# Previous runs viewer
# ---------------------------------------------------------------------------

def _display_previous_runs(general_collection: str):
    st.markdown("---")
    st.markdown("### Previous Multi-Ingestion Runs")

    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        st.info("No previous runs found")
        return

    multi_runs = []
    for run_id in os.listdir(outputs_dir):
        run_path = os.path.join(outputs_dir, run_id)
        multi_path = os.path.join(run_path, "multi_ingestion")
        if os.path.isdir(run_path) and os.path.isdir(multi_path):
            multi_runs.append(run_id)

    if not multi_runs:
        st.info("No multi-ingestion runs found in outputs/ yet.")
        return

    selected_run = st.selectbox(
        "Select a previous run:",
        sorted(multi_runs, reverse=True),
        key="multi_prev_run_select",
    )

    if not selected_run:
        return

    multi_path = os.path.join(outputs_dir, selected_run, "multi_ingestion")
    st.markdown(f"**Run ID:** `{selected_run}`")

    video_dirs = sorted(
        [d for d in os.listdir(multi_path) if os.path.isdir(os.path.join(multi_path, d))],
        reverse=True,
    )

    if not video_dirs:
        st.info("No video folders in this run")
        return

    for vid_idx, vdir in enumerate(video_dirs):
        vpath = os.path.join(multi_path, vdir)
        outputs = {
            "whisper_segments":       os.path.join(vpath, "whisper_segments.xlsx"),
            "paragraphs":             os.path.join(vpath, "paragraphs.xlsx"),
            "topic_segments":         os.path.join(vpath, "topic_segments.xlsx"),
            "cleaned_chunks":         os.path.join(vpath, "cleaned_chunks.xlsx"),
            "cleaned_chunks_tagged":  os.path.join(vpath, "cleaned_chunks_tagged.xlsx"),
            "persona_snippet":        os.path.join(vpath, "persona_snippet.txt"),
            "density_report":         os.path.join(vpath, "density_report.json"),
            "ingest_summary":         os.path.join(vpath, "ingest_summary.json"),
            "general_ingested_count": "",  # not stored separately in outputs dict for prev runs
        }
        _render_video_results(
            video_name=vdir,
            url="(previous run)",
            out_dir=vpath,
            outputs=outputs,
            general_collection=general_collection,
            skip_ingest=False,
            vid=vid_idx,
        )


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def show():
    st.header("🧬 Multi-Collection Ingestion (6 Qdrant Collections)")
    st.markdown("""
    Processes YouTube videos through the **full multi-collection pipeline**:
    Whisper → topic segment → LLM clean → energy tag → **6 Qdrant collections** in one run.

    **Collections populated per video:**
    | Collection | What it stores |
    |---|---|
    | `souli_chunks_improved` | General semantic search (existing) |
    | `souli_healing` | Healing principles from the coach |
    | `souli_activities` | Practices and exercises |
    | `souli_stories` | Stories, metaphors, signature phrases |
    | `souli_commitment` | Readiness challenge questions |
    | `souli_patterns` | How the coach describes each problem pattern |

    > Requires **Ollama** (llama3.1 + qwen2.5:1.5b) and **Qdrant** running locally.
    """)

    # ── Instructions ──────────────────────────────────────────────────────
    with st.expander("📋 CSV Format & Instructions", expanded=False):
        st.markdown("""
### Expected CSV Format

| url | name | title |
|-----|------|-------|
| https://youtu.be/VIDEO_ID_1 | session_01_blocked | Blocked Energy Session 1 |
| https://youtu.be/VIDEO_ID_2 | session_02_depleted | Depleted Energy Session 2 |

**Columns:**
- **url / youtube_url / yt_links** (required): YouTube URL
- **name** (optional): used as `source_label` in Qdrant payloads — helps you track which video a chunk came from
- **title** (optional): display label only
        """)
        st.download_button(
            label="⬇ Download Example CSV",
            data=_create_example_csv(),
            file_name="sample_multi_ingestion.csv",
            mime="text/csv",
        )

    # ── Config ────────────────────────────────────────────────────────────
    st.markdown("### Configuration")
    col_cfg, col_status = st.columns([3, 1])
    with col_cfg:
        config_path = st.text_input(
            "Config file path",
            value="configs/pipeline.yaml",
            key="multi_config_path",
        )
    with col_status:
        st.write("")
        if os.path.exists(config_path):
            st.success("✅ Config found")
        else:
            st.warning("⚠️ Config not found")

    # ── Parameters ────────────────────────────────────────────────────────
    st.markdown("### Pipeline Parameters")

    row1_c1, row1_c2, row1_c3 = st.columns(3)
    with row1_c1:
        whisper_model = st.selectbox(
            "Whisper model",
            ["tiny", "base", "small", "medium", "large"],
            index=3,
            key="multi_whisper",
        )
    with row1_c2:
        similarity_threshold = st.slider(
            "Topic similarity threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.45,
            step=0.05,
            help="Lower = more topic splits. Higher = fewer, larger chunks.",
            key="multi_sim_thresh",
        )
    with row1_c3:
        general_collection = st.text_input(
            "General collection name",
            value="souli_chunks_improved",
            key="multi_gen_col",
        )

    row2_c1, row2_c2, row2_c3, row2_c4 = st.columns(4)
    with row2_c1:
        min_topic_words = st.number_input(
            "Min topic words", value=80, min_value=20, step=10, key="multi_min_w"
        )
    with row2_c2:
        max_topic_words = st.number_input(
            "Max topic words", value=600, min_value=100, step=50, key="multi_max_w"
        )
    with row2_c3:
        skip_persona = st.checkbox(
            "Skip persona extraction",
            value=False,
            help="Faster — skips coach persona update",
            key="multi_skip_persona",
        )
    with row2_c4:
        skip_ingest = st.checkbox(
            "Skip Qdrant ingest",
            value=False,
            help="Produce extraction files only — don't write to Qdrant",
            key="multi_skip_ingest",
        )

    # ── CSV Upload ────────────────────────────────────────────────────────
    st.markdown("### Upload CSV")
    uploaded_file = st.file_uploader(
        "CSV with YouTube links",
        type=["csv"],
        key="multi_csv_upload",
    )

    if uploaded_file is None:
        _display_previous_runs(general_collection)
        return

    df_csv = pd.read_csv(uploaded_file)
    st.markdown("**Preview:**")
    st.dataframe(df_csv, use_container_width=True)

    if not _validate_csv(df_csv):
        st.error("❌ CSV must have a column named `url`, `youtube_url`, or `yt_links`")
        _display_previous_runs(general_collection)
        return

    st.success(f"✅ CSV valid — {len(df_csv)} video(s) found")

    # ── Start button ──────────────────────────────────────────────────────
    if not st.button(
        "🚀 Start Multi-Collection Ingestion",
        type="primary",
        use_container_width=True,
        key="multi_start_btn",
    ):
        _display_previous_runs(general_collection)
        return

    # ── Load config ───────────────────────────────────────────────────────
    if not os.path.exists(config_path):
        st.error(f"Config not found: {config_path}")
        return

    try:
        from souli_pipeline.config_loader import load_config
        cfg = load_config(config_path)
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return

    from souli_pipeline.utils.run_id import get_run_id
    from souli_pipeline.youtube.multi_data_ingestion_improved import run_multi_ingestion_pipeline

    rid   = get_run_id()
    total = len(df_csv)
    successful = 0
    failed     = 0

    progress_bar = st.progress(0)
    status_text  = st.empty()

    for idx, row in df_csv.iterrows():
        url          = _get_url(row)
        source_label = str(row.get("name", f"video_{idx + 1:03d}"))

        if not url:
            st.warning(f"Row {idx + 1}: no URL found — skipping")
            failed += 1
            continue

        status_text.info(f"Processing {idx + 1}/{total}: {source_label} ...")

        out_dir = os.path.join(
            "outputs", rid, "multi_ingestion", f"video_{idx + 1:03d}"
        )

        try:
            outputs = run_multi_ingestion_pipeline(
                cfg=cfg,
                youtube_url=url,
                out_dir=out_dir,
                source_label=source_label,
                whisper_model=whisper_model,
                similarity_threshold=similarity_threshold,
                min_topic_words=int(min_topic_words),
                max_topic_words=int(max_topic_words),
                general_collection=general_collection,
                skip_persona=skip_persona,
                skip_ingest=skip_ingest,
            )

            _render_video_results(
                video_name=source_label,
                url=url,
                out_dir=out_dir,
                outputs=outputs,
                general_collection=general_collection,
                skip_ingest=skip_ingest,
                vid=idx,
            )
            successful += 1

        except Exception as e:
            st.error(f"❌ Failed for {source_label}: {e}")
            failed += 1

        progress_bar.progress((idx + 1) / total)

    # ── Final summary ─────────────────────────────────────────────────────
    status_text.empty()
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total videos", total)
    col2.metric("✅ Successful", successful)
    col3.metric("❌ Failed", failed)

    if successful > 0:
        st.success(
            f"Run complete! Run ID: `{rid}` — "
            f"outputs saved to `outputs/{rid}/multi_ingestion/`"
        )

    _display_previous_runs(general_collection)