"""
Data Ingestion (Improved) UI
Runs the improved pipeline: Whisper → Topic Segmentation → LLM Cleaning → Persona → Qdrant
Shows full step-by-step data visibility after each video completes.

Fix: duplicate key bug in _render_step4 — video_idx is now passed through
     so every global_persona text_area gets a unique Streamlit key.
"""
import os
import streamlit as st
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_excel_safe(path: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path)
    except Exception:
        return pd.DataFrame()


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
        "url": ["https://youtu.be/VIDEO_ID_1", "https://youtu.be/VIDEO_ID_2"],
        "name": ["video_1", "video_2"],
        "title": ["My Video 1", "My Video 2"],
    })
    return df.to_csv(index=False)


def _retention_badge(original_words: int, cleaned_words: int) -> str:
    if original_words == 0:
        return "N/A"
    pct = int(cleaned_words / original_words * 100)
    if pct >= 75:
        return f"✅ {pct}%"
    elif pct >= 55:
        return f"⚠️ {pct}%"
    else:
        return f"❌ {pct}%"


def _file_size_kb(path: str) -> float:
    try:
        return os.path.getsize(path) / 1024
    except Exception:
        return 0.0


def _download_btn(path: str, label: str):
    try:
        with open(path, "rb") as f:
            st.download_button(
                label=f"Download {label}",
                data=f.read(),
                file_name=label,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_{path}",
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Step rendering helpers — used both for live results and previous-run viewer
# video_idx is passed everywhere so Streamlit keys stay unique across videos.
# ---------------------------------------------------------------------------

def _render_step1(whisper_path: str, video_idx: int = 0):
    st.markdown("#### Step 1 — Whisper Transcription")
    if not whisper_path or not os.path.exists(whisper_path):
        st.warning("whisper_segments.xlsx not found")
        return
    df = _read_excel_safe(whisper_path)
    if df.empty:
        st.warning("No segments extracted")
        return
    col1, col2 = st.columns(2)
    col1.metric("Segments", len(df))
    if "text" in df.columns:
        total_words = df["text"].dropna().apply(lambda t: len(str(t).split())).sum()
        col2.metric("Total Words", int(total_words))
    cols_to_show = [c for c in ["start", "end", "text"] if c in df.columns]
    st.dataframe(df[cols_to_show], width="stretch", height=250, key=f"whisper_df_{video_idx}")
    _download_btn(whisper_path, "whisper_segments.xlsx")


def _render_step2(paragraphs_path: str, topics_path: str, video_idx: int = 0):
    st.markdown("#### Step 2 — Topic Segmentation")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Paragraphs**")
        if paragraphs_path and os.path.exists(paragraphs_path):
            df_p = _read_excel_safe(paragraphs_path)
            st.metric("Paragraph count", len(df_p))
            cols = [c for c in ["index", "start", "end", "word_count", "text"] if c in df_p.columns]
            st.dataframe(df_p[cols], width="stretch", height=220, key=f"para_df_{video_idx}")
            _download_btn(paragraphs_path, "paragraphs.xlsx")
        else:
            st.warning("paragraphs.xlsx not found")

    with c2:
        st.markdown("**Topic Segments**")
        if topics_path and os.path.exists(topics_path):
            df_t = _read_excel_safe(topics_path)
            st.metric("Topic count", len(df_t))
            cols = [c for c in ["topic_index", "start", "end", "word_count", "text"] if c in df_t.columns]
            st.dataframe(df_t[cols], width="stretch", height=220, key=f"topic_df_{video_idx}")
            _download_btn(topics_path, "topic_segments.xlsx")
        else:
            st.warning("topic_segments.xlsx not found")


def _render_step3(cleaned_path: str, video_idx: int = 0):
    st.markdown("#### Step 3 — LLM Cleaning")
    if not cleaned_path or not os.path.exists(cleaned_path):
        st.warning("cleaned_chunks.xlsx not found")
        return

    df = _read_excel_safe(cleaned_path)
    if df.empty:
        st.warning("No cleaned chunks")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Chunks", len(df))
    if "original_words" in df.columns and "cleaned_words" in df.columns:
        m = df[df["original_words"] > 0]
        if not m.empty:
            avg_ret = int((m["cleaned_words"] / m["original_words"]).mean() * 100)
            c2.metric("Avg Retention", f"{avg_ret}%")
            c3.metric("Total Words", int(df["cleaned_words"].sum()))

    # Side-by-side original vs cleaned for each topic
    orig_col = "original_text" if "original_text" in df.columns else None
    clean_col = "cleaned_text" if "cleaned_text" in df.columns else None

    if orig_col and clean_col:
        for i, row in df.iterrows():
            topic_idx = row.get("topic_index", i)
            orig = str(row.get(orig_col, ""))
            cleaned = str(row.get(clean_col, ""))
            orig_w = int(row.get("original_words", len(orig.split())))
            clean_w = int(row.get("cleaned_words", len(cleaned.split())))
            badge = _retention_badge(orig_w, clean_w)
            with st.expander(
                f"Topic {topic_idx} — retention {badge}  ({orig_w}w → {clean_w}w)",
                expanded=False,
            ):
                left, right = st.columns(2)
                left.markdown("**Original**")
                # KEY FIX: include video_idx so keys are unique across different videos
                left.text_area(
                    "Original",
                    orig,
                    height=180,
                    disabled=True,
                    key=f"orig_{video_idx}_{topic_idx}_{orig_w}",
                )
                right.markdown("**Cleaned**")
                right.text_area(
                    "Cleaned",
                    cleaned,
                    height=180,
                    disabled=True,
                    key=f"clean_{video_idx}_{topic_idx}_{clean_w}",
                )

    st.markdown("**Full table**")
    show_cols = [
        c for c in [
            "topic_index", "start", "end",
            "original_words", "cleaned_words",
            "original_text", "cleaned_text",
        ]
        if c in df.columns
    ]
    st.dataframe(df[show_cols], width="stretch", height=250, key=f"cleaned_df_{video_idx}")
    _download_btn(cleaned_path, "cleaned_chunks.xlsx")


def _render_step4(persona_snippet_path: str, video_idx: int = 0):
    """
    Render Step 4 — Persona extraction.

    video_idx MUST be passed to ensure the global_persona text_area key
    is unique across multiple video cards. Without this, Streamlit raises
    StreamlitDuplicateElementKey when the same data/coach_persona.txt is
    rendered more than once on the same page.
    """
    st.markdown("#### Step 4 — Coach Persona Extraction")

    snippet = _read_text_safe(persona_snippet_path) if persona_snippet_path else ""
    if snippet:
        st.markdown("**Persona snippet from this video:**")
        st.text_area(
            "",
            snippet,
            height=120,
            disabled=True,
            # video_idx + length together guarantee uniqueness
            key=f"snippet_{video_idx}_{len(snippet)}",
        )
    else:
        st.info("No persona snippet file found (may have been skipped)")

    global_persona = _read_text_safe("data/coach_persona.txt")
    if global_persona:
        with st.expander("Global coach persona (data/coach_persona.txt)", expanded=False):
            st.text_area(
                "",
                global_persona,
                height=180,
                disabled=True,
                # BUG FIX: was key=f"global_persona_{len(global_persona)}"
                # which collided when 2+ videos have the same persona file length.
                # Adding video_idx makes it unique per card.
                key=f"global_persona_{video_idx}_{len(global_persona)}",
            )


def _render_step5_energy():
    st.markdown("#### Step 5 — Energy Node Tagging")
    st.info("Coming soon — energy node tagging will be added to the improved pipeline.")


def _render_step6_qdrant(ingested_count: str, collection: str, skipped: bool):
    st.markdown("#### Step 6 — Qdrant Ingest")
    if skipped:
        st.warning("Qdrant ingest was skipped (skip_ingest=True)")
        return
    if ingested_count:
        col1, col2 = st.columns(2)
        col1.metric("Points ingested", ingested_count)
        col2.metric("Collection", collection)
        st.success(f"Successfully ingested into `{collection}`")
    else:
        st.warning("Ingest count not available")


# ---------------------------------------------------------------------------
# Render results for one video (given its output dict from the pipeline)
# video_idx is threaded down to every sub-renderer to guarantee unique keys.
# ---------------------------------------------------------------------------

def _render_video_results(
    video_name: str,
    url: str,
    outputs: dict,
    collection: str,
    skip_ingest: bool,
    video_idx: int = 0,
):
    with st.expander(f"📹 {video_name} — {url}", expanded=True):
        _render_step1(outputs.get("whisper_segments", ""), video_idx=video_idx)
        st.divider()
        _render_step2(
            outputs.get("paragraphs", ""),
            outputs.get("topic_segments", ""),
            video_idx=video_idx,
        )
        st.divider()
        _render_step3(outputs.get("cleaned_chunks", ""), video_idx=video_idx)
        st.divider()
        _render_step4(outputs.get("persona_snippet", ""), video_idx=video_idx)
        st.divider()
        _render_step5_energy()
        st.divider()
        _render_step6_qdrant(
            ingested_count=outputs.get("ingested_count", ""),
            collection=collection,
            skipped=skip_ingest,
        )


# ---------------------------------------------------------------------------
# Previous runs viewer
# ---------------------------------------------------------------------------

def _display_previous_runs(qdrant_collection: str):
    st.markdown("---")
    st.markdown("### Previous Runs (Improved Pipeline)")

    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        st.info("No previous runs found")
        return

    improved_runs = []
    for run_id in os.listdir(outputs_dir):
        run_path = os.path.join(outputs_dir, run_id)
        if os.path.isdir(run_path) and os.path.isdir(
            os.path.join(run_path, "youtube_improved")
        ):
            improved_runs.append(run_id)

    if not improved_runs:
        st.info("No improved pipeline runs found in outputs/")
        return

    selected_run = st.selectbox(
        "Select a previous run:",
        sorted(improved_runs, reverse=True),
        key="prev_run_select",
    )

    if not selected_run:
        return

    yt_improved_path = os.path.join(outputs_dir, selected_run, "youtube_improved")
    if not os.path.exists(yt_improved_path):
        st.warning("No youtube_improved folder found in this run")
        return

    st.markdown(f"**Run ID:** `{selected_run}`")

    video_dirs = sorted(
        [
            d
            for d in os.listdir(yt_improved_path)
            if os.path.isdir(os.path.join(yt_improved_path, d))
        ],
        reverse=True,
    )

    if not video_dirs:
        st.info("No video output folders found")
        return

    # KEY FIX: enumerate gives each video card a unique index so
    # _render_step4 (and others) can produce non-colliding Streamlit keys.
    for video_idx, vdir in enumerate(video_dirs):
        vpath = os.path.join(yt_improved_path, vdir)
        outputs = {
            "whisper_segments": os.path.join(vpath, "whisper_segments.xlsx"),
            "paragraphs":       os.path.join(vpath, "paragraphs.xlsx"),
            "topic_segments":   os.path.join(vpath, "topic_segments.xlsx"),
            "cleaned_chunks":   os.path.join(vpath, "cleaned_chunks.xlsx"),
            "persona_snippet":  os.path.join(vpath, "persona_snippet.txt"),
        }
        _render_video_results(
            video_name=vdir,
            url="(stored run)",
            outputs=outputs,
            collection=qdrant_collection,
            skip_ingest=False,
            video_idx=video_idx,          # ← crucial fix
        )


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def show():
    st.header("🚀 Data Ingestion (Improved Pipeline)")
    st.markdown("""
    Processes YouTube videos using the **improved pipeline**:
    always-Whisper transcription → embedding-based topic segmentation → LLM cleaning → persona extraction → Qdrant ingest.

    > All steps run locally. Requires **Ollama** (llama3.1) and **Qdrant** to be running.
    """)

    # ── Instructions ──────────────────────────────────────────────────────
    with st.expander("CSV Format & Instructions", expanded=False):
        st.markdown("""
### Expected CSV Format

| url | name | title |
|-----|------|-------|
| https://youtu.be/VIDEO_ID_1 | video_1 | My Video 1 |
| https://youtu.be/VIDEO_ID_2 | video_2 | My Video 2 |

**Columns:**
- **url / youtube_url / yt_links** (required): YouTube URL
- **name** (optional): label used for the output folder
- **title** (optional): display title
        """)
        st.download_button(
            label="Download Example CSV",
            data=_create_example_csv(),
            file_name="sample_videos_improved.csv",
            mime="text/csv",
        )

    # ── Config ────────────────────────────────────────────────────────────
    st.markdown("### Configuration")
    col_cfg, col_status = st.columns([3, 1])
    with col_cfg:
        config_path = st.text_input("Config file path", value="configs/pipeline.yaml")
    with col_status:
        st.write("")
        if os.path.exists(config_path):
            st.success("Config found")
        else:
            st.warning("Config not found")

    # ── Pipeline Parameters ───────────────────────────────────────────────
    st.markdown("### Pipeline Parameters")
    row1_c1, row1_c2, row1_c3 = st.columns(3)
    with row1_c1:
        whisper_model = st.selectbox(
            "Whisper model",
            ["tiny", "base", "small", "medium", "large"],
            index=3,
        )
    with row1_c2:
        similarity_threshold = st.slider(
            "Topic similarity threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.45,
            step=0.05,
            help="Lower = more topic splits (more chunks). Higher = fewer, larger chunks.",
        )
    with row1_c3:
        qdrant_collection = st.text_input("Qdrant collection", value="souli_chunks_improved")

    row2_c1, row2_c2, row2_c3, row2_c4 = st.columns(4)
    with row2_c1:
        min_topic_words = st.number_input("Min topic words", value=80, min_value=20, step=10)
    with row2_c2:
        max_topic_words = st.number_input("Max topic words", value=600, min_value=100, step=50)
    with row2_c3:
        skip_persona = st.checkbox(
            "Skip persona extraction",
            value=False,
            help="Faster, skips Step 4",
        )
    with row2_c4:
        skip_ingest = st.checkbox(
            "Skip Qdrant ingest",
            value=False,
            help="Process but don't store in Qdrant",
        )

    # ── Upload ────────────────────────────────────────────────────────────
    st.markdown("### Upload CSV")
    uploaded_file = st.file_uploader("CSV with YouTube links", type=["csv"])

    if uploaded_file is None:
        _display_previous_runs(qdrant_collection)
        return

    df_csv = pd.read_csv(uploaded_file)
    st.markdown("**Preview:**")
    st.dataframe(df_csv, width="stretch")

    if not _validate_csv(df_csv):
        st.error("CSV must have a column named url, youtube_url, or yt_links")
        _display_previous_runs(qdrant_collection)
        return

    st.success(f"CSV valid — {len(df_csv)} video(s) found")

    # ── Process ───────────────────────────────────────────────────────────
    if not st.button("Start Processing", type="primary", use_container_width=True):
        _display_previous_runs(qdrant_collection)
        return

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
    from souli_pipeline.youtube.pipeline_improved import run_improved_pipeline

    rid = get_run_id()
    total = len(df_csv)
    successful = 0
    failed = 0
    all_results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, row in df_csv.iterrows():
        url = _get_url(row)
        name = str(row.get("name", f"video_{idx + 1:03d}"))
        if not url:
            st.warning(f"Row {idx + 1}: no URL, skipping")
            failed += 1
            continue

        status_text.info(f"Processing {idx + 1}/{total}: {name} ...")
        out_dir = os.path.join(
            cfg.run.outputs_dir, rid, "youtube_improved", name
        )

        try:
            outputs = run_improved_pipeline(
                cfg=cfg,
                youtube_url=url,
                out_dir=out_dir,
                source_label=name,
                whisper_model=whisper_model,
                similarity_threshold=float(similarity_threshold),
                min_topic_words=int(min_topic_words),
                max_topic_words=int(max_topic_words),
                qdrant_collection=qdrant_collection,
                skip_persona=skip_persona,
                skip_ingest=skip_ingest,
            )
            successful += 1
            all_results.append((name, url, outputs, True, None))
        except Exception as e:
            failed += 1
            all_results.append((name, url, {}, False, str(e)))

        progress_bar.progress((idx + 1) / total)

    status_text.empty()

    # ── Summary metrics ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Run Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Videos", total)
    m2.metric("Successful", successful)
    m3.metric("Failed", failed)
    m4.metric("Run ID", rid)

    if successful:
        st.success(
            f"Processing complete. Outputs in `outputs/{rid}/youtube_improved/`"
        )

    # ── Per-video results — enumerate gives unique video_idx ──────────────
    st.markdown("### Results by Video (latest first)")
    for video_idx, (name, url, outputs, ok, err) in enumerate(reversed(all_results)):
        if not ok:
            with st.expander(f"❌ {name} — {url}", expanded=False):
                st.error(f"Pipeline error: {err}")
            continue
        _render_video_results(
            name, url, outputs, qdrant_collection, skip_ingest,
            video_idx=video_idx,
        )

    _display_previous_runs(qdrant_collection)