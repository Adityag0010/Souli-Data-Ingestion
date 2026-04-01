"""
Gold Data Viewer — browse gold.xlsx (standard pipeline) and cleaned_chunks.xlsx
(improved pipeline) across all runs. Latest runs first.
"""
import os
import glob
import datetime
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_run_ts(run_id: str) -> datetime.datetime:
    """Parse timestamp from run_id like 20240512_143022_abc123."""
    try:
        return datetime.datetime.strptime(run_id[:15], "%Y%m%d_%H%M%S")
    except Exception:
        return datetime.datetime.min


def _human_ts(run_id: str) -> str:
    dt = _parse_run_ts(run_id)
    if dt == datetime.datetime.min:
        return run_id
    return dt.strftime("%d %b %Y  %H:%M:%S")


def _file_size_kb(path: str) -> float:
    try:
        return os.path.getsize(path) / 1024
    except Exception:
        return 0.0


def _read_excel(path: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(path)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception as e:
        return pd.DataFrame()


def _scan_runs(outputs_dir: str = "outputs"):
    """
    Scan outputs/ and return sorted list of run dicts (latest first).
    Each dict has:
        run_id, ts_human, gold_path, reject_path, cleaned_paths (list), type
    """
    if not os.path.exists(outputs_dir):
        return []

    runs = []
    for run_id in os.listdir(outputs_dir):
        run_path = os.path.join(outputs_dir, run_id)
        if not os.path.isdir(run_path):
            continue

        entry = {
            "run_id":        run_id,
            "ts":            _parse_run_ts(run_id),
            "ts_human":      _human_ts(run_id),
            "run_path":      run_path,
            "gold_path":     None,
            "reject_path":   None,
            "cleaned_paths": [],   # improved pipeline
            "has_standard":  False,
            "has_improved":  False,
        }

        # Standard pipeline — energy/gold.xlsx
        gold_path = os.path.join(run_path, "energy", "gold.xlsx")
        reject_path = os.path.join(run_path, "energy", "reject.xlsx")
        if os.path.exists(gold_path):
            entry["gold_path"]    = gold_path
            entry["reject_path"]  = reject_path if os.path.exists(reject_path) else None
            entry["has_standard"] = True

        # Improved pipeline — youtube_improved/**/cleaned_chunks.xlsx
        cleaned = glob.glob(
            os.path.join(run_path, "youtube_improved", "**", "cleaned_chunks.xlsx"),
            recursive=True,
        )
        if cleaned:
            entry["cleaned_paths"] = sorted(cleaned)
            entry["has_improved"]  = True

        if entry["has_standard"] or entry["has_improved"]:
            runs.append(entry)

    runs.sort(key=lambda r: r["ts"], reverse=True)
    return runs


# ---------------------------------------------------------------------------
# Node colour helpers
# ---------------------------------------------------------------------------

_NODE_COLORS = {
    "blocked_energy":       "#e74c3c",
    "depleted_energy":      "#e67e22",
    "scattered_energy":     "#f1c40f",
    "outofcontrol_energy":  "#9b59b6",
    "normal_energy":        "#27ae60",
}

_NODE_LABELS = {
    "blocked_energy":       "Blocked",
    "depleted_energy":      "Depleted",
    "scattered_energy":     "Scattered",
    "outofcontrol_energy":  "Out of Control",
    "normal_energy":        "Normal / Growth",
}

_NODE_COL = "energy_node/energy block behind it/ inner block"


def _node_pill(node: str) -> str:
    color = _NODE_COLORS.get(node, "#555")
    label = _NODE_LABELS.get(node, node.replace("_", " ").title())
    return (
        f'<span style="background:{color}22;color:{color};'
        f'border:1px solid {color}66;padding:1px 8px;border-radius:20px;'
        f'font-size:0.72rem;font-weight:600;white-space:nowrap;">{label}</span>'
    )


def _node_distribution(df: pd.DataFrame, node_col: str) -> dict:
    if node_col not in df.columns:
        return {}
    return df[node_col].value_counts().to_dict()


# ---------------------------------------------------------------------------
# Standard pipeline gold renderer
# ---------------------------------------------------------------------------

def _render_standard_gold(gold_path: str, reject_path: str | None):
    df = _read_excel(gold_path)

    # Detect node column
    node_col = _NODE_COL if _NODE_COL in df.columns else None
    if node_col is None:
        for c in df.columns:
            if "energy" in c.lower() and "node" in c.lower():
                node_col = c
                break

    # ── Top metrics ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Gold rows", len(df))
    if reject_path and os.path.exists(reject_path):
        df_rej = _read_excel(reject_path)
        total = len(df) + len(df_rej)
        m2.metric("Rejected rows", len(df_rej))
        m3.metric("Pass rate", f"{len(df)/max(1,total)*100:.0f}%")
    else:
        m2.metric("Rejected", "—")
        m3.metric("Pass rate", "—")
    m4.metric("Columns", len(df.columns))

    # ── Node distribution ─────────────────────────────────────────────────
    if node_col:
        dist = _node_distribution(df, node_col)
        if dist:
            st.markdown("**Energy node distribution**")
            cols = st.columns(len(dist))
            for i, (node, count) in enumerate(
                sorted(dist.items(), key=lambda x: -x[1])
            ):
                color = _NODE_COLORS.get(node, "#888")
                label = _NODE_LABELS.get(node, node.replace("_","").title())
                with cols[i]:
                    st.markdown(
                        f'<div style="text-align:center;background:{color}18;'
                        f'border:1px solid {color}44;border-radius:8px;padding:10px 4px;">'
                        f'<div style="color:{color};font-size:1.5rem;font-weight:700;">{count}</div>'
                        f'<div style="color:{color};font-size:0.68rem;font-weight:600;">{label}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    st.divider()

    # ── Filter / search ───────────────────────────────────────────────────
    filter_col1, filter_col2 = st.columns([2, 3])
    with filter_col1:
        node_opts = ["All nodes"]
        if node_col and dist:
            node_opts += [
                f"{_NODE_LABELS.get(n, n)} ({c})"
                for n, c in sorted(dist.items(), key=lambda x: -x[1])
            ]
        selected_node_label = st.selectbox("Filter by node", node_opts, key="gold_node_filter")

    with filter_col2:
        search = st.text_input("🔍 Search in Problem Statement", key="gold_search",
                               placeholder="Type to filter rows…")

    # Apply filters
    filtered = df.copy()
    if selected_node_label != "All nodes" and node_col:
        # Extract raw node key from label
        for node, label in _NODE_LABELS.items():
            if label in selected_node_label:
                filtered = filtered[filtered[node_col].astype(str).str.lower() == node]
                break
    if search.strip():
        prob_col = "Problem statement"
        if prob_col not in filtered.columns:
            prob_col = next((c for c in filtered.columns if "problem" in c.lower()), None)
        if prob_col:
            filtered = filtered[
                filtered[prob_col].astype(str).str.contains(search, case=False, na=False)
            ]

    st.caption(f"Showing {len(filtered)} of {len(df)} rows")

    # ── Column selector ───────────────────────────────────────────────────
    priority_cols = [
        "Problem statement", "Aspects of Woman Track",
        _NODE_COL, "Duality Check",
        "deeper_blocks/ pshychlogical issues",
        "primary_healing_principles",
        "primary_practices ( 7 min quick relief)",
    ]
    available_priority = [c for c in priority_cols if c in filtered.columns]
    other_cols = [c for c in filtered.columns if c not in priority_cols and not c.startswith("_")]
    all_displayable = available_priority + other_cols

    with st.expander("Choose columns to display", expanded=False):
        selected_cols = st.multiselect(
            "Columns",
            all_displayable,
            default=available_priority[:5],
            key="gold_cols",
        )

    if not selected_cols:
        selected_cols = available_priority[:4] if available_priority else filtered.columns.tolist()[:4]

    # Inject node badge column for display
    display_df = filtered[selected_cols].copy()
    if node_col and node_col in selected_cols:
        display_df[node_col] = display_df[node_col].apply(
            lambda v: _NODE_LABELS.get(str(v).strip().lower(), str(v))
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        height=min(600, 80 + len(filtered) * 36),
    )

    # ── Row detail viewer ─────────────────────────────────────────────────
    st.markdown("**Row detail viewer**")
    if len(filtered) > 0:
        row_idx = st.number_input(
            "Row index (0-based)",
            min_value=0,
            max_value=len(filtered) - 1,
            value=0,
            key="gold_row_idx",
        )
        row = filtered.iloc[int(row_idx)]
        for col in filtered.columns:
            if col.startswith("_"):
                continue
            val = str(row.get(col, ""))
            if not val or val in ("nan", "None", ""):
                continue
            if col == node_col:
                st.markdown(
                    f"**{col}:** " + _node_pill(str(row[col]).strip().lower()),
                    unsafe_allow_html=True,
                )
            else:
                with st.expander(f"**{col}**", expanded=False):
                    st.write(val)

    # ── Download ──────────────────────────────────────────────────────────
    with open(gold_path, "rb") as f:
        st.download_button(
            "⬇️ Download gold.xlsx",
            data=f.read(),
            file_name="gold.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    if reject_path and os.path.exists(reject_path):
        with open(reject_path, "rb") as f:
            st.download_button(
                "⬇️ Download reject.xlsx",
                data=f.read(),
                file_name="reject.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


# ---------------------------------------------------------------------------
# Improved pipeline cleaned_chunks renderer
# ---------------------------------------------------------------------------

def _render_improved_gold(cleaned_paths: list):
    st.markdown(
        """
        <div style="background:#0e2230;border-left:3px solid #38bdf8;border-radius:6px;
        padding:10px 14px;font-size:0.82rem;color:#b8d0e8;margin-bottom:12px;">
        <b style="color:#38bdf8;">ℹ️ Improved pipeline note:</b>  
        The improved pipeline does not produce a separate <code>gold.xlsx</code>.
        Its quality gate is applied earlier — only LLM-cleaned, content-passing chunks
        reach <code>cleaned_chunks.xlsx</code>.  
        Think of <b>cleaned_chunks.xlsx</b> as the gold equivalent.
        </div>
        """,
        unsafe_allow_html=True,
    )

    for path in cleaned_paths:
        video_name = os.path.basename(os.path.dirname(path))
        df = _read_excel(path)

        with st.expander(
            f"📹 {video_name}  —  {len(df)} chunks  ({_file_size_kb(path):.1f} KB)",
            expanded=True,
        ):
            if df.empty:
                st.warning("No data found in cleaned_chunks.xlsx")
                continue

            # Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Chunks", len(df))
            if "original_words" in df.columns and "cleaned_words" in df.columns:
                m = df[df["original_words"] > 0]
                if not m.empty:
                    avg_ret = int((m["cleaned_words"] / m["original_words"]).mean() * 100)
                    c2.metric("Avg retention", f"{avg_ret}%")
                    total_w = int(df["cleaned_words"].sum())
                    c3.metric("Total cleaned words", total_w)

            # Energy node distribution (if present)
            if "energy_node" in df.columns:
                dist = df["energy_node"].value_counts().to_dict()
                if dist:
                    st.markdown("**Energy nodes**")
                    node_cols = st.columns(len(dist))
                    for i, (node, count) in enumerate(sorted(dist.items(), key=lambda x: -x[1])):
                        color = _NODE_COLORS.get(node, "#888")
                        label = _NODE_LABELS.get(node, node)
                        with node_cols[i]:
                            st.markdown(
                                f'<div style="text-align:center;background:{color}18;'
                                f'border:1px solid {color}44;border-radius:8px;padding:8px 4px;">'
                                f'<div style="color:{color};font-size:1.3rem;font-weight:700;">{count}</div>'
                                f'<div style="color:{color};font-size:0.66rem;">{label}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

            st.divider()

            # Filter
            search_key = f"improved_search_{video_name}"
            search = st.text_input("🔍 Search in cleaned text", key=search_key,
                                   placeholder="Type to filter…")
            disp = df.copy()
            if search.strip() and "cleaned_text" in disp.columns:
                disp = disp[disp["cleaned_text"].str.contains(search, case=False, na=False)]
            st.caption(f"Showing {len(disp)} chunks")

            # Table
            show_cols = [
                c for c in ["topic_index", "start", "end",
                             "original_words", "cleaned_words",
                             "energy_node", "cleaned_text"]
                if c in disp.columns
            ]
            st.dataframe(disp[show_cols], use_container_width=True, height=300)

            # Side-by-side detail
            if "cleaned_text" in df.columns or "original_text" in df.columns:
                st.markdown("**Chunk detail**")
                row_i_key = f"improved_row_{video_name}"
                row_i = st.number_input("Chunk index", min_value=0,
                                        max_value=max(0, len(disp) - 1),
                                        value=0, key=row_i_key)
                if len(disp) > 0:
                    row = disp.iloc[int(row_i)]
                    left, right = st.columns(2)
                    left.markdown("**Original**")
                    left.text_area("", str(row.get("original_text", "")), height=180,
                                   disabled=True, key=f"oi_{video_name}_{row_i}")
                    right.markdown("**Cleaned**")
                    right.text_area("", str(row.get("cleaned_text", "")), height=180,
                                    disabled=True, key=f"ci_{video_name}_{row_i}")

            with open(path, "rb") as f:
                st.download_button(
                    f"⬇️ Download cleaned_chunks.xlsx ({video_name})",
                    data=f.read(),
                    file_name=f"cleaned_chunks_{video_name}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_{path}",
                )


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def show():
    st.header("🥇 Gold Data Viewer")
    st.markdown(
        "Browse quality-filtered data from all pipeline runs — **latest first**. "
        "Standard pipeline shows `gold.xlsx`; improved pipeline shows `cleaned_chunks.xlsx` "
        "(the equivalent quality gate)."
    )

    outputs_dir = "outputs"
    runs = _scan_runs(outputs_dir)

    if not runs:
        st.info(
            "No runs found yet. Run the **Data Ingestion** or **CLI pipeline** first, "
            f"then outputs will appear in `{outputs_dir}/`."
        )
        return

    # ── Run selector at the top ───────────────────────────────────────────
    st.markdown(f"**{len(runs)} run(s) found** — most recent first")

    for idx, run in enumerate(runs):
        run_id     = run["run_id"]
        ts_human   = run["ts_human"]
        has_std    = run["has_standard"]
        has_imp    = run["has_improved"]

        # Badge row
        badges = []
        if has_std:
            gold_size = _file_size_kb(run["gold_path"])
            df_g = _read_excel(run["gold_path"])
            badges.append(
                f'<span style="background:#0e2a1a;color:#4ade80;border:1px solid #4ade8044;'
                f'padding:2px 10px;border-radius:20px;font-size:0.72rem;font-weight:600;">'
                f'📊 Standard  ·  {len(df_g)} gold rows  ·  {gold_size:.1f} KB</span>'
            )
        if has_imp:
            n_chunks = sum(
                len(_read_excel(p)) for p in run["cleaned_paths"]
            )
            badges.append(
                f'<span style="background:#0e1a2e;color:#38bdf8;border:1px solid #38bdf844;'
                f'padding:2px 10px;border-radius:20px;font-size:0.72rem;font-weight:600;">'
                f'🚀 Improved  ·  {n_chunks} chunks  ·  {len(run["cleaned_paths"])} video(s)</span>'
            )

        badge_html = "  ".join(badges)
        is_latest  = idx == 0

        header_label = (
            f"{'🆕 ' if is_latest else ''}Run: {ts_human}  ·  ID: {run_id}"
        )

        with st.expander(header_label, expanded=is_latest):
            st.markdown(badge_html, unsafe_allow_html=True)
            st.markdown("")

            # Tabs per run: Standard | Improved (only if both exist)
            if has_std and has_imp:
                tab_std, tab_imp = st.tabs(["📊 Standard Pipeline (gold.xlsx)", "🚀 Improved Pipeline (cleaned_chunks)"])
                with tab_std:
                    _render_standard_gold(run["gold_path"], run["reject_path"])
                with tab_imp:
                    _render_improved_gold(run["cleaned_paths"])
            elif has_std:
                _render_standard_gold(run["gold_path"], run["reject_path"])
            else:
                _render_improved_gold(run["cleaned_paths"])