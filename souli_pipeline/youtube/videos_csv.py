"""Load video list from CSV for batch processing."""
from __future__ import annotations
import pandas as pd
from typing import List, Dict, Any

# Accepted column names for URL (first found wins)
URL_COLUMNS = ["youtube_url", "url", "video_url", "link"]
# Optional metadata columns to pass through
META_COLUMNS = ["video_id", "name", "title", "id"]


def load_videos_csv(path: str) -> List[Dict[str, Any]]:
    """
    Load CSV with at least one URL column. Returns list of dicts with keys:
    - url: str (required)
    - video_index: int (1-based)
    - source_label: str (for merge: url or name/title if present)
    - any META_COLUMNS present in CSV
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    url_col = None
    for c in URL_COLUMNS:
        if c in df.columns:
            url_col = c
            break
    if not url_col:
        raise ValueError(
            f"CSV must have one of these columns: {URL_COLUMNS}. Found: {list(df.columns)}"
        )

    out: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        url = str(row.get(url_col, "")).strip()
        if not url or url.lower() in ("nan", ""):
            continue
        label = url
        for meta in ["name", "title", "video_id"]:
            if meta in df.columns and pd.notna(row.get(meta)):
                v = str(row[meta]).strip()
                if v:
                    label = v
                    break
        rec: Dict[str, Any] = {
            "url": url,
            "video_index": len(out) + 1,
            "source_label": label,
        }
        for meta in META_COLUMNS:
            if meta in df.columns and pd.notna(row.get(meta)):
                rec[meta] = row[meta]
        out.append(rec)
    return out
