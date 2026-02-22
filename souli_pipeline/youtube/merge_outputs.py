"""Merge per-video teaching_ready and teaching_cards into single files with source_video column."""
from __future__ import annotations
import os
import pandas as pd
from typing import List, Dict, Any

def merge_teaching_outputs(
    video_results: List[Dict[str, Any]],
    out_dir: str,
    teaching_ready_filename: str = "teaching_ready.xlsx",
    teaching_cards_filename: str = "teaching_cards.xlsx",
) -> Dict[str, str]:
    """
    video_results: list of dicts from run_youtube_pipeline, each must have
      "out_dir" (folder for that video) and "source_label" (e.g. URL or name).
    Writes merged_teaching_ready.xlsx and merged_teaching_cards.xlsx in out_dir.
    Returns paths to merged files (only for files that were written).
    """
    os.makedirs(out_dir, exist_ok=True)
    merged_paths: Dict[str, str] = {}

    # Merge teaching_ready
    ready_dfs: List[pd.DataFrame] = []
    for v in video_results:
        out_dir_v = v.get("out_dir", "")
        label = v.get("source_label", out_dir_v)
        path_ready = os.path.join(out_dir_v, teaching_ready_filename)
        if os.path.isfile(path_ready):
            df = pd.read_excel(path_ready)
            df.insert(0, "source_video", label)
            ready_dfs.append(df)
    if ready_dfs:
        merged_ready = pd.concat(ready_dfs, ignore_index=True)
        path_merged_ready = os.path.join(out_dir, "merged_teaching_ready.xlsx")
        merged_ready.to_excel(path_merged_ready, index=False)
        merged_paths["merged_teaching_ready"] = path_merged_ready

    # Merge teaching_cards (if LLM was used)
    card_dfs: List[pd.DataFrame] = []
    for v in video_results:
        out_dir_v = v.get("out_dir", "")
        label = v.get("source_label", out_dir_v)
        path_cards = os.path.join(out_dir_v, teaching_cards_filename)
        if os.path.isfile(path_cards):
            df = pd.read_excel(path_cards)
            df.insert(0, "source_video", label)
            card_dfs.append(df)
    if card_dfs:
        merged_cards = pd.concat(card_dfs, ignore_index=True)
        path_merged_cards = os.path.join(out_dir, "merged_teaching_cards.xlsx")
        merged_cards.to_excel(path_merged_cards, index=False)
        merged_paths["merged_teaching_cards"] = path_merged_cards

    return merged_paths
