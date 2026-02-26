from __future__ import annotations
import os
import pandas as pd
from typing import Optional, Dict, List, Tuple
from ..config import PipelineConfig
from ..utils.logging import setup_logging
from .captions import download_captions, parse_vtt
from .whisper_fallback import whisper_transcribe
from .segments_clean import clean_and_merge_segments
from .chunking import chunk_by_time_and_words, chunk_dedupe_heavy, split_by_words
from .classify import classify
from .scoring import meaning_score, junk_score_generic
from ..llm.factory import make_llm

logger = setup_logging(__name__)

def _explode_chunks(df_chunks: pd.DataFrame, overlap_words: int, max_words: int) -> pd.DataFrame:
    rows = []
    for _, r in df_chunks.iterrows():
        txt = chunk_dedupe_heavy(str(r["text"] or ""))
        if not txt:
            continue
        parts = split_by_words(txt, max_words=max_words, overlap=overlap_words)
        start = float(r["start"]); end = float(r["end"])
        dur = max(0.001, end - start)
        step = dur / len(parts)
        for k, p in enumerate(parts):
            rows.append({
                "start": start + k * step,
                "end": start + (k+1) * step,
                "words": len(p.split()),
                "text": p
            })
    return pd.DataFrame(rows)

def run_youtube_pipeline(
    cfg: PipelineConfig,
    youtube_url: str,
    out_dir: str,
    source_label: str = "",
    tag_energy: bool = True,
) -> Dict[str, str]:
    y = cfg.youtube
    os.makedirs(out_dir, exist_ok=True)

    caption_file = download_captions(youtube_url, langs=y.caption_langs)
    if caption_file:
        logger.info("Captions found: %s", caption_file)
        segments = parse_vtt(caption_file)
    else:
        logger.warning("No captions found. Falling back to whisper.")
        segments = whisper_transcribe(youtube_url, model_name=y.whisper_model)

    df_segments = pd.DataFrame(segments)
    path_segments = os.path.join(out_dir, "segments.xlsx")
    df_segments.to_excel(path_segments, index=False)

    seg_cfg = y.segments
    segments = clean_and_merge_segments(
        segments,
        min_dur=seg_cfg.min_dur,
        min_words=seg_cfg.min_words,
        max_gap=seg_cfg.max_gap,
    )
    logger.info("After clean+merge: %d segments", len(segments))

    chunks = chunk_by_time_and_words(
        segments,
        max_seconds=y.chunking.max_seconds,
        max_words=y.chunking.max_words,
        max_gap=y.chunking.max_gap,
        min_words_to_split=y.chunking.min_words_to_split,
    )
    df_chunks = pd.DataFrame(chunks)
    path_chunks_raw = os.path.join(out_dir, "chunks_raw.xlsx")
    df_chunks.to_excel(path_chunks_raw, index=False)

    df_chunks_clean = _explode_chunks(df_chunks, overlap_words=y.cleaning.overlap_words, max_words=y.chunking.max_words)
    path_chunks_clean = os.path.join(out_dir, "chunks_clean.xlsx")
    df_chunks_clean.to_excel(path_chunks_clean, index=False)

    df_chunks_clean["chunk_type"] = df_chunks_clean["text"].apply(
        lambda t: classify(t, min_words_noise=y.classify.min_words_noise, min_words_teaching=y.classify.min_words_teaching)
    )
    path_keep = os.path.join(out_dir, "chunks_keep.xlsx")
    df_keep = df_chunks_clean[df_chunks_clean["chunk_type"].isin(["problem","teaching"])].copy()
    df_keep.to_excel(path_keep, index=False)

    # scoring
    df_keep["meaning_score"] = df_keep["text"].apply(meaning_score)
    df_keep["junk_score"] = df_keep["text"].apply(junk_score_generic)
    df_ready = df_keep[
        (df_keep["meaning_score"] >= y.scoring.meaning_min_score) &
        (df_keep["junk_score"] <= y.scoring.junk_drop_threshold)
    ].copy()

    # Attach source info
    if source_label:
        df_ready["source_video"] = source_label
    if youtube_url:
        df_ready["youtube_url"] = youtube_url

    # ---------------------------------------------------------------
    # Qwen energy node tagging (runs locally via Ollama)
    # ---------------------------------------------------------------
    if tag_energy and not df_ready.empty:
        try:
            from .energy_tagger import tag_dataframe
            conv = cfg.conversation
            logger.info(
                "Tagging %d chunks with energy nodes via %s ...",
                len(df_ready),
                conv.tagger_model,
            )
            df_ready = tag_dataframe(
                df_ready,
                text_col="text",
                ollama_model=conv.tagger_model,
                ollama_endpoint=conv.ollama_endpoint,
            )
            logger.info("Energy tagging done.")
        except Exception as exc:
            logger.warning("Energy tagging skipped: %s", exc)

    path_ready = os.path.join(out_dir, "teaching_ready.xlsx")
    df_ready.to_excel(path_ready, index=False)

    outputs = {
        "segments": path_segments,
        "chunks_raw": path_chunks_raw,
        "chunks_clean": path_chunks_clean,
        "chunks_keep": path_keep,
        "teaching_ready": path_ready,
    }

    # Optional LLM teaching card extraction
    llm = make_llm(cfg)
    if llm:
        cards = []
        for _, r in df_ready.iterrows():
            card = llm.extract_teaching_card(str(r["text"]))
            card["_start"] = float(r["start"])
            card["_end"] = float(r["end"])
            card["_words"] = int(r["words"])
            if "energy_node" in r.index:
                card["energy_node"] = str(r["energy_node"])
            if "energy_node_reason" in r.index:
                card["energy_node_reason"] = str(r["energy_node_reason"])
            if source_label:
                card["source_video"] = source_label
            if youtube_url:
                card["youtube_url"] = youtube_url
            cards.append(card)
        df_cards = pd.DataFrame(cards)
        path_cards = os.path.join(out_dir, "teaching_cards.xlsx")
        df_cards.to_excel(path_cards, index=False)
        outputs["teaching_cards"] = path_cards

    return outputs
