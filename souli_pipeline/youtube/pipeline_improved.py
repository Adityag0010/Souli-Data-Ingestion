"""
souli_pipeline/youtube/pipeline_improved.py

Improved YouTube processing pipeline.
Completely separate from pipeline.py — zero shared state.

Flow:
    URL → Whisper → Paragraph grouping → Topic segmentation (embeddings)
        → LLM segment cleaning → Persona extraction → Energy tagging → Qdrant ingest

Outputs per video (in out_dir):
    whisper_segments.xlsx       — raw Whisper output
    paragraphs.xlsx             — grouped paragraphs
    topic_segments.xlsx         — detected topic boundaries
    cleaned_chunks.xlsx         — LLM-cleaned prose per topic
    cleaned_chunks_tagged.xlsx  — same + energy_node column (this goes to Qdrant)

Global (updated after each video):
    data/coach_persona.txt   — evolving coach persona string
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import pandas as pd

from souli_pipeline.config import PipelineConfig
from souli_pipeline.utils.logging import setup_logging

logger = setup_logging(__name__)


def run_improved_pipeline(
    cfg: PipelineConfig,
    youtube_url: str,
    out_dir: str,
    source_label: str = "",
    # Improved pipeline specific params (come from improved_pipeline config block)
    whisper_model: str = "medium",
    similarity_threshold: float = 0.45,
    min_topic_words: int = 80,
    max_topic_words: int = 600,
    num_ctx_processing: int = 8192,
    qdrant_collection: str = "souli_chunks_improved",
    persona_path: str = "data/coach_persona.txt",
    skip_persona: bool = False,
    skip_ingest: bool = False,
) -> Dict[str, str]:
    """
    Run the improved pipeline for a single YouTube video.

    Returns dict of output file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    c = cfg.conversation
    r = cfg.retrieval

    outputs: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Step 1 — Whisper transcription
    # ------------------------------------------------------------------
    logger.info("[IMPROVED] Step 1/6 — Whisper transcription: %s", youtube_url)
    from souli_pipeline.youtube.whisper_transcribe import transcribe_url

    segments = transcribe_url(
        url=youtube_url,
        out_dir=out_dir,
        whisper_model=whisper_model,
        language="en",
    )

    if not segments:
        logger.warning("[IMPROVED] No segments from Whisper — aborting for %s", youtube_url)
        return outputs

    df_segments = pd.DataFrame(segments)
    path_segments = os.path.join(out_dir, "whisper_segments.xlsx")
    df_segments.to_excel(path_segments, index=False)
    outputs["whisper_segments"] = path_segments
    logger.info("[IMPROVED] Whisper segments saved: %d rows", len(df_segments))

    # ------------------------------------------------------------------
    # Step 2 — Topic segmentation (paragraph grouping + embedding boundary detection)
    # ------------------------------------------------------------------
    logger.info("[IMPROVED] Step 2/6 — Topic segmentation...")
    from souli_pipeline.youtube.topic_segmenter import detect_topics

    paragraphs, topic_segments = detect_topics(
        segments=segments,
        similarity_threshold=similarity_threshold,
        min_topic_words=min_topic_words,
        max_topic_words=max_topic_words,
        embedding_model=r.embedding_model or "sentence-transformers/all-MiniLM-L6-v2",
    )

    # Save paragraphs (for inspection / threshold tuning)
    df_paragraphs = pd.DataFrame([
        {
            "index":      p.index,
            "start":      p.start,
            "end":        p.end,
            "text":       p.text,
            "word_count": p.word_count,
        }
        for p in paragraphs
    ])
    path_paragraphs = os.path.join(out_dir, "paragraphs.xlsx")
    df_paragraphs.to_excel(path_paragraphs, index=False)
    outputs["paragraphs"] = path_paragraphs

    # Save topic segments (for inspection)
    df_topics = pd.DataFrame([
        {
            "topic_index":       t.topic_index,
            "start":             t.start,
            "end":               t.end,
            "word_count":        t.word_count,
            "paragraph_indices": str(t.paragraph_indices),
            "text":              t.text,
        }
        for t in topic_segments
    ])
    path_topics = os.path.join(out_dir, "topic_segments.xlsx")
    df_topics.to_excel(path_topics, index=False)
    outputs["topic_segments"] = path_topics

    logger.info(
        "[IMPROVED] %d topic segments detected from %d paragraphs",
        len(topic_segments), len(paragraphs),
    )

    if not topic_segments:
        logger.warning("[IMPROVED] No topic segments detected — aborting.")
        return outputs

    # ------------------------------------------------------------------
    # Step 3 — LLM segment cleaning
    # ------------------------------------------------------------------
    logger.info("[IMPROVED] Step 3/6 — LLM segment cleaning (%d segments)...", len(topic_segments))
    from souli_pipeline.youtube.segment_cleaner import clean_all_segments

    cleaned_results = clean_all_segments(
        topic_segments=topic_segments,
        ollama_model=c.chat_model,
        ollama_endpoint=c.ollama_endpoint,
        num_ctx=num_ctx_processing,
        temperature=0.3,
    )

    # Attach source info
    for r_item in cleaned_results:
        if source_label:
            r_item["source_video"] = source_label
        r_item["youtube_url"] = youtube_url

    df_cleaned = pd.DataFrame(cleaned_results)
    path_cleaned = os.path.join(out_dir, "cleaned_chunks.xlsx")
    df_cleaned.to_excel(path_cleaned, index=False)
    outputs["cleaned_chunks"] = path_cleaned
    logger.info("[IMPROVED] Cleaned chunks saved: %d rows", len(df_cleaned))

    # ------------------------------------------------------------------
    # Step 4 — Persona extraction + update
    # ------------------------------------------------------------------
    if not skip_persona:
        logger.info("[IMPROVED] Step 4/6 — Persona extraction...")
        from souli_pipeline.youtube.persona_extractor import (
            extract_from_video,
            update_persona_file,
        )

        # Join all cleaned segments into one transcript for persona extraction
        full_cleaned = " ".join(
            r_item.get("cleaned_text", "") for r_item in cleaned_results
        ).strip()

        snippet = extract_from_video(
            cleaned_transcript=full_cleaned,
            ollama_model=c.chat_model,
            ollama_endpoint=c.ollama_endpoint,
            num_ctx=num_ctx_processing,
        )

        if snippet:
            snippet_path = os.path.join(out_dir, "persona_snippet.txt")
            with open(snippet_path, "w", encoding="utf-8") as f:
                f.write(snippet)
            outputs["persona_snippet"] = snippet_path

            # Merge into the global persona file
            updated_persona = update_persona_file(
                persona_path=persona_path,
                new_snippet=snippet,
                ollama_model=c.chat_model,
                ollama_endpoint=c.ollama_endpoint,
                num_ctx=num_ctx_processing,
            )
            outputs["coach_persona"] = persona_path
            logger.info("[IMPROVED] Persona updated at %s", persona_path)
        else:
            logger.warning("[IMPROVED] No persona snippet extracted for this video.")
    else:
        logger.info("[IMPROVED] Persona extraction skipped.")

    # ------------------------------------------------------------------
    # Step 5 — Energy node tagging (Qwen via Ollama)
    # ------------------------------------------------------------------
    if not df_cleaned.empty:
        logger.info(
            "[IMPROVED] Step 5/6 — Energy node tagging (%d chunks) via %s ...",
            len(df_cleaned),
            c.tagger_model,
        )
        try:
            from souli_pipeline.youtube.energy_tagger import tag_dataframe

            df_cleaned = tag_dataframe(
                df_cleaned,
                text_col="cleaned_text",      # improved pipeline stores text in cleaned_text column
                ollama_model=c.tagger_model,
                ollama_endpoint=c.ollama_endpoint,
            )

            # Save the tagged version so you can inspect it in the UI
            tagged_path = os.path.join(out_dir, "cleaned_chunks_tagged.xlsx")
            df_cleaned.to_excel(tagged_path, index=False)
            outputs["cleaned_chunks_tagged"] = tagged_path

            # Count how many got a real tag vs keyword fallback
            tagged_count = df_cleaned["energy_node"].notna().sum()
            fallback_count = (df_cleaned.get("energy_node_reason", pd.Series()) == "keyword_fallback").sum()
            logger.info(
                "[IMPROVED] Energy tagging done — %d tagged, %d keyword fallbacks.",
                tagged_count,
                fallback_count,
            )

        except Exception as exc:
            logger.warning(
                "[IMPROVED] Energy tagging failed: %s — chunks will have empty energy_node. "
                "Check that Ollama is running at %s and model %s is available.",
                exc,
                c.ollama_endpoint,
                c.tagger_model,
            )
    else:
        logger.info("[IMPROVED] Skipping energy tagging — df_cleaned is empty.")

    # ------------------------------------------------------------------
    # Step 6 — Ingest to souli_chunks_improved
    # ------------------------------------------------------------------
    if not skip_ingest and not df_cleaned.empty:
        logger.info(
            "[IMPROVED] Step 6/6 — Ingesting %d chunks to Qdrant collection '%s'...",
            len(df_cleaned),
            qdrant_collection,
        )
        from souli_pipeline.retrieval.qdrant_store_improved import ingest_improved_chunks

        n = ingest_improved_chunks(
            df=df_cleaned,
            collection=qdrant_collection,
            embedding_model=r.embedding_model or "sentence-transformers/all-MiniLM-L6-v2",
            host=r.qdrant_host,
            port=r.qdrant_port,
        )
        logger.info("[IMPROVED] Ingested %d chunks.", n)
        outputs["ingested_count"] = str(n)
    else:
        logger.info("[IMPROVED] Qdrant ingest skipped.")

    logger.info("[IMPROVED] Pipeline complete. Outputs: %s", list(outputs.keys()))
    return outputs