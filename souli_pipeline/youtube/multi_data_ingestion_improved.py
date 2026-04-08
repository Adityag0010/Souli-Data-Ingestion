"""
souli_pipeline/youtube/multi_data_ingestion_improved.py

Multi-collection ingestion pipeline — runs everything in one click.

Builds on pipeline_improved.py's core steps (Whisper → segment → clean)
and adds a parallel extraction layer that populates 6 Qdrant collections:

  souli_chunks_improved  — general semantic search (existing, reused)
  souli_healing          — healing principles
  souli_activities       — practices and exercises
  souli_stories          — metaphors, stories, signature phrases
  souli_commitment       — readiness challenge questions
  souli_patterns         — problem pattern descriptions

Flow:
  URL
  → Step 1: Whisper transcription          (existing: whisper_transcribe)
  → Step 2: Topic segment + clean          (existing: topic_segmenter + segment_cleaner)
  → Step 3: Persona extraction             (existing: persona_extractor)
  → Step 4: Energy node tagging            (existing: energy_tagger)
  → Step 5: General ingest                 (existing: qdrant_store_improved.ingest_improved_chunks)
  → Step 6: Content density detection      (NEW: content_density_detector)
  → Step 7: Multi-extractor run            (NEW: multi_extractors.run_extractors_from_density)
  → Step 8: Typed collection ingest        (NEW: qdrant_store_multi.ingest_all_extractor_outputs)

Outputs per video (in out_dir):
  whisper_segments.xlsx          — raw Whisper output
  paragraphs.xlsx                — grouped paragraphs
  topic_segments.xlsx            — topic boundaries
  cleaned_chunks.xlsx            — LLM-cleaned prose per topic
  cleaned_chunks_tagged.xlsx     — with energy_node column
  density_report.json            — content density detection result
  extracted_healing.xlsx         — healing principle chunks
  extracted_activities.xlsx      — activity chunks
  extracted_stories.xlsx         — story/phrase chunks
  extracted_commitment.xlsx      — commitment prompt chunks
  extracted_patterns.xlsx        — problem pattern chunks
  ingest_summary.json            — counts per collection

ZERO changes to pipeline_improved.py or any existing file.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import pandas as pd

from souli_pipeline.config import PipelineConfig
from souli_pipeline.utils.logging import setup_logging

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_multi_ingestion_pipeline(
    cfg: PipelineConfig,
    youtube_url: str,
    out_dir: str,
    source_label: str = "",
    whisper_model: str = "medium",
    similarity_threshold: float = 0.45,
    min_topic_words: int = 80,
    max_topic_words: int = 600,
    num_ctx_processing: int = 8192,
    general_collection: str = "souli_chunks_improved",
    persona_path: str = "data/coach_persona.txt",
    skip_persona: bool = False,
    skip_ingest: bool = False,
) -> Dict[str, str]:
    """
    Run the full multi-collection ingestion pipeline for one YouTube video.

    Args:
        cfg:                 Loaded PipelineConfig
        youtube_url:         Full YouTube video URL
        out_dir:             Directory to save all output files
        source_label:        Human-readable label for this video (used in Qdrant payloads)
        whisper_model:       Whisper model size (tiny/base/small/medium/large)
        similarity_threshold: Topic boundary sensitivity — lower = more, smaller chunks
        min_topic_words:     Minimum words per topic segment
        max_topic_words:     Maximum words per topic segment
        num_ctx_processing:  Ollama context window for cleaning/extraction
        general_collection:  Name for the general souli_chunks_improved collection
        persona_path:        Path to the evolving coach persona file
        skip_persona:        Skip persona extraction (faster testing)
        skip_ingest:         Skip ALL Qdrant ingestion (produce files only)

    Returns:
        Dict of output_key → file_path (for display/logging)
    """
    os.makedirs(out_dir, exist_ok=True)
    c = cfg.conversation
    r = cfg.retrieval

    outputs: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Step 1 — Whisper transcription  (existing, unchanged)
    # ------------------------------------------------------------------
    logger.info("[MULTI] Step 1/8 — Whisper transcription: %s", youtube_url)
    from souli_pipeline.youtube.whisper_transcribe import transcribe_url

    segments = transcribe_url(
        url=youtube_url,
        out_dir=out_dir,
        whisper_model=whisper_model,
        language="en",
    )

    if not segments:
        logger.warning("[MULTI] No segments from Whisper — aborting for %s", youtube_url)
        return outputs

    df_segments = pd.DataFrame(segments)
    path_segments = os.path.join(out_dir, "whisper_segments.xlsx")
    df_segments.to_excel(path_segments, index=False)
    outputs["whisper_segments"] = path_segments
    logger.info("[MULTI] Whisper done: %d segments", len(df_segments))

    # ------------------------------------------------------------------
    # Step 2 — Topic segmentation + LLM cleaning  (existing, unchanged)
    # ------------------------------------------------------------------
    logger.info("[MULTI] Step 2/8 — Topic segmentation + cleaning...")
    from souli_pipeline.youtube.topic_segmenter import detect_topics
    from souli_pipeline.youtube.segment_cleaner import clean_all_segments

    paragraphs, topic_segments = detect_topics(
        segments=segments,
        similarity_threshold=similarity_threshold,
        min_topic_words=min_topic_words,
        max_topic_words=max_topic_words,
        embedding_model=r.embedding_model or "sentence-transformers/all-MiniLM-L6-v2",
    )

    # Save paragraphs
    df_paragraphs = pd.DataFrame([
        {"index": p.index, "start": p.start, "end": p.end,
         "text": p.text, "word_count": p.word_count}
        for p in paragraphs
    ])
    path_paragraphs = os.path.join(out_dir, "paragraphs.xlsx")
    df_paragraphs.to_excel(path_paragraphs, index=False)
    outputs["paragraphs"] = path_paragraphs

    # Save topic segments
    df_topics = pd.DataFrame([
        {"topic_index": t.topic_index, "start": t.start, "end": t.end,
         "word_count": t.word_count, "text": t.text}
        for t in topic_segments
    ])
    path_topics = os.path.join(out_dir, "topic_segments.xlsx")
    df_topics.to_excel(path_topics, index=False)
    outputs["topic_segments"] = path_topics

    logger.info("[MULTI] %d topic segments from %d paragraphs", len(topic_segments), len(paragraphs))

    if not topic_segments:
        logger.warning("[MULTI] No topic segments — aborting.")
        return outputs

    # LLM cleaning
    cleaned_results = clean_all_segments(
        topic_segments=topic_segments,
        ollama_model=c.chat_model,
        ollama_endpoint=c.ollama_endpoint,
        num_ctx=num_ctx_processing,
        temperature=0.3,
    )

    # Attach source metadata
    for item in cleaned_results:
        item["source_video"] = source_label or youtube_url
        item["youtube_url"]  = youtube_url

    df_cleaned = pd.DataFrame(cleaned_results)
    path_cleaned = os.path.join(out_dir, "cleaned_chunks.xlsx")
    df_cleaned.to_excel(path_cleaned, index=False)
    outputs["cleaned_chunks"] = path_cleaned
    logger.info("[MULTI] Cleaned chunks saved: %d rows", len(df_cleaned))

    # Build full transcript string — used for density detection + extractors
    full_transcript = " ".join(
        item.get("cleaned_text", "") for item in cleaned_results
    ).strip()

    # ------------------------------------------------------------------
    # Step 3 — Persona extraction  (existing, unchanged)
    # ------------------------------------------------------------------
    if not skip_persona:
        logger.info("[MULTI] Step 3/8 — Persona extraction...")
        try:
            from souli_pipeline.youtube.persona_extractor import (
                extract_from_video,
                update_persona_file,
            )
            snippet = extract_from_video(
                cleaned_transcript=full_transcript,
                ollama_model=c.chat_model,
                ollama_endpoint=c.ollama_endpoint,
                num_ctx=num_ctx_processing,
            )
            if snippet:
                snippet_path = os.path.join(out_dir, "persona_snippet.txt")
                with open(snippet_path, "w", encoding="utf-8") as f:
                    f.write(snippet)
                outputs["persona_snippet"] = snippet_path
                update_persona_file(
                    persona_path=persona_path,
                    new_snippet=snippet,
                    ollama_model=c.chat_model,
                    ollama_endpoint=c.ollama_endpoint,
                    num_ctx=num_ctx_processing,
                )
                outputs["coach_persona"] = persona_path
                logger.info("[MULTI] Persona updated at %s", persona_path)
            else:
                logger.warning("[MULTI] No persona snippet extracted.")
        except Exception as exc:
            logger.warning("[MULTI] Persona extraction failed: %s — continuing.", exc)
    else:
        logger.info("[MULTI] Step 3/8 — Persona extraction skipped.")

    # ------------------------------------------------------------------
    # Step 4 — Energy node tagging  (existing, unchanged)
    # ------------------------------------------------------------------
    logger.info("[MULTI] Step 4/8 — Energy node tagging (%d chunks)...", len(df_cleaned))
    detected_node: Optional[str] = None

    try:
        from souli_pipeline.youtube.energy_tagger import tag_dataframe

        df_cleaned = tag_dataframe(
            df_cleaned,
            text_col="cleaned_text",
            ollama_model=c.tagger_model,
            ollama_endpoint=c.ollama_endpoint,
        )

        tagged_path = os.path.join(out_dir, "cleaned_chunks_tagged.xlsx")
        df_cleaned.to_excel(tagged_path, index=False)
        outputs["cleaned_chunks_tagged"] = tagged_path

        # Pick the most common node across chunks as the video's dominant node
        if "energy_node" in df_cleaned.columns:
            node_counts = df_cleaned["energy_node"].value_counts()
            if not node_counts.empty:
                detected_node = node_counts.index[0]
                logger.info("[MULTI] Dominant energy node detected: %s", detected_node)

        tagged_count = df_cleaned["energy_node"].notna().sum()
        logger.info("[MULTI] Energy tagging done: %d tagged", tagged_count)

    except Exception as exc:
        logger.warning("[MULTI] Energy tagging failed: %s — continuing without node.", exc)

    # ------------------------------------------------------------------
    # Step 5 — General ingest → souli_chunks_improved  (existing function, new call)
    # ------------------------------------------------------------------
    if not skip_ingest and not df_cleaned.empty:
        logger.info("[MULTI] Step 5/8 — General ingest → '%s'...", general_collection)
        try:
            from souli_pipeline.retrieval.qdrant_store_improved import ingest_improved_chunks

            n_general = ingest_improved_chunks(
                df=df_cleaned,
                collection=general_collection,
                embedding_model=r.embedding_model or "sentence-transformers/all-MiniLM-L6-v2",
                host=r.qdrant_host,
                port=r.qdrant_port,
            )
            logger.info("[MULTI] General ingest done: %d chunks → '%s'", n_general, general_collection)
            outputs["general_ingested_count"] = str(n_general)
        except Exception as exc:
            logger.warning("[MULTI] General ingest failed: %s", exc)
            n_general = 0
    else:
        logger.info("[MULTI] Step 5/8 — General ingest skipped.")
        n_general = 0

    # ------------------------------------------------------------------
    # Step 6 — Content density detection  (NEW)
    # ------------------------------------------------------------------
    logger.info("[MULTI] Step 6/8 — Content density detection...")
    from souli_pipeline.youtube.content_density_detector import detect_content_density

    density_report = detect_content_density(
        transcript=full_transcript,
        energy_node=detected_node,
        ollama_model=c.chat_model,
        ollama_endpoint=c.ollama_endpoint,
    )

    # Save density report for inspection
    density_path = os.path.join(out_dir, "density_report.json")
    with open(density_path, "w", encoding="utf-8") as f:
        json.dump(density_report, f, indent=2)
    outputs["density_report"] = density_path

    logger.info(
        "[MULTI] Density: healing=%s activity=%s story=%s commitment=%s pattern=%s node=%s",
        density_report["healing_rich"], density_report["activity_rich"],
        density_report["story_rich"],   density_report["commitment_rich"],
        density_report["pattern_rich"], density_report["dominant_node"],
    )

    # Use density report's node if we didn't get one from tagging
    if not detected_node:
        detected_node = density_report.get("dominant_node", "blocked_energy")

    # ------------------------------------------------------------------
    # Step 7 — Multi-extractor run  (NEW)
    # ------------------------------------------------------------------
    logger.info("[MULTI] Step 7/8 — Running typed extractors...")
    from souli_pipeline.youtube.multi_extractors import run_extractors_from_density

    extractor_outputs = run_extractors_from_density(
        transcript=full_transcript,
        energy_node=detected_node,
        density_report=density_report,
        ollama_model=c.chat_model,
        ollama_endpoint=c.ollama_endpoint,
    )

    # Save each extractor's output to its own Excel file for inspection
    extractor_excels: Dict[str, str] = {}
    for chunk_type, chunks in extractor_outputs.items():
        if chunks:
            df_ext = pd.DataFrame(chunks)
            ext_path = os.path.join(out_dir, f"extracted_{chunk_type}.xlsx")
            df_ext.to_excel(ext_path, index=False)
            extractor_excels[chunk_type] = ext_path
            outputs[f"extracted_{chunk_type}"] = ext_path
            logger.info("[MULTI] Saved %d %s chunks → %s", len(chunks), chunk_type, ext_path)
        else:
            logger.info("[MULTI] No %s chunks extracted (skipped or empty).", chunk_type)

    # ------------------------------------------------------------------
    # Step 8 — Typed collection ingest  (NEW)
    # ------------------------------------------------------------------
    ingest_summary: Dict[str, int] = {"general": n_general}

    if not skip_ingest:
        logger.info("[MULTI] Step 8/8 — Typed collection ingest...")
        from souli_pipeline.retrieval.qdrant_store_multi import ingest_all_extractor_outputs

        # Attach source metadata to all extractor chunks before ingesting
        for chunk_type, chunks in extractor_outputs.items():
            for chunk in chunks:
                chunk["source_video"] = source_label or youtube_url
                chunk["youtube_url"]  = youtube_url

        typed_counts = ingest_all_extractor_outputs(
            extractor_outputs=extractor_outputs,
            embedding_model=r.embedding_model or "sentence-transformers/all-MiniLM-L6-v2",
            host=r.qdrant_host,
            port=r.qdrant_port,
        )
        ingest_summary.update(typed_counts)

        logger.info("[MULTI] Typed ingest done: %s", typed_counts)
    else:
        logger.info("[MULTI] Step 8/8 — Typed ingest skipped (skip_ingest=True).")

    # Save ingest summary
    summary_path = os.path.join(out_dir, "ingest_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(ingest_summary, f, indent=2)
    outputs["ingest_summary"] = summary_path

    # Final log
    total_ingested = sum(ingest_summary.values())
    logger.info(
        "[MULTI] Pipeline complete. Collections populated: %s | Total chunks: %d",
        ingest_summary, total_ingested,
    )

    return outputs