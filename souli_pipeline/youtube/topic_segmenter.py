"""
souli_pipeline/youtube/topic_segmenter.py

Two-stage process:
  Stage A — Paragraph grouping
      Groups consecutive Whisper segments into paragraphs of ~150 words,
      respecting natural silence gaps (gap > silence_gap_s = new paragraph).

  Stage B — Topic boundary detection
      Embeds each paragraph using all-MiniLM-L6-v2.
      Computes cosine similarity between every consecutive paragraph pair.
      Where similarity drops below threshold → topic boundary.
      Merges paragraphs within the same topic into one TopicSegment.

No LLM involved. Fast and deterministic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Paragraph:
    index: int
    start: float
    end: float
    text: str
    word_count: int
    embedding: Optional[List[float]] = field(default=None, repr=False)


@dataclass
class TopicSegment:
    topic_index: int
    start: float
    end: float
    text: str          # all paragraphs in this topic joined
    word_count: int
    paragraph_indices: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage A — Paragraph grouping
# ---------------------------------------------------------------------------

def group_into_paragraphs(
    segments: List[Dict],
    target_words: int = 150,
    silence_gap_s: float = 1.5,
) -> List[Paragraph]:
    """
    Group Whisper segments into paragraphs.

    A new paragraph starts when:
      - Gap between segments exceeds silence_gap_s (natural pause), OR
      - Current paragraph already has >= target_words

    Args:
        segments:      List of {start, end, text, confidence} dicts from whisper_transcribe
        target_words:  Soft word limit per paragraph
        silence_gap_s: Gap in seconds that forces a paragraph break

    Returns:
        List of Paragraph objects
    """
    if not segments:
        return []

    paragraphs: List[Paragraph] = []
    current_texts: List[str] = []
    current_start: float = segments[0]["start"]
    current_end: float = segments[0]["end"]
    current_words: int = 0
    prev_end: float = segments[0]["end"]

    def flush(end_time: float):
        nonlocal current_texts, current_start, current_end, current_words
        text = " ".join(current_texts).strip()
        if text:
            paragraphs.append(Paragraph(
                index=len(paragraphs),
                start=current_start,
                end=end_time,
                text=text,
                word_count=current_words,
            ))
        current_texts = []
        current_words = 0

    for seg in segments:
        gap = float(seg["start"]) - prev_end
        words_here = len(seg["text"].split())

        # Force new paragraph on silence gap OR word limit reached
        if current_texts and (gap > silence_gap_s or current_words >= target_words):
            flush(prev_end)
            current_start = float(seg["start"])

        current_texts.append(seg["text"])
        current_end = float(seg["end"])
        current_words += words_here
        prev_end = float(seg["end"])

    if current_texts:
        flush(current_end)

    logger.info("Grouped %d segments into %d paragraphs.", len(segments), len(paragraphs))
    return paragraphs


# ---------------------------------------------------------------------------
# Stage B — Embedding + topic boundary detection
# ---------------------------------------------------------------------------

def _cosine_sim(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _embed_paragraphs(
    paragraphs: List[Paragraph],
    model_name: str = _DEFAULT_MODEL,
) -> List[Paragraph]:
    """Embed each paragraph's text in-place. Returns same list."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        texts = [p.text for p in paragraphs]
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()
        for p, emb in zip(paragraphs, embeddings):
            p.embedding = emb
        logger.info("Embedded %d paragraphs.", len(paragraphs))
    except Exception as e:
        logger.warning("Embedding failed: %s — topic detection will use single segment.", e)
    return paragraphs


def _find_boundaries(
    paragraphs: List[Paragraph],
    threshold: float = 0.45,
) -> List[int]:
    """
    Return indices of paragraphs that START a new topic.
    Index 0 is always a boundary (start of video).

    A boundary is placed BEFORE paragraph i when:
        cosine_similarity(paragraph[i-1], paragraph[i]) < threshold

    Lower threshold = more sensitive = more chunks.
    Higher threshold = less sensitive = fewer, larger chunks.
    """
    if len(paragraphs) <= 1:
        return [0]

    boundaries = [0]
    similarities: List[Tuple[int, float]] = []

    for i in range(1, len(paragraphs)):
        a = paragraphs[i - 1].embedding
        b = paragraphs[i].embedding
        if a is None or b is None:
            # No embeddings available — treat every paragraph as its own topic
            boundaries.append(i)
            continue
        sim = _cosine_sim(a, b)
        similarities.append((i, sim))
        if sim < threshold:
            boundaries.append(i)
            logger.debug(
                "Topic boundary at paragraph %d (sim=%.3f < %.3f)", i, sim, threshold
            )

    # Log similarity stats for threshold tuning
    if similarities:
        sims = [s for _, s in similarities]
        avg_sim = sum(sims) / len(sims)
        min_sim = min(sims)
        max_sim = max(sims)
        logger.info(
            "Similarity stats — min: %.3f, avg: %.3f, max: %.3f | boundaries: %d",
            min_sim, avg_sim, max_sim, len(boundaries),
        )

    return boundaries


def _merge_into_topics(
    paragraphs: List[Paragraph],
    boundaries: List[int],
    min_words: int = 80,
    max_words: int = 600,
) -> List[TopicSegment]:
    """
    Merge paragraphs within each topic boundary into TopicSegment objects.

    If a merged segment exceeds max_words, it is split into sub-segments
    of max_words each (simple word-count split — happens rarely).

    If a merged segment is under min_words, it is merged into the previous
    topic (avoids micro-chunks).
    """
    if not paragraphs:
        return []

    boundary_set = set(boundaries)
    topics: List[TopicSegment] = []
    current_paras: List[Paragraph] = []

    def flush_topic():
        if not current_paras:
            return
        combined_text = " ".join(p.text for p in current_paras).strip()
        total_words = sum(p.word_count for p in current_paras)

        # If too small, absorb into previous topic
        if total_words < min_words and topics:
            prev = topics[-1]
            topics[-1] = TopicSegment(
                topic_index=prev.topic_index,
                start=prev.start,
                end=current_paras[-1].end,
                text=prev.text + " " + combined_text,
                word_count=prev.word_count + total_words,
                paragraph_indices=prev.paragraph_indices + [p.index for p in current_paras],
            )
            return

        # If too large, word-count split
        if total_words > max_words:
            words = combined_text.split()
            start_t = current_paras[0].start
            end_t   = current_paras[-1].end
            step    = max_words
            for i in range(0, len(words), step):
                chunk_words = words[i : i + step]
                if len(chunk_words) < min_words // 2:
                    # Too small final piece — absorb into previous
                    if topics:
                        prev = topics[-1]
                        topics[-1] = TopicSegment(
                            topic_index=prev.topic_index,
                            start=prev.start,
                            end=end_t,
                            text=prev.text + " " + " ".join(chunk_words),
                            word_count=prev.word_count + len(chunk_words),
                            paragraph_indices=prev.paragraph_indices,
                        )
                    continue
                # Approximate timestamps for sub-splits
                frac_start = start_t + (end_t - start_t) * (i / len(words))
                frac_end   = start_t + (end_t - start_t) * (min(i + step, len(words)) / len(words))
                topics.append(TopicSegment(
                    topic_index=len(topics),
                    start=round(frac_start, 3),
                    end=round(frac_end, 3),
                    text=" ".join(chunk_words),
                    word_count=len(chunk_words),
                    paragraph_indices=[p.index for p in current_paras],
                ))
            return

        topics.append(TopicSegment(
            topic_index=len(topics),
            start=current_paras[0].start,
            end=current_paras[-1].end,
            text=combined_text,
            word_count=total_words,
            paragraph_indices=[p.index for p in current_paras],
        ))

    for i, para in enumerate(paragraphs):
        if i in boundary_set and current_paras:
            flush_topic()
            current_paras = []
        current_paras.append(para)

    flush_topic()

    logger.info(
        "Produced %d topic segments (words: min=%d, max=%d, avg=%d)",
        len(topics),
        min(t.word_count for t in topics) if topics else 0,
        max(t.word_count for t in topics) if topics else 0,
        int(sum(t.word_count for t in topics) / max(1, len(topics))),
    )
    return topics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_topics(
    segments: List[Dict],
    similarity_threshold: float = 0.45,
    target_paragraph_words: int = 150,
    silence_gap_s: float = 1.5,
    min_topic_words: int = 80,
    max_topic_words: int = 600,
    embedding_model: str = _DEFAULT_MODEL,
) -> Tuple[List[Paragraph], List[TopicSegment]]:
    """
    Full pipeline: Whisper segments → paragraphs → topic segments.

    Returns (paragraphs, topic_segments).
    Both are returned so intermediate data can be saved to xlsx for inspection.

    Args:
        segments:                 Output of whisper_transcribe.transcribe_url()
        similarity_threshold:     Cosine sim below this = new topic (tune via config)
        target_paragraph_words:   Soft word limit for paragraph grouping
        silence_gap_s:            Silence gap forcing a paragraph break
        min_topic_words:          Minimum words per topic (smaller get absorbed)
        max_topic_words:          Maximum words per topic (larger get split)
        embedding_model:          SentenceTransformer model name
    """
    paragraphs   = group_into_paragraphs(segments, target_paragraph_words, silence_gap_s)
    paragraphs   = _embed_paragraphs(paragraphs, embedding_model)
    boundaries   = _find_boundaries(paragraphs, similarity_threshold)
    topic_segs   = _merge_into_topics(paragraphs, boundaries, min_topic_words, max_topic_words)
    return paragraphs, topic_segs