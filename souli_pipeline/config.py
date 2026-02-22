from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class RunConfig(BaseModel):
    outputs_dir: str = "outputs"
    max_workers: int = 2

class EnergyGates(BaseModel):
    min_problem_len: int = 12
    min_duality_len: int = 25
    min_blocks_len: int = 8
    min_blocks_count: int = 2

class EnergyConfig(BaseModel):
    expressions_sheet: str = "ExpressionsMapping"
    framework_sheet: str = "Inner energy Framework"
    required_expr_cols: List[str]
    framework_key_col: str = "energy_node"
    framework_cols: List[str]
    aspects_allowed: List[str]
    nodes_allowed: List[str]
    gates: EnergyGates = Field(default_factory=EnergyGates)
    # Optional: map Excel column names -> internal names (e.g. "Main Question" -> "Problem statement")
    expr_column_map: Optional[Dict[str, str]] = None

class ChunkingConfig(BaseModel):
    max_seconds: float = 55
    max_words: int = 220
    max_gap: float = 1.3
    min_words_to_split: int = 35

class YoutubeSegmentsConfig(BaseModel):
    min_dur: float = 0.35
    min_words: int = 2
    max_gap: float = 0.20

class YoutubeCleaningConfig(BaseModel):
    overlap_words: int = 20

class YoutubeClassifyConfig(BaseModel):
    min_words_noise: int = 25
    min_words_teaching: int = 30

class YoutubeScoringConfig(BaseModel):
    meaning_min_score: int = 3
    junk_drop_threshold: int = 7

class YoutubeConfig(BaseModel):
    caption_langs: str = "en,hi"
    whisper_model: str = "medium"
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    segments: YoutubeSegmentsConfig = Field(default_factory=YoutubeSegmentsConfig)
    cleaning: YoutubeCleaningConfig = Field(default_factory=YoutubeCleaningConfig)
    classify: YoutubeClassifyConfig = Field(default_factory=YoutubeClassifyConfig)
    scoring: YoutubeScoringConfig = Field(default_factory=YoutubeScoringConfig)

class LLMHttpJsonConfig(BaseModel):
    endpoint: str
    timeout_s: int = 60

class LLMConfig(BaseModel):
    enabled: bool = False
    adapter: str = "none"  # none | http_json | local_hf
    http_json: Optional[LLMHttpJsonConfig] = None


class RetrievalConfig(BaseModel):
    """Local-only retrieval: no data sent to external APIs."""
    embedding_model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"
    top_k_teaching: int = 5


class PipelineConfig(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    energy: EnergyConfig
    youtube: YoutubeConfig = Field(default_factory=YoutubeConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
