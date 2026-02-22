from __future__ import annotations
from typing import Optional
from ..config import PipelineConfig
from .http_json import HttpJsonLLM

def make_llm(cfg: PipelineConfig):
    if not cfg.llm.enabled or cfg.llm.adapter == "none":
        return None
    if cfg.llm.adapter == "http_json":
        if not cfg.llm.http_json:
            raise ValueError("llm.http_json missing in config")
        return HttpJsonLLM(cfg.llm.http_json.endpoint, cfg.llm.http_json.timeout_s)
    raise ValueError(f"Unknown LLM adapter: {cfg.llm.adapter}")
