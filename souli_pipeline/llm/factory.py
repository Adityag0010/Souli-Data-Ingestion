from __future__ import annotations
from typing import Optional
from ..config import PipelineConfig
from .http_json import HttpJsonLLM
from .ollama import OllamaLLM


def make_llm(cfg: PipelineConfig):
    if not cfg.llm.enabled or cfg.llm.adapter == "none":
        return None
    if cfg.llm.adapter == "http_json":
        if not cfg.llm.http_json:
            raise ValueError("llm.http_json missing in config")
        return HttpJsonLLM(cfg.llm.http_json.endpoint, cfg.llm.http_json.timeout_s)
    if cfg.llm.adapter == "ollama":
        if not cfg.llm.ollama:
            raise ValueError("llm.ollama missing in config")
        o = cfg.llm.ollama
        return OllamaLLM(
            model=o.model,
            endpoint=o.endpoint,
            timeout_s=o.timeout_s,
            temperature=o.temperature,
            num_ctx=o.num_ctx,
        )
    raise ValueError(f"Unknown LLM adapter: {cfg.llm.adapter}")


def make_ollama(model: str = "llama3.1", endpoint: str = "http://localhost:11434") -> OllamaLLM:
    """Quick helper to get an OllamaLLM without needing full PipelineConfig."""
    return OllamaLLM(model=model, endpoint=endpoint)
