# Copilot Instructions — souli-voice-pipeline

## Architecture Overview

This is a **batch data pipeline** with two independent domains, orchestrated by a Typer CLI (`souli_pipeline/cli.py`) and driven by a single YAML config (`configs/pipeline.yaml`):

1. **YouTube → Teaching Cards** (`souli_pipeline/youtube/`): Downloads captions (VTT via `yt-dlp`), falls back to `faster-whisper` audio transcription, chunks/dedupes/classifies/scores text, then optionally calls an LLM adapter to extract structured teaching cards.
2. **Energy Framework** (`souli_pipeline/energy/`): Reads a multi-sheet Excel workbook, normalizes free-text fields with `rapidfuzz` fuzzy matching, infers missing `energy_node` values via keyword heuristics, enriches rows from a framework lookup, and applies a quality gate producing `gold` / `reject` splits.

Data flows as pandas DataFrames through pure-function transforms; each pipeline writes intermediate `.xlsx` files to `outputs/<run_id>/`.

## Project Conventions

- **Config**: All tunables live in `configs/pipeline.yaml`, parsed into nested Pydantic models in `souli_pipeline/config.py`. Never hard-code thresholds—add them to the relevant Pydantic model and YAML section.
- **CLI entry point**: `souli` (registered via `[project.scripts]` in `pyproject.toml`). Subcommands are `souli run energy`, `souli run youtube`, `souli run playlist`.
- **Logging**: Use `setup_logging(__name__)` from `souli_pipeline/utils/logging.py`. Log level is controlled by `SOULI_LOG_LEVEL` env var.
- **Run IDs**: Each invocation gets a unique `<timestamp>_<hex>` run ID (or override via `SOULI_RUN_ID`). Outputs are always scoped under `outputs/<run_id>/`.
- **LLM adapter pattern**: `souli_pipeline/llm/base.py` defines an `LLMAdapter` Protocol with a single method `extract_teaching_card(transcript) -> Dict[str, str]`. New adapters implement this protocol and are wired in `llm/factory.py`. The factory returns `None` when LLM is disabled.

## Key Patterns to Follow

- **Pipeline functions** accept `PipelineConfig` plus domain-specific args, return paths or DataFrames. See `youtube/pipeline.py:run_youtube_pipeline` and `energy/pipeline.py:run_energy_pipeline` as the canonical examples.
- **Text normalization** in both domains uses regex + `rapidfuzz` for fuzzy matching against allowed-value lists. When adding new normalizers, follow the pattern in `energy/normalize.py` (try exact → fuzzy with score threshold → fallback).
- **Classification & scoring** (`youtube/classify.py`, `youtube/scoring.py`) use keyword/regex heuristics with configurable thresholds, not ML models. Keep these deterministic and config-driven.
- **External tool calls** (`yt-dlp`, `ffmpeg`) are invoked via `subprocess.run` in `youtube/captions.py`, `youtube/playlist.py`, and `youtube/whisper_fallback.py`.

## Development Workflow

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or: make install
make fmt                                # ruff format
make lint                               # ruff check
python -m pytest tests/                 # tests (currently smoke only)
souli --help                            # verify CLI
```

- Formatter/linter: **ruff** (no dedicated config file yet—uses defaults).
- Docker image includes `ffmpeg` for whisper fallback; local dev needs `ffmpeg` installed separately.
- No CI pipeline exists yet; run `make lint && python -m pytest tests/` before committing.

## Adding a New LLM Adapter

1. Create `souli_pipeline/llm/<name>.py` with a class implementing the `LLMAdapter` protocol (must have `extract_teaching_card(self, transcript: str) -> Dict[str, str]`).
2. Add any config fields to `LLMConfig` / a new nested Pydantic model in `config.py`.
3. Register the adapter in `llm/factory.py:make_llm` with a new `elif` branch matching the `llm.adapter` string.
4. Add the YAML config block under `llm:` in `configs/pipeline.yaml`.

## File Reference

| Path | Purpose |
|---|---|
| `configs/pipeline.yaml` | All runtime config (thresholds, model names, allowed values) |
| `souli_pipeline/config.py` | Pydantic models mirroring the YAML structure |
| `souli_pipeline/cli.py` | Typer CLI: `souli run {energy,youtube,playlist}` |
| `souli_pipeline/youtube/pipeline.py` | YouTube orchestrator: captions → chunks → classify → score → LLM |
| `souli_pipeline/energy/pipeline.py` | Energy orchestrator: normalize → enrich → quality gate |
| `souli_pipeline/llm/base.py` | `LLMAdapter` Protocol definition |
| `souli_pipeline/llm/factory.py` | Factory that returns the correct adapter (or `None`) |
