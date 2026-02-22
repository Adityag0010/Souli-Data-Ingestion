# Souli Voice Data Pipeline (Production Skeleton)

This repo turns **YouTube videos / playlists** into clean, structured datasets (chunks, teaching cards, etc.)
and supports your existing **Energy Framework** Excel normalization/enrichment workflow.

It is designed to run:
- locally (Mac/Linux/Windows)
- on a GPU VM (Compute Engine / any docker host)
- as a batch job (playlist → many videos)

## What it does

### A) Energy Framework (Excel)
- Loads `ExpressionsMapping` + `Inner energy Framework`
- Normalizes `Aspects of Woman Track` and `energy_node`
- Heuristically fills missing `energy_node` when blank
- Enriches rows with framework columns
- Applies a strict quality gate → `gold` and `reject`
- Optional recovery passes similar to your notebook (dual/blocks repair)

### B) YouTube → Chunks → Teaching Cards
- Tries captions first (VTT via yt-dlp)
- Falls back to audio + faster-whisper (optional; requires ffmpeg)
- Builds coherent chunks based on time, word limits, gaps
- Cleans/dedupes noisy captions
- Classifies chunks (problem / teaching / noise)
- Scores and filters teaching chunks
- (Optional) Calls an LLM adapter to extract **strict JSON cards**

> Note: For LLM extraction you can plug your own model endpoint or local HF model.
> This skeleton ships with a clean interface so you can swap providers safely.

## Quickstart

### 1) Create venv
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run a single video
```bash
souli run youtube \
  --config configs/pipeline.yaml \
  --youtube-url "https://youtu.be/VIDEO_ID"
```

### 3) Run a playlist (batch)
```bash
souli run playlist \
  --config configs/pipeline.yaml \
  --playlist-url "https://www.youtube.com/playlist?list=PLAYLIST_ID"
```

### 4) Process Excel energy framework
```bash
souli run energy \
  --config configs/pipeline.yaml \
  --excel-path "data/Souli_EnergyFramework_PW (1).xlsx"
```
The default config supports the PW Excel format: sheets **ExpressionsMapping** (columns: Main Question, Category, Related Inner Issues, Reality Commitment Check, energy_node/…) and **Inner energy Framework**. Column mapping is in `configs/pipeline.yaml` under `energy.expr_column_map`.

### 5) Run many videos from a CSV (different data per video)
Use a CSV with column `url` or `youtube_url`. Optional columns: `name`, `video_id`, `title` (used as `source_video` in merged outputs).
```bash
souli run videos \
  --config configs/pipeline.yaml \
  --videos-csv data/videos.csv \
  --merge
```
Each video gets its own folder: `outputs/<run_id>/youtube/video_001/`, `video_002/`, …  
With `--merge` you also get:
- `outputs/<run_id>/youtube/merged_teaching_ready.xlsx` — all teaching chunks with **source_video** column
- `outputs/<run_id>/youtube/merged_teaching_cards.xlsx` — all teaching cards with **source_video** (when LLM is enabled)

See `data/videos.csv.example` for CSV format.

### 6) Full run: energy + all videos in one go
```bash
souli run all \
  --config configs/pipeline.yaml \
  --videos-csv data/videos.csv \
  --excel-path data/Souli_EnergyFramework_PW.xlsx \
  --merge
```

### 7) Match user venting → diagnosis + solution + teaching (local only)
When someone comes to you with a problem (e.g. “I am very sad”), you:
1. **Diagnose** their problem (which energy node: blocked, depleted, scattered, etc.)
2. Get the **framework solution** (practices, meditations) for that node from your Excel
3. Get **teaching content** from your YouTube pipeline output for that same node

All of this runs **locally**. No user data is sent to any external API.

```bash
souli match \
  --config configs/pipeline.yaml \
  --gold outputs/<run_id>/energy/gold.xlsx \
  --teaching outputs/<run_id>/youtube/merged_teaching_cards.xlsx \
  --query "I am very sad" \
  --output json
```

Use `--output text` for readable output. If you omit `--teaching`, you still get diagnosis + framework solution (from gold).  
**Optional:** install `sentence-transformers` for better problem matching; otherwise keyword-based diagnosis is used (no extra deps).

```bash
pip install sentence-transformers
```

## Privacy & open-source

- **No data is sent to any external LLM or API** for training or inference unless you explicitly enable and point to your own endpoint (e.g. local Ollama). Your Excel, YouTube outputs, and user queries stay on your machine.
- **Retrieval/matching** uses optional local embeddings (`sentence-transformers`) or keyword rules only. Everything runs in-process.
- Use only open-source models and local tools so your data is never used to train third-party models.

## Outputs

All outputs go to `outputs/<run_id>/...`:
- **Per video**: `youtube/video_001/`, `video_002/`, … each with:
  - `segments.xlsx`, `chunks_raw.xlsx`, `chunks_clean.xlsx`, `chunks_keep.xlsx`, `teaching_ready.xlsx`, `teaching_cards.xlsx` (if LLM on)
- **Merged** (when using `run videos` or `run all` with `--merge`):
  - `youtube/merged_teaching_ready.xlsx`, `youtube/merged_teaching_cards.xlsx` (column **source_video** = URL or name from CSV)
- **Energy**: `energy/gold.xlsx`, `energy/reject.xlsx`
- **Match** (diagnosis + solution + teaching): use `souli match` with paths to `gold.xlsx` and (optional) `merged_teaching_cards.xlsx`

## Docker & Cloud

Build:
```bash
docker build -t souli-pipeline .
```

Run (mount working directory so config, data, and outputs live on host):
```bash
docker run --rm -it \
  -v "$PWD:/app" \
  -w /app \
  souli-pipeline \
  souli run videos --config configs/pipeline.yaml --videos-csv data/videos.csv --merge
```

For **all videos** from CSV + optional energy Excel:
```bash
docker run --rm -it \
  -v "$PWD:/app" \
  -w /app \
  souli-pipeline \
  souli run all --config configs/pipeline.yaml --videos-csv data/videos.csv --excel-path data/Souli_EnergyFramework_PW.xlsx --merge
```

- Put your `videos.csv` and `Souli_EnergyFramework_PW.xlsx` in `data/` (or any path you mount).
- Outputs go to `outputs/<run_id>/` on the host.
- If using whisper fallback, you need `ffmpeg` (included in Docker image).
- On a cloud VM, run the same commands; use a persistent disk or object storage for `outputs/` and `data/`.

## Environment variables

- `SOULI_LOG_LEVEL` (default: INFO)
- `SOULI_RUN_ID` (optional; otherwise autogenerated)
- LLM adapter vars (depends on which adapter you enable)

## License
Internal use.
