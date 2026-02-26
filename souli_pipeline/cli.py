from __future__ import annotations
import json
import os
import asyncio
import typer
from rich import print
from .config_loader import load_config
from .utils.run_id import get_run_id
from .energy.pipeline import run_energy_pipeline
from .youtube.pipeline import run_youtube_pipeline
from .youtube.playlist import list_playlist_videos
from .youtube.videos_csv import load_videos_csv
from .youtube.merge_outputs import merge_teaching_outputs
from .retrieval.match import run_match

app = typer.Typer(no_args_is_help=True)

@app.command()
def health():
    """Check pipeline health and Ollama/Qdrant availability."""
    print("[green]Pipeline: ok[/green]")
    try:
        from .llm.ollama import OllamaLLM
        llm = OllamaLLM()
        if llm.is_available():
            models = llm.list_models()
            print(f"[green]Ollama: ok[/green] — models: {models}")
        else:
            print("[yellow]Ollama: not running[/yellow] — start with: ollama serve")
    except Exception as e:
        print(f"[red]Ollama check failed: {e}[/red]")
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(host="localhost", port=6333, timeout=3)
        qc.get_collections()
        print("[green]Qdrant: ok[/green]")
    except Exception:
        print("[yellow]Qdrant: not running[/yellow] — start with: docker run -p 6333:6333 qdrant/qdrant")


run_app = typer.Typer(no_args_is_help=True)
app.add_typer(run_app, name="run")

@run_app.command("energy")
def run_energy(
    config: str = typer.Option(..., "--config", "-c"),
    excel_path: str = typer.Option(..., "--excel-path"),
):
    cfg = load_config(config)
    rid = get_run_id()
    out_dir = os.path.join(cfg.run.outputs_dir, rid, "energy")
    gold, rej = run_energy_pipeline(cfg, excel_path=excel_path, out_dir=out_dir)
    print(f"[green]Saved gold:[/green] {gold}")
    print(f"[yellow]Saved reject:[/yellow] {rej}")

@run_app.command("youtube")
def run_youtube(
    config: str = typer.Option(..., "--config", "-c"),
    youtube_url: str = typer.Option(..., "--youtube-url"),
    no_tag: bool = typer.Option(False, "--no-tag", help="Skip Qwen energy node tagging"),
):
    cfg = load_config(config)
    rid = get_run_id()
    out_dir = os.path.join(cfg.run.outputs_dir, rid, "youtube")
    out = run_youtube_pipeline(cfg, youtube_url=youtube_url, out_dir=out_dir, tag_energy=not no_tag)
    print("[green]Done.[/green]")
    for k, v in out.items():
        print(f" - {k}: {v}")

@run_app.command("playlist")
def run_playlist(
    config: str = typer.Option(..., "--config", "-c"),
    playlist_url: str = typer.Option(..., "--playlist-url"),
    no_tag: bool = typer.Option(False, "--no-tag"),
):
    cfg = load_config(config)
    rid = get_run_id()
    urls = list_playlist_videos(playlist_url)
    print(f"[cyan]Found {len(urls)} videos[/cyan]")
    for i, url in enumerate(urls, 1):
        out_dir = os.path.join(cfg.run.outputs_dir, rid, "youtube", f"video_{i:03d}")
        out = run_youtube_pipeline(cfg, youtube_url=url, out_dir=out_dir, tag_energy=not no_tag)
        print(f"[green]{i}/{len(urls)}[/green] {url}")
        for k, v in out.items():
            print(f"   - {k}: {v}")

@run_app.command("videos")
def run_videos(
    config: str = typer.Option(..., "--config", "-c"),
    videos_csv: str = typer.Option(..., "--videos-csv"),
    merge: bool = typer.Option(True, "--merge/--no-merge"),
    no_tag: bool = typer.Option(False, "--no-tag", help="Skip Qwen energy node tagging"),
):
    """Run YouTube pipeline for each video in CSV with Qwen energy tagging."""
    cfg = load_config(config)
    rid = get_run_id()
    videos = load_videos_csv(videos_csv)
    if not videos:
        print("[red]No videos found in CSV.[/red]")
        raise SystemExit(1)
    print(f"[cyan]Found {len(videos)} videos in CSV[/cyan]")
    base_dir = os.path.join(cfg.run.outputs_dir, rid, "youtube")
    video_results: list = []
    for v in videos:
        i = v["video_index"]
        url = v["url"]
        out_dir = os.path.join(base_dir, f"video_{i:03d}")
        out = run_youtube_pipeline(
            cfg, youtube_url=url, out_dir=out_dir,
            source_label=v["source_label"], tag_energy=not no_tag,
        )
        video_results.append({"out_dir": out_dir, "source_label": v["source_label"], **out})
        print(f"[green]{i}/{len(videos)}[/green] {v['source_label'][:60]}...")
        for k, path in out.items():
            if isinstance(path, str):
                print(f"   - {k}: {path}")
    if merge and video_results:
        merged = merge_teaching_outputs(video_results, base_dir)
        if merged:
            print("[green]Merged outputs:[/green]")
            for k, path in merged.items():
                print(f" - {k}: {path}")

@run_app.command("all")
def run_all(
    config: str = typer.Option(..., "--config", "-c"),
    videos_csv: str = typer.Option(..., "--videos-csv"),
    excel_path: str = typer.Option(None, "--excel-path"),
    merge: bool = typer.Option(True, "--merge/--no-merge"),
    no_tag: bool = typer.Option(False, "--no-tag"),
):
    """Run energy pipeline + all videos. Single run_id."""
    cfg = load_config(config)
    rid = get_run_id()
    base = os.path.join(cfg.run.outputs_dir, rid)
    if excel_path:
        if not os.path.isfile(excel_path):
            print(f"[red]Excel not found: {excel_path}[/red]")
            raise SystemExit(1)
        energy_dir = os.path.join(base, "energy")
        gold, rej = run_energy_pipeline(cfg, excel_path=excel_path, out_dir=energy_dir)
        print(f"[green]Energy done:[/green] {gold} | {rej}")
    videos = load_videos_csv(videos_csv)
    if not videos:
        print("[red]No videos in CSV.[/red]")
        raise SystemExit(1)
    print(f"[cyan]Videos: {len(videos)}[/cyan]")
    base_dir = os.path.join(base, "youtube")
    video_results = []
    for v in videos:
        i = v["video_index"]
        out_dir = os.path.join(base_dir, f"video_{i:03d}")
        out = run_youtube_pipeline(
            cfg, youtube_url=v["url"], out_dir=out_dir,
            source_label=v["source_label"], tag_energy=not no_tag,
        )
        video_results.append({"out_dir": out_dir, "source_label": v["source_label"], **out})
        print(f"[green]{i}/{len(videos)}[/green] {v['source_label'][:60]}...")
    if merge and video_results:
        merged = merge_teaching_outputs(video_results, base_dir)
        if merged:
            print("[green]Merged (source_video):[/green]")
            for k, path in merged.items():
                print(f" - {k}: {path}")

@app.command("match")
def match_cmd(
    config: str = typer.Option(..., "--config", "-c"),
    gold_path: str = typer.Option(..., "--gold"),
    teaching_path: str = typer.Option(None, "--teaching"),
    query: str = typer.Option(..., "--query", "-q"),
    output: str = typer.Option("json", "--output", "-o"),
):
    """Diagnose user text -> energy_node + framework solution + teaching content. All local."""
    cfg = load_config(config)
    nodes = cfg.energy.nodes_allowed
    emb = cfg.retrieval.embedding_model if hasattr(cfg, "retrieval") else None
    top_k = getattr(cfg.retrieval, "top_k_teaching", 5) if hasattr(cfg, "retrieval") else 5
    result = run_match(
        user_query=query, gold_path=gold_path, nodes_allowed=nodes,
        teaching_path=teaching_path, embedding_model=emb, top_k_teaching=top_k,
    )
    if output == "text":
        print("[cyan]Diagnosis[/cyan]")
        print("  energy_node:", result["diagnosis"]["energy_node"])
        print("  aspect:", result["diagnosis"]["aspect"])
        print("  matched_problem:", result["diagnosis"].get("matched_problem") or "(keyword fallback)")
        print("[cyan]Framework solution[/cyan]")
        for k, v in (result.get("framework_solution") or {}).items():
            if v:
                print(f"  {k}: {v[:200]}..." if len(str(v)) > 200 else f"  {k}: {v}")
        print("[cyan]Teaching content (top %d)[/cyan]" % top_k)
        for i, t in enumerate(result.get("teaching_content") or [], 1):
            print(f"  --- {i} ---")
            for k, v in t.items():
                if v:
                    print(f"    {k}: {v[:150]}..." if len(str(v)) > 150 else f"    {k}: {v}")
    else:
        out = {k: v for k, v in result.items() if k != "local_only"}
        print(json.dumps(out, indent=2, ensure_ascii=False))

@app.command("ingest")
def ingest_cmd(
    config: str = typer.Option(..., "--config", "-c"),
    outputs_dir: str = typer.Option(None, "--outputs-dir", "-o"),
    teaching_file: str = typer.Option(None, "--file", "-f"),
    collection: str = typer.Option(None, "--collection"),
):
    """Embed YouTube teaching chunks and ingest into Qdrant vector store."""
    cfg = load_config(config)
    r = cfg.retrieval
    coll = collection or r.qdrant_collection
    emb = r.embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
    from .retrieval.qdrant_store import ingest_from_excel, ingest_pipeline_outputs
    if teaching_file:
        if not os.path.isfile(teaching_file):
            print(f"[red]File not found: {teaching_file}[/red]")
            raise SystemExit(1)
        n = ingest_from_excel(teaching_file, collection=coll, embedding_model=emb, host=r.qdrant_host, port=r.qdrant_port)
        print(f"[green]Ingested {n} chunks from {teaching_file}[/green]")
    else:
        search_dir = outputs_dir or cfg.run.outputs_dir
        print(f"[cyan]Walking {search_dir} for teaching_ready.xlsx files...[/cyan]")
        n = ingest_pipeline_outputs(search_dir, collection=coll, embedding_model=emb, host=r.qdrant_host, port=r.qdrant_port)
        print(f"[green]Total ingested: {n} chunks into collection '{coll}'[/green]")

@app.command("chat")
def chat_cmd(
    config: str = typer.Option(..., "--config", "-c"),
    gold_path: str = typer.Option(None, "--gold"),
    excel_path: str = typer.Option(None, "--excel"),
    stream: bool = typer.Option(True, "--stream/--no-stream"),
):
    """Start a text-based Souli wellness conversation. Type 'quit' to exit."""
    from rich.console import Console
    from rich.panel import Panel
    from .conversation.engine import ConversationEngine

    console = Console()
    cfg = load_config(config)

    console.print(Panel.fit(
        "[bold cyan]Welcome to Souli[/bold cyan]\n"
        "[dim]Your inner wellness companion. Type 'quit' to exit.[/dim]",
        border_style="cyan",
    ))

    engine = ConversationEngine.from_config(cfg, gold_path=gold_path, excel_path=excel_path)
    greeting = engine.greeting()
    console.print(f"\n[bold green]Souli:[/bold green] {greeting}\n")

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session ended.[/dim]")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            console.print("\n[bold green]Souli:[/bold green] Take care of yourself. I'm always here.")
            break

        console.print(f"\n[bold green]Souli:[/bold green] ", end="")
        if stream:
            try:
                for chunk in engine.turn_stream(user_input):
                    console.print(chunk, end="")
                console.print()
            except Exception:
                response = engine.turn(user_input)
                console.print(response)
        else:
            response = engine.turn(user_input)
            console.print(response)

        diag = engine.diagnosis_summary
        if diag.get("energy_node"):
            console.print(
                f"[dim]  [node: {diag['energy_node']} | phase: {diag['phase']} | turn: {diag['turn_count']}][/dim]\n"
            )

@app.command("voice")
def voice_cmd(
    config: str = typer.Option(..., "--config", "-c"),
    gold_path: str = typer.Option(None, "--gold"),
    excel_path: str = typer.Option(None, "--excel"),
    local: bool = typer.Option(False, "--local", help="Run local voice test (no LiveKit server needed)"),
):
    """Start Souli voice pipeline. Use --local for mic/speaker test without LiveKit server."""
    cfg = load_config(config)
    from .voice.livekit_agent import SouliVoiceAgent
    agent = SouliVoiceAgent(cfg, gold_path=gold_path, excel_path=excel_path)
    if local:
        print("[cyan]Starting local voice mode (mic -> Souli -> speaker)...[/cyan]")
        print("[dim]Press Ctrl+C to stop.[/dim]")
        asyncio.run(agent.run_local_voice())
    else:
        v = cfg.voice
        print(f"[cyan]Connecting to LiveKit room '{v.room_name}' at {v.livekit_url}...[/cyan]")
        asyncio.run(agent.run())

@app.command("tag")
def tag_cmd(
    config: str = typer.Option(..., "--config", "-c"),
    input_file: str = typer.Option(..., "--input", "-i"),
    output_file: str = typer.Option(None, "--output", "-o"),
):
    """Tag an existing teaching_ready.xlsx with Qwen energy nodes."""
    import pandas as pd
    from .youtube.energy_tagger import tag_dataframe
    cfg = load_config(config)
    c = cfg.conversation
    if not os.path.isfile(input_file):
        print(f"[red]File not found: {input_file}[/red]")
        raise SystemExit(1)
    df = pd.read_excel(input_file)
    print(f"[cyan]Tagging {len(df)} chunks with {c.tagger_model}...[/cyan]")
    df = tag_dataframe(df, text_col="text", ollama_model=c.tagger_model, ollama_endpoint=c.ollama_endpoint)
    out = output_file or input_file
    df.to_excel(out, index=False)
    print(f"[green]Tagged and saved to: {out}[/green]")
