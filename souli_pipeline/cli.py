from __future__ import annotations
import os
import typer
from rich import print
from .config_loader import load_config
from .utils.run_id import get_run_id
from .energy.pipeline import run_energy_pipeline
from .youtube.pipeline import run_youtube_pipeline
from .youtube.playlist import list_playlist_videos
from .youtube.videos_csv import load_videos_csv
from .youtube.merge_outputs import merge_teaching_outputs

app = typer.Typer(no_args_is_help=True)

@app.command()
def health():
    print("[green]ok[/green]")

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
):
    cfg = load_config(config)
    rid = get_run_id()
    out_dir = os.path.join(cfg.run.outputs_dir, rid, "youtube")
    out = run_youtube_pipeline(cfg, youtube_url=youtube_url, out_dir=out_dir)
    print("[green]Done.[/green]")
    for k, v in out.items():
        print(f" - {k}: {v}")

@run_app.command("playlist")
def run_playlist(
    config: str = typer.Option(..., "--config", "-c"),
    playlist_url: str = typer.Option(..., "--playlist-url"),
):
    cfg = load_config(config)
    rid = get_run_id()
    urls = list_playlist_videos(playlist_url)
    print(f"[cyan]Found {len(urls)} videos[/cyan]")
    for i, url in enumerate(urls, 1):
        out_dir = os.path.join(cfg.run.outputs_dir, rid, "youtube", f"video_{i:03d}")
        out = run_youtube_pipeline(cfg, youtube_url=url, out_dir=out_dir)
        print(f"[green]{i}/{len(urls)}[/green] {url}")
        for k, v in out.items():
            print(f"   - {k}: {v}")


@run_app.command("videos")
def run_videos(
    config: str = typer.Option(..., "--config", "-c"),
    videos_csv: str = typer.Option(..., "--videos-csv", help="CSV with column url or youtube_url (optional: name, video_id)"),
    merge: bool = typer.Option(True, "--merge/--no-merge", help="Write merged_teaching_ready.xlsx and merged_teaching_cards.xlsx with source_video column"),
):
    """Run YouTube pipeline for each row in CSV. Each video gets its own folder; optionally merge outputs with source_video column."""
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
        out = run_youtube_pipeline(cfg, youtube_url=url, out_dir=out_dir)
        video_results.append({"out_dir": out_dir, "source_label": v["source_label"], **out})
        print(f"[green]{i}/{len(videos)}[/green] {v['source_label'][:60]}...")
        for k, path in out.items():
            if isinstance(path, str):
                print(f"   - {k}: {path}")
    if merge and video_results:
        merged = merge_teaching_outputs(video_results, base_dir)
        if merged:
            print("[green]Merged outputs (source_video column):[/green]")
            for k, path in merged.items():
                print(f" - {k}: {path}")


@run_app.command("all")
def run_all(
    config: str = typer.Option(..., "--config", "-c"),
    videos_csv: str = typer.Option(..., "--videos-csv", help="CSV with url or youtube_url column"),
    excel_path: str = typer.Option(None, "--excel-path", help="Optional: run energy pipeline first with this Excel"),
    merge: bool = typer.Option(True, "--merge/--no-merge", help="Merge teaching outputs with source_video"),
):
    """Run energy (if --excel-path) then YouTube pipeline for all videos in CSV. One run_id for everything."""
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
        out = run_youtube_pipeline(cfg, youtube_url=v["url"], out_dir=out_dir)
        video_results.append({"out_dir": out_dir, "source_label": v["source_label"], **out})
        print(f"[green]{i}/{len(videos)}[/green] {v['source_label'][:60]}...")
    if merge and video_results:
        merged = merge_teaching_outputs(video_results, base_dir)
        if merged:
            print("[green]Merged (source_video):[/green]")
            for k, path in merged.items():
                print(f" - {k}: {path}")
