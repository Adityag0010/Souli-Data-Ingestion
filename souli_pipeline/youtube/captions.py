from __future__ import annotations
import os
import subprocess
import webvtt
from typing import List, Dict, Optional

def ts_to_seconds(ts: str) -> float:
    h, m, rest = ts.split(":")
    return int(h)*3600 + int(m)*60 + float(rest)

def download_captions(url: str, langs: str = "en,hi") -> Optional[str]:
    for f in os.listdir():
        if f.endswith(".vtt"):
            try: os.remove(f)
            except: pass

    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-auto-sub",
        "--write-sub",
        "--sub-lang", langs,
        "--sub-format", "vtt",
        url
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    for f in os.listdir():
        if f.endswith(".vtt"):
            return f
    return None

def parse_vtt(vtt_file: str) -> List[Dict]:
    segs = []
    v = webvtt.read(vtt_file)
    for cap in v:
        segs.append({
            "start": ts_to_seconds(cap.start),
            "end": ts_to_seconds(cap.end),
            "text": cap.text.replace("\n", " ").strip()
        })
    return segs
