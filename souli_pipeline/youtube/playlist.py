from __future__ import annotations
import subprocess
import json
from typing import List

def list_playlist_videos(playlist_url: str) -> List[str]:
    # yt-dlp JSON lines
    cmd = ["yt-dlp", "--flat-playlist", "-J", playlist_url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    r.check_returncode()
    data = json.loads(r.stdout)
    urls = []
    for e in data.get("entries", []):
        vid = e.get("id")
        if vid:
            urls.append(f"https://www.youtube.com/watch?v={vid}")
    return urls
