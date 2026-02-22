from __future__ import annotations
import os
import subprocess
from typing import List, Dict
from faster_whisper import WhisperModel

AUDIO_FILE = "yt_audio.m4a"

def whisper_transcribe(url: str, model_name: str = "medium") -> List[Dict]:
    if os.path.exists(AUDIO_FILE):
        os.remove(AUDIO_FILE)

    subprocess.run(["yt-dlp", "-f", "bestaudio", "-o", AUDIO_FILE, url], check=True)

    # device selection inside faster-whisper
    model = WhisperModel(model_name, device="auto", compute_type="auto")
    seg_iter, _info = model.transcribe(AUDIO_FILE, vad_filter=True)
    seg_list = list(seg_iter)

    segs = []
    for s in seg_list:
        tx = (s.text or "").strip()
        if not tx:
            continue
        segs.append({"start": float(s.start), "end": float(s.end), "text": tx})
    return segs
