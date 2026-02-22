from __future__ import annotations
from typing import Dict
import requests

TEACH_KEYS = [
 "Concept/Principle",
 "Core explanation",
 "When it applies",
 "Concrete example (1-2 lines)",
 "Mapped energy_node"
]

class HttpJsonLLM:
    def __init__(self, endpoint: str, timeout_s: int = 60):
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    def extract_teaching_card(self, transcript: str) -> Dict[str, str]:
        payload = {"transcript": transcript, "keys": TEACH_KEYS}
        r = requests.post(self.endpoint, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        # Return only allowed keys; missing => ""
        return {k: str(data.get(k, "") or "") for k in TEACH_KEYS}
