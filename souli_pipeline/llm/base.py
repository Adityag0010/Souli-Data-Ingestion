from __future__ import annotations
from typing import Protocol, Dict

class LLMAdapter(Protocol):
    def extract_teaching_card(self, transcript: str) -> Dict[str, str]:
        ...
