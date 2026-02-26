"""
Ollama LLM adapter — works with any model served by Ollama (llama3.1, qwen2.5, etc.)
All inference is local. No data leaves the machine.
"""
from __future__ import annotations
import json
import requests
from typing import Dict, Generator, List, Optional


class OllamaLLM:
    """
    Thin wrapper around Ollama's /api/chat and /api/generate endpoints.

    Usage:
        llm = OllamaLLM(model="llama3.1")
        reply = llm.chat([{"role": "user", "content": "Hello"}])

        # streaming
        for chunk in llm.chat_stream([{"role": "user", "content": "Hello"}]):
            print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        model: str = "llama3.1",
        endpoint: str = "http://localhost:11434",
        timeout_s: int = 120,
        temperature: float = 0.7,
        num_ctx: int = 4096,
    ):
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.num_ctx = num_ctx

    # ------------------------------------------------------------------
    # Chat (multi-turn)
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a multi-turn chat and return the full response string.
        messages: [{"role": "user"|"assistant"|"system", "content": "..."}]
        """
        msgs = self._prepend_system(messages, system)
        payload = {
            "model": self.model,
            "messages": msgs,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_ctx": self.num_ctx,
            },
        }
        r = requests.post(
            f"{self.endpoint}/api/chat",
            json=payload,
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """
        Stream a multi-turn chat. Yields text chunks as they arrive.
        """
        msgs = self._prepend_system(messages, system)
        payload = {
            "model": self.model,
            "messages": msgs,
            "stream": True,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_ctx": self.num_ctx,
            },
        }
        with requests.post(
            f"{self.endpoint}/api/chat",
            json=payload,
            stream=True,
            timeout=self.timeout_s,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break

    # ------------------------------------------------------------------
    # Generate (single prompt, no history)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        format: Optional[str] = None,
    ) -> str:
        """
        Single-turn generation. Use format="json" to force JSON output.
        """
        payload: Dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_ctx": self.num_ctx,
            },
        }
        if system:
            payload["system"] = system
        if format:
            payload["format"] = format
        r = requests.post(
            f"{self.endpoint}/api/generate",
            json=payload,
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return r.json()["response"].strip()

    # ------------------------------------------------------------------
    # Teaching card extraction (matches existing LLMAdapter protocol)
    # ------------------------------------------------------------------

    def extract_teaching_card(self, transcript: str) -> Dict[str, str]:
        """
        Extract a structured teaching card from a YouTube transcript chunk.
        Compatible with the existing LLMAdapter protocol.
        """
        keys = [
            "Concept/Principle",
            "Core explanation",
            "When it applies",
            "Concrete example (1-2 lines)",
            "Mapped energy_node",
        ]
        nodes = (
            "blocked_energy, depleted_energy, scattered_energy, "
            "outofcontrol_energy, normal_energy"
        )
        prompt = (
            f"You are a wellness content analyst. Extract a teaching card from the "
            f"transcript below. Return ONLY valid JSON with exactly these keys:\n"
            f"{json.dumps(keys)}\n\n"
            f"For 'Mapped energy_node', pick ONE from: {nodes}\n\n"
            f"Transcript:\n{transcript}\n\n"
            f"JSON:"
        )
        raw = self.generate(prompt, temperature=0.1, format="json")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Attempt to extract JSON from noisy output
            start = raw.find("{")
            end = raw.rfind("}") + 1
            data = json.loads(raw[start:end]) if start != -1 and end > start else {}
        return {k: str(data.get(k, "") or "").strip() for k in keys}

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            r = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Return list of locally available model names."""
        try:
            r = requests.get(f"{self.endpoint}/api/tags", timeout=10)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepend_system(
        messages: List[Dict[str, str]], system: Optional[str]
    ) -> List[Dict[str, str]]:
        if not system:
            return messages
        # Don't prepend if first message is already system
        if messages and messages[0].get("role") == "system":
            return messages
        return [{"role": "system", "content": system}] + list(messages)
