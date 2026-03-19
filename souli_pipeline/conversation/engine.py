"""
Souli Conversation Engine  — DEBUG-INSTRUMENTED VERSION
Drop this in as souli_pipeline/conversation/engine.py on your dev branch.

Every turn pushes a structured debug event to self._debug_events so the
Streamlit dev UI can inspect exactly what happened: phase transitions,
diagnosis result, full Qdrant results, LLM call details.

No changes to the public API — .turn(), .turn_stream(), .greeting() all
work identically.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conversation phases
# ---------------------------------------------------------------------------
PHASE_GREETING     = "greeting"
PHASE_INTAKE       = "intake"
PHASE_SHARING      = "sharing"
PHASE_DEEPENING    = "deepening"
PHASE_SUMMARY      = "summary"
PHASE_INTENT_CHECK = "intent_check"
PHASE_SOLUTION     = "solution"
PHASE_VENTING      = "venting"


@dataclass
class ConversationState:
    phase: str = PHASE_GREETING
    turn_count: int = 0
    user_name: Optional[str] = None
    messages: List[Dict[str, str]] = field(default_factory=list)
    energy_node: Optional[str] = None
    node_confidence: str = "unknown"
    used_probe_indices: Dict[str, List[int]] = field(default_factory=dict)
    used_sharing_probe_indices: Dict[str, List[int]] = field(default_factory=dict)
    short_answer_count: int = 0
    intent: Optional[str] = None
    framework_loaded: bool = False
    user_text_buffer: str = ""
    summary_attempted: bool = False
    summary_confirmed: bool = False
    rich_opening: bool = False


# ---------------------------------------------------------------------------
# Debug event structure (one per turn)
# ---------------------------------------------------------------------------
def _empty_debug(turn: int, phase_before: str, user_text: str) -> Dict[str, Any]:
    return {
        "turn": turn,
        "timestamp": time.time(),
        "phase_before": phase_before,
        "phase_after": phase_before,          # updated at end of _process()
        "user_text": user_text,
        # ── Diagnosis ──────────────────────────────────────────────────
        "diagnosis": {
            "ran": False,
            "input_snippet": "",
            "energy_node": None,
            "confidence": "unknown",
            "matched_problem": None,
            "similarity": None,
            "method": "not_run",
        },
        # ── RAG / Qdrant ────────────────────────────────────────────────
        "rag": {
            "ran": False,
            "query": "",
            "energy_node_filter": None,
            "top_k_requested": 0,
            "results": [],          # list of dicts from Qdrant
            "error": None,
        },
        # ── LLM call ────────────────────────────────────────────────────
        "llm": {
            "ran": False,
            "model": "",
            "endpoint": "",
            "phase": "",
            "history_length": 0,
            "rag_chunks_injected": 0,
            "system_prompt": "",
            "used_fallback": False,
            "fallback_reason": "",
            "latency_ms": None,
        },
        # ── Full state after turn ────────────────────────────────────────
        "state_after": {},
    }


class ConversationEngine:
    """
    Main engine — identical public API to original, with debug instrumentation.
    Access debug events via self._debug_events (list, one dict per turn).
    """

    def __init__(
        self,
        chat_model: str = "llama3.1",
        tagger_model: str = "qwen2.5:1.5b",
        ollama_endpoint: str = "http://localhost:11434",
        rag_top_k: int = 3,
        max_intake_turns: int = 4,
        temperature: float = 0.75,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_collection: str = "souli_chunks",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        nodes_allowed: Optional[List[str]] = None,
        framework: Optional[Dict] = None,
        gold_df=None,
    ):
        self.chat_model = chat_model
        self.tagger_model = tagger_model
        self.ollama_endpoint = ollama_endpoint
        self.rag_top_k = rag_top_k
        self.max_intake_turns = max_intake_turns
        self.temperature = temperature
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.qdrant_collection = qdrant_collection
        self.embedding_model = embedding_model
        self.nodes_allowed = nodes_allowed or [
            "blocked_energy", "depleted_energy", "scattered_energy",
            "outofcontrol_energy", "normal_energy",
        ]
        self.framework = framework or {}
        self.gold_df = gold_df
        self.state = ConversationState()

        # ── Debug instrumentation ──────────────────────────────────────
        self._debug_events: List[Dict[str, Any]] = []   # full history
        self._current_debug: Dict[str, Any] = {}        # in-flight turn

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg, gold_path: Optional[str] = None, excel_path: Optional[str] = None):
        from .solution import load_framework_from_gold, load_framework_from_excel
        from ..retrieval.match import load_gold

        c = cfg.conversation
        r = cfg.retrieval
        e = cfg.energy

        framework = {}
        gold_df = None

        if gold_path:
            try:
                framework = load_framework_from_gold(gold_path)
                gold_df = load_gold(gold_path, e.nodes_allowed)
                logger.info("Loaded framework from gold.xlsx (%d nodes)", len(framework))
            except Exception as exc:
                logger.warning("Could not load gold.xlsx: %s", exc)

        if not framework and excel_path:
            try:
                framework = load_framework_from_excel(excel_path)
                logger.info("Loaded framework from Excel (%d nodes)", len(framework))
            except Exception as exc:
                logger.warning("Could not load Excel framework: %s", exc)

        return cls(
            chat_model=c.chat_model,
            tagger_model=c.tagger_model,
            ollama_endpoint=c.ollama_endpoint,
            rag_top_k=c.rag_top_k,
            max_intake_turns=c.max_intake_turns,
            temperature=c.temperature,
            qdrant_host=r.qdrant_host,
            qdrant_port=r.qdrant_port,
            qdrant_collection=r.qdrant_collection,
            embedding_model=r.embedding_model or "sentence-transformers/all-MiniLM-L6-v2",
            nodes_allowed=e.nodes_allowed,
            framework=framework,
            gold_df=gold_df,
        )

    # ------------------------------------------------------------------
    # Public API (unchanged from original)
    # ------------------------------------------------------------------

    def reset(self):
        self.state = ConversationState()
        self._debug_events.clear()
        self._current_debug = {}

    def turn(self, user_text: str) -> str:
        result = self._process(user_text, stream=False)
        assert isinstance(result, str)
        return result

    def turn_stream(self, user_text: str) -> Generator[str, None, None]:
        result = self._process(user_text, stream=True)
        if isinstance(result, str):
            yield result
        else:
            yield from result

    def greeting(self) -> str:
        from .intake import get_greeting
        return get_greeting()

    # ------------------------------------------------------------------
    # Internal processing — instrumented
    # ------------------------------------------------------------------

    def _process(self, user_text: str, stream: bool):
        s = self.state
        s.turn_count += 1
        user_text = (user_text or "").strip()
        s.user_text_buffer += " " + user_text
        s.messages.append({"role": "user", "content": user_text})

        # ── Init debug event for this turn ────────────────────────────
        phase_before = s.phase
        self._current_debug = _empty_debug(s.turn_count, phase_before, user_text)

        # ── Phase routing ─────────────────────────────────────────────
        if s.phase == PHASE_GREETING:
            response = self._handle_greeting(user_text, stream)
        elif s.phase == PHASE_INTAKE:
            response = self._handle_intake(user_text, stream)
        elif s.phase == PHASE_SHARING:
            response = self._handle_sharing(user_text, stream)
        elif s.phase == PHASE_DEEPENING:
            response = self._handle_deepening(user_text, stream)
        elif s.phase == PHASE_SUMMARY:
            response = self._handle_summary_response(user_text, stream)
        elif s.phase == PHASE_INTENT_CHECK:
            response = self._handle_intent_check(user_text, stream)
        elif s.phase == PHASE_VENTING:
            response = self._handle_venting(user_text, stream)
        elif s.phase == PHASE_SOLUTION:
            response = self._handle_solution(user_text, stream)
        else:
            response = self._handle_venting(user_text, stream)

        # ── Finalise debug event ──────────────────────────────────────
        self._current_debug["phase_after"] = s.phase
        self._current_debug["state_after"] = {
            "phase": s.phase,
            "energy_node": s.energy_node,
            "node_confidence": s.node_confidence,
            "intent": s.intent,
            "turn_count": s.turn_count,
            "user_name": s.user_name,
            "summary_attempted": s.summary_attempted,
            "summary_confirmed": s.summary_confirmed,
            "rich_opening": s.rich_opening,
            "short_answer_count": s.short_answer_count,
            "messages_count": len(s.messages),
            "user_text_buffer_words": len(s.user_text_buffer.split()),
            "framework_nodes_loaded": list(self.framework.keys()),
            "gold_df_rows": len(self.gold_df) if self.gold_df is not None else 0,
        }
        self._debug_events.append(dict(self._current_debug))

        if isinstance(response, str):
            s.messages.append({"role": "assistant", "content": response})

        return response

    # ------------------------------------------------------------------
    # Phase handlers (identical logic, unchanged)
    # ------------------------------------------------------------------

    def _handle_greeting(self, user_text: str, stream: bool):
        s = self.state
        from .intake import is_rich_message

        name = _extract_name(user_text)
        s.user_name = name
        words = user_text.lower().split()
        shared_feelings = any(w in _NOT_NAMES for w in words)

        if is_rich_message(user_text):
            s.rich_opening = True
            self._diagnose(user_text)
            s.phase = PHASE_SHARING
            name_part = f"{name}, " if name else ""
            return (
                f"{name_part}I hear you, and I'm really glad you felt comfortable sharing that. "
                f"That takes courage. Tell me more — what's been the hardest part of all this for you?"
            )

        s.phase = PHASE_INTAKE
        if name and not shared_feelings:
            return f"Lovely to meet you, {name}. How are you feeling today?"
        elif name and shared_feelings:
            return (
                f"I hear you, {name}. I'm glad you're here. "
                f"Tell me more — what's been going on?"
            )
        else:
            return (
                "I hear you. I'm glad you reached out. "
                "Tell me more — what's been going on for you?"
            )

    def _handle_intake(self, user_text: str, stream: bool):
        s = self.state
        from .intake import is_short_answer, get_short_follow_up, is_rich_message

        if s.turn_count >= 2:
            self._diagnose(s.user_text_buffer)

        if is_rich_message(user_text) and s.turn_count >= 2:
            s.phase = PHASE_SHARING
            return self._handle_sharing(user_text, stream)

        if s.turn_count >= self.max_intake_turns and s.energy_node and not s.summary_attempted:
            return self._trigger_summary(stream)

        if is_short_answer(user_text) and s.short_answer_count < 2:
            s.short_answer_count += 1
            follow_up = get_short_follow_up(s.short_answer_count)
            rag = self._rag_retrieve(user_text, s.energy_node)
            reply = self._llm_response(user_text, rag, stream)
            if isinstance(reply, str) and not stream:
                return reply + "\n\n" + follow_up
            return reply

        s.phase = PHASE_DEEPENING
        rag = self._rag_retrieve(user_text, s.energy_node)
        return self._llm_response(user_text, rag, stream)

    def _handle_sharing(self, user_text: str, stream: bool):
        s = self.state
        self._diagnose(s.user_text_buffer)

        from .intent import detect_intent
        intent = detect_intent(user_text)
        if intent == "solution":
            s.intent = "solution"
            s.phase = PHASE_SOLUTION
            return self._handle_solution(user_text, stream)

        sharing_turns = self._count_turns_in_phase(PHASE_SHARING)
        if sharing_turns >= 2 and s.energy_node and not s.summary_attempted:
            return self._trigger_summary(stream)

        from .intake import get_sharing_probe
        probe_idx_list = s.used_sharing_probe_indices.setdefault(
            s.energy_node or "blocked_energy", []
        )
        probe = get_sharing_probe(s.energy_node or "blocked_energy", probe_idx_list)
        if probe:
            probe_idx_list.append(len(probe_idx_list))

        rag = self._rag_retrieve(user_text, s.energy_node)
        reply = self._llm_response(user_text, rag, stream)
        if probe and isinstance(reply, str) and not stream:
            return reply + "\n\n" + probe
        return reply

    def _handle_deepening(self, user_text: str, stream: bool):
        s = self.state
        from .intake import get_probe, is_rich_message

        self._diagnose(s.user_text_buffer)

        if is_rich_message(user_text):
            s.phase = PHASE_SHARING
            return self._handle_sharing(user_text, stream)

        if s.turn_count >= self.max_intake_turns and s.energy_node and not s.summary_attempted:
            return self._trigger_summary(stream)

        probe_idx_list = s.used_probe_indices.setdefault(s.energy_node or "blocked_energy", [])
        probe = get_probe(s.energy_node or "blocked_energy", probe_idx_list)
        if probe:
            probe_idx_list.append(len(probe_idx_list))

        rag = self._rag_retrieve(user_text, s.energy_node)
        reply = self._llm_response(user_text, rag, stream)
        if probe and isinstance(reply, str) and not stream:
            return reply + "\n\n" + probe
        return reply

    def _trigger_summary(self, stream: bool) -> str:
        s = self.state
        s.summary_attempted = True
        s.phase = PHASE_SUMMARY

        from .summarizer import generate_summary
        return generate_summary(
            user_text_buffer=s.user_text_buffer.strip(),
            energy_node=s.energy_node,
            user_name=s.user_name,
            ollama_model=self.chat_model,
            ollama_endpoint=self.ollama_endpoint,
            temperature=self.temperature,
        )

    def _handle_summary_response(self, user_text: str, stream: bool):
        s = self.state
        from .intent import detect_summary_response, detect_intent

        intent = detect_intent(user_text)
        if intent == "solution":
            s.intent = "solution"
            s.summary_confirmed = True
            s.phase = PHASE_SOLUTION
            return self._handle_solution(user_text, stream)

        response_type = detect_summary_response(user_text)

        if response_type == "confirmed":
            s.summary_confirmed = True
            s.phase = PHASE_INTENT_CHECK
            return self._handle_intent_check(user_text, stream)
        elif response_type == "wants_more":
            s.phase = PHASE_SHARING
            rag = self._rag_retrieve(user_text, s.energy_node)
            return self._llm_response(user_text, rag, stream)
        elif response_type == "correction":
            s.phase = PHASE_INTAKE
            s.summary_attempted = False
            name_part = f"{s.user_name}, " if s.user_name else ""
            return (
                f"{name_part}I appreciate you correcting me — I want to make sure I really understand. "
                f"What felt off? What's the part that's weighing on you most right now?"
            )
        else:
            s.phase = PHASE_SHARING
            rag = self._rag_retrieve(user_text, s.energy_node)
            return self._llm_response(user_text, rag, stream)

    def _handle_intent_check(self, user_text: str, stream: bool):
        s = self.state
        from .intent import detect_intent, INTENT_BRIDGE

        intent = detect_intent(
            user_text,
            history_texts=[m["content"] for m in s.messages[-4:] if m["role"] == "user"],
        )

        if intent == "solution":
            s.intent = "solution"
            s.phase = PHASE_SOLUTION
            return self._handle_solution(user_text, stream)

        if intent in ("venting", "sharing"):
            s.intent = "venting"
            s.phase = PHASE_VENTING
            return self._handle_venting(user_text, stream)

        s.phase = PHASE_VENTING
        rag = self._rag_retrieve(user_text, s.energy_node)
        reply = self._llm_response(user_text, rag, stream)
        if isinstance(reply, str) and not stream:
            return reply + "\n\n" + INTENT_BRIDGE
        return reply

    def _handle_venting(self, user_text: str, stream: bool):
        s = self.state
        from .intent import detect_intent, INTENT_BRIDGE

        intent = detect_intent(user_text)
        if intent == "solution":
            s.intent = "solution"
            s.phase = PHASE_SOLUTION
            return self._handle_solution(user_text, stream)

        _short = len(user_text.strip().split()) <= 3
        if _short:
            s.short_answer_count += 1
        else:
            s.short_answer_count = 0

        if s.short_answer_count >= 3:
            s.short_answer_count = 0
            s.phase = PHASE_INTENT_CHECK
            name_part = f"{s.user_name}, " if s.user_name else ""
            return (
                f"{name_part}I hear you. "
                "Would you like me to suggest something that might actually help, "
                "or do you just want to keep talking?"
            )

        rag = self._rag_retrieve(user_text, s.energy_node)
        return self._llm_response(user_text, rag, stream)

    def _handle_solution(self, user_text: str, stream: bool):
        s = self.state
        from .counselor import generate_solution_response
        from .solution import get_solution_for_node, format_solution_text

        node = s.energy_node or "blocked_energy"
        sol = get_solution_for_node(node, self.framework)

        if not sol:
            logger.warning("No framework solution for node '%s' — using LLM only", node)
            rag = self._rag_retrieve(user_text, node)
            return self._llm_response(user_text, rag, stream)

        user_context = s.user_text_buffer.strip()
        try:
            return generate_solution_response(
                energy_node=node,
                framework_solution=sol,
                user_context=user_context,
                ollama_model=self.chat_model,
                ollama_endpoint=self.ollama_endpoint,
                temperature=self.temperature,
                stream=stream,
            )
        except Exception as exc:
            logger.warning("Ollama solution generation failed: %s", exc)
            return format_solution_text(node, sol)

    # ------------------------------------------------------------------
    # Core helpers — instrumented versions
    # ------------------------------------------------------------------

    def _diagnose(self, text: str):
        """Update energy_node based on accumulated user text. Captures debug info."""
        s = self.state
        from ..energy.normalize import infer_node
        from ..retrieval.match import diagnose as retrieval_diagnose

        d = self._current_debug["diagnosis"]
        d["ran"] = True
        d["input_snippet"] = text.strip()[-500:]  # last 500 chars of buffer

        try:
            if self.gold_df is not None and not self.gold_df.empty:
                result = retrieval_diagnose(
                    text,
                    self.gold_df,
                    self.nodes_allowed,
                    embedding_model=self.embedding_model,
                )
                s.energy_node = result.get("energy_node") or "blocked_energy"
                s.node_confidence = result.get("confidence", "keyword_fallback")

                d["energy_node"] = s.energy_node
                d["confidence"] = s.node_confidence
                d["matched_problem"] = result.get("matched_problem")
                d["similarity"] = result.get("similarity")
                d["method"] = (
                    "embedding_match" if result.get("confidence") == "embedding_match"
                    else "keyword_fallback"
                )
                d["framework_row_preview"] = {
                    k: v[:120] if isinstance(v, str) else v
                    for k, v in (result.get("framework_row") or {}).items()
                }
            else:
                node = infer_node(text, "")
                s.energy_node = node
                s.node_confidence = "keyword_fallback"

                d["energy_node"] = node
                d["confidence"] = "keyword_fallback"
                d["method"] = "keyword_fallback_no_gold"
                d["matched_problem"] = None
                d["similarity"] = None

        except Exception as exc:
            logger.warning("Diagnosis error: %s", exc)
            s.energy_node = s.energy_node or "blocked_energy"
            d["energy_node"] = s.energy_node
            d["error"] = str(exc)

    def _rag_retrieve(self, query: str, energy_node: Optional[str]) -> list:
        """Retrieve Qdrant chunks. Captures full results for debug."""
        r = self._current_debug["rag"]
        r["ran"] = True
        r["query"] = query
        r["energy_node_filter"] = energy_node
        r["top_k_requested"] = self.rag_top_k

        try:
            from ..retrieval.qdrant_store import query_chunks
            t0 = time.time()
            results = query_chunks(
                user_text=query,
                collection=self.qdrant_collection,
                energy_node=energy_node,
                top_k=self.rag_top_k,
                embedding_model=self.embedding_model,
                host=self.qdrant_host,
                port=self.qdrant_port,
            )
            r["latency_ms"] = round((time.time() - t0) * 1000, 1)
            r["results"] = results   # full list — text, score, energy_node, source_video
            r["results_count"] = len(results)
            return results
        except Exception as exc:
            logger.debug("Qdrant retrieval failed: %s", exc)
            r["error"] = str(exc)
            r["results"] = []
            r["results_count"] = 0
            return []

    def _llm_response(self, user_text: str, rag_chunks: list, stream: bool):
        """Generate counselor response. Captures model call details for debug."""
        from .counselor import generate_counselor_response, fallback_response, _build_counselor_system

        history = self.state.messages[:-1][-8:]

        asked_topics = []
        _topic_words = ["sleep", "eat", "food", "relax", "break", "support", "colleague",
                        "manager", "family", "friend", "work", "office", "exercise", "hobby"]
        for m in self.state.messages:
            if m["role"] == "assistant":
                low = m["content"].lower()
                for t in _topic_words:
                    if t in low and t not in asked_topics:
                        asked_topics.append(t)

        # Build the system prompt for debug capture
        system_prompt = _build_counselor_system(
            user_name=self.state.user_name,
            phase=self.state.phase,
            asked_topics=asked_topics,
        )

        llm_d = self._current_debug["llm"]
        llm_d["ran"] = True
        llm_d["model"] = self.chat_model
        llm_d["endpoint"] = self.ollama_endpoint
        llm_d["phase"] = self.state.phase
        llm_d["history_length"] = len(history)
        llm_d["rag_chunks_injected"] = len(rag_chunks)
        llm_d["system_prompt"] = system_prompt
        llm_d["asked_topics"] = asked_topics
        llm_d["used_fallback"] = False

        t0 = time.time()
        try:
            result = generate_counselor_response(
                history=history,
                user_message=user_text,
                rag_chunks=rag_chunks,
                energy_node=self.state.energy_node,
                ollama_model=self.chat_model,
                ollama_endpoint=self.ollama_endpoint,
                temperature=self.temperature,
                stream=stream,
                user_name=self.state.user_name,
                phase=self.state.phase,
                asked_topics=asked_topics,
            )
            if not stream:
                llm_d["latency_ms"] = round((time.time() - t0) * 1000, 1)
            return result
        except Exception as exc:
            llm_d["used_fallback"] = True
            llm_d["fallback_reason"] = str(exc)
            llm_d["latency_ms"] = round((time.time() - t0) * 1000, 1)
            logger.warning("Ollama response failed: %s — using fallback.", exc)
            return fallback_response(self.state.energy_node, user_text)

    def _count_turns_in_phase(self, phase: str) -> int:
        count = 0
        for msg in reversed(self.state.messages):
            if msg["role"] == "assistant":
                count += 1
            if count >= 4:
                break
        sharing_count = sum(
            1 for m in self.state.messages
            if m["role"] == "assistant" and self.state.phase == phase
        )
        return sharing_count

    # ------------------------------------------------------------------
    # Convenience info
    # ------------------------------------------------------------------

    @property
    def diagnosis_summary(self) -> Dict:
        s = self.state
        return {
            "energy_node": s.energy_node,
            "confidence": s.node_confidence,
            "intent": s.intent,
            "phase": s.phase,
            "turn_count": s.turn_count,
        }

    @property
    def latest_debug(self) -> Optional[Dict]:
        """Convenience: the debug event from the most recent turn."""
        return self._debug_events[-1] if self._debug_events else None


# ---------------------------------------------------------------------------
# Module-level helpers (unchanged)
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "hello", "hi", "hey", "yes", "no", "ok", "okay", "sure", "thanks",
    "hlo", "hii", "helo", "yrr", "yaar", "bhai", "dost", "sir", "mam",
    "na", "ha", "haan", "nahi", "hn", "hmm", "hm", "um", "uh",
}

_NOT_NAMES = {
    "very", "so", "really", "quite", "just", "feeling", "not", "too", "a", "an", "the",
    "good", "bad", "okay", "fine", "great", "terrible", "horrible", "well", "better",
    "sad", "happy", "angry", "tired", "exhausted", "stressed", "anxious", "worried",
    "scared", "nervous", "depressed", "confused", "lost", "desperate", "frustrated",
    "overwhelmed", "lonely", "alone", "hurt", "broken", "stuck", "empty", "numb",
    "excited", "grateful", "blessed", "unsure", "unsettled", "restless",
    "here", "there", "new", "back", "trying", "going", "looking", "feeling", "thinking",
    "also", "still", "already", "always", "never", "sometimes", "often", "just",
    "bit", "little", "kind", "sort", "totally", "completely", "absolutely",
}


def _extract_name(text: str) -> Optional[str]:
    text = (text or "").strip()
    for pattern in [
        r"(?:my name is|name(?:'?s)? is|call me|they call me)\s+([A-Za-z]+)",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            candidate = m.group(1).lower()
            if candidate not in _NOT_NAMES and candidate not in _STOP_WORDS:
                return m.group(1).capitalize()

    for pattern in [r"(?:i'?m|i am)\s+([A-Za-z]+)"]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            candidate = m.group(1).lower()
            if candidate not in _NOT_NAMES and candidate not in _STOP_WORDS:
                return m.group(1).capitalize()

    words = [w for w in text.split() if w.isalpha()]
    meaningful = [w for w in words if w.lower() not in _STOP_WORDS and w.lower() not in _NOT_NAMES]
    if meaningful and len(words) <= 2:
        return meaningful[0].capitalize()

    return None