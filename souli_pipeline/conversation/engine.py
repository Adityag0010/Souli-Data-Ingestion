"""
Souli Conversation Engine

State machine that orchestrates:
  1. Intake — empathetic questioning to understand the user's energy state
  2. Diagnosis — maps conversation to an energy node
  3. Intent detection — venting vs solution-seeking
  4. RAG — retrieves relevant YouTube counselor content from Qdrant
  5. Response — Ollama llama3.1 generates a warm, contextual reply
  6. Solution — presents healing framework from gold.xlsx when requested

Usage:
    engine = ConversationEngine.from_config(cfg, gold_path="outputs/.../gold.xlsx")
    while True:
        user_input = input("You: ")
        response = engine.turn(user_input)
        print("Souli:", response)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# Conversation phases
PHASE_INTAKE = "intake"
PHASE_DEEPENING = "deepening"
PHASE_INTENT_CHECK = "intent_check"
PHASE_SOLUTION = "solution"
PHASE_VENTING = "venting"


@dataclass
class ConversationState:
    phase: str = PHASE_INTAKE
    turn_count: int = 0
    # LLM message history for Ollama
    messages: List[Dict[str, str]] = field(default_factory=list)
    # Detected or inferred energy node
    energy_node: Optional[str] = None
    node_confidence: str = "unknown"
    # Probes used per node
    used_probe_indices: Dict[str, List[int]] = field(default_factory=dict)
    # Short-answer follow-up counter
    short_answer_count: int = 0
    # Has intent been resolved?
    intent: Optional[str] = None  # "venting" | "solution"
    # Framework solution loaded?
    framework_loaded: bool = False
    # Concatenated user text for embedding-based diagnosis
    user_text_buffer: str = ""


class ConversationEngine:
    """
    Main engine. Create via from_config() or from_paths().
    Call .turn(user_text) for each user message.
    Call .turn_stream(user_text) for streaming responses.
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
            "blocked_energy",
            "depleted_energy",
            "scattered_energy",
            "outofcontrol_energy",
            "normal_energy",
        ]
        self.framework = framework or {}  # {energy_node: {col: val}}
        self.gold_df = gold_df  # optional DataFrame for embedding-based diagnosis
        self.state = ConversationState()

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg, gold_path: Optional[str] = None, excel_path: Optional[str] = None):
        """Build engine from PipelineConfig + optional gold.xlsx or Excel path."""
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
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Start a fresh conversation."""
        self.state = ConversationState()

    def turn(self, user_text: str) -> str:
        """Process one user message and return Souli's response."""
        result = self._process(user_text, stream=False)
        assert isinstance(result, str)
        return result

    def turn_stream(self, user_text: str) -> Generator[str, None, None]:
        """Process one user message and stream Souli's response token by token."""
        result = self._process(user_text, stream=True)
        if isinstance(result, str):
            yield result
        else:
            yield from result

    def greeting(self) -> str:
        """Return the opening greeting (call before first user message)."""
        from .intake import get_opening
        return get_opening()

    # ------------------------------------------------------------------
    # Internal processing
    # ------------------------------------------------------------------

    def _process(self, user_text: str, stream: bool):
        s = self.state
        s.turn_count += 1
        user_text = (user_text or "").strip()
        s.user_text_buffer += " " + user_text

        # Add user message to history
        s.messages.append({"role": "user", "content": user_text})

        # ---- Phase routing ----

        if s.phase == PHASE_INTAKE:
            response = self._handle_intake(user_text, stream)

        elif s.phase == PHASE_DEEPENING:
            response = self._handle_deepening(user_text, stream)

        elif s.phase == PHASE_INTENT_CHECK:
            response = self._handle_intent_check(user_text, stream)

        elif s.phase == PHASE_VENTING:
            response = self._handle_venting(user_text, stream)

        elif s.phase == PHASE_SOLUTION:
            response = self._handle_solution(user_text, stream)

        else:
            response = self._handle_venting(user_text, stream)

        # Add assistant reply to history (for non-streaming, full string)
        if isinstance(response, str):
            s.messages.append({"role": "assistant", "content": response})

        return response

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def _handle_intake(self, user_text: str, stream: bool):
        s = self.state
        from .intake import is_short_answer, get_short_follow_up

        # After 2+ turns in intake, try to diagnose
        if s.turn_count >= 2:
            self._diagnose(s.user_text_buffer)

        # Check if we should move to intent check
        if s.turn_count >= self.max_intake_turns and s.energy_node:
            s.phase = PHASE_INTENT_CHECK
            return self._handle_intent_check(user_text, stream)

        # If user gave a very short answer, gently prompt for more
        if is_short_answer(user_text) and s.short_answer_count < 2:
            s.short_answer_count += 1
            follow_up = get_short_follow_up(s.short_answer_count)
            # Still generate a warm response + follow-up
            rag = self._rag_retrieve(user_text, s.energy_node)
            reply = self._llm_response(user_text, rag, stream)
            if isinstance(reply, str) and not stream:
                return reply + "\n\n" + follow_up
            return reply

        # Normal intake: respond + possibly ask a probe question
        s.phase = PHASE_DEEPENING
        rag = self._rag_retrieve(user_text, s.energy_node)
        return self._llm_response(user_text, rag, stream)

    def _handle_deepening(self, user_text: str, stream: bool):
        s = self.state
        from .intake import get_probe

        # Re-diagnose with accumulated text
        self._diagnose(s.user_text_buffer)

        if s.turn_count >= self.max_intake_turns:
            s.phase = PHASE_INTENT_CHECK
            return self._handle_intent_check(user_text, stream)

        # Optionally weave in a probe question
        probe_idx_list = s.used_probe_indices.setdefault(s.energy_node or "blocked_energy", [])
        probe = get_probe(s.energy_node or "blocked_energy", probe_idx_list)
        if probe:
            probe_idx_list.append(len(probe_idx_list))

        rag = self._rag_retrieve(user_text, s.energy_node)
        reply = self._llm_response(user_text, rag, stream)

        # Append probe as a gentle follow-up (non-streaming only)
        if probe and isinstance(reply, str) and not stream:
            return reply + "\n\n" + probe
        return reply

    def _handle_intent_check(self, user_text: str, stream: bool):
        s = self.state
        from .intent import detect_intent, INTENT_BRIDGE, nudge_toward_intent

        # Detect if this message already clarifies intent
        intent = detect_intent(
            user_text,
            history_texts=[m["content"] for m in s.messages[-4:] if m["role"] == "user"],
        )

        if intent == "solution":
            s.intent = "solution"
            s.phase = PHASE_SOLUTION
            return self._handle_solution(user_text, stream)

        if intent == "venting":
            s.intent = "venting"
            s.phase = PHASE_VENTING
            return self._handle_venting(user_text, stream)

        # Unclear — ask the bridge question
        s.phase = PHASE_VENTING  # default to venting while we wait
        rag = self._rag_retrieve(user_text, s.energy_node)
        reply = self._llm_response(user_text, rag, stream)
        if isinstance(reply, str) and not stream:
            return reply + "\n\n" + INTENT_BRIDGE
        return reply

    def _handle_venting(self, user_text: str, stream: bool):
        s = self.state
        from .intent import detect_intent

        # Check if user suddenly wants a solution
        intent = detect_intent(user_text)
        if intent == "solution":
            s.intent = "solution"
            s.phase = PHASE_SOLUTION
            return self._handle_solution(user_text, stream)

        rag = self._rag_retrieve(user_text, s.energy_node)
        return self._llm_response(user_text, rag, stream)

    def _handle_solution(self, user_text: str, stream: bool):
        s = self.state
        from .counselor import generate_solution_response, fallback_response
        from .solution import get_solution_for_node

        node = s.energy_node or "blocked_energy"
        sol = get_solution_for_node(node, self.framework)

        if not sol:
            logger.warning("No framework solution for node '%s'", node)
            return (
                fallback_response(node)
                if not stream
                else iter([fallback_response(node)])
            )

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
            from .solution import format_solution_text
            return format_solution_text(node, sol)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _diagnose(self, text: str):
        """Update energy_node based on accumulated user text."""
        s = self.state
        from ..energy.normalize import infer_node
        from ..retrieval.match import diagnose as retrieval_diagnose

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
            else:
                s.energy_node = infer_node(text, "")
                s.node_confidence = "keyword_fallback"
        except Exception as exc:
            logger.warning("Diagnosis error: %s", exc)
            s.energy_node = s.energy_node or "blocked_energy"

    def _rag_retrieve(self, query: str, energy_node: Optional[str]) -> list:
        """Retrieve relevant YouTube teaching chunks from Qdrant."""
        try:
            from ..retrieval.qdrant_store import query_chunks
            return query_chunks(
                user_text=query,
                collection=self.qdrant_collection,
                energy_node=energy_node,
                top_k=self.rag_top_k,
                embedding_model=self.embedding_model,
                host=self.qdrant_host,
                port=self.qdrant_port,
            )
        except Exception as exc:
            logger.debug("Qdrant retrieval failed: %s", exc)
            return []

    def _llm_response(self, user_text: str, rag_chunks: list, stream: bool):
        """Generate counselor response via Ollama."""
        from .counselor import generate_counselor_response, fallback_response

        # Use history minus the just-added user message (it's in rag context)
        history = self.state.messages[:-1]  # exclude current user message

        try:
            return generate_counselor_response(
                history=history,
                user_message=user_text,
                rag_chunks=rag_chunks,
                energy_node=self.state.energy_node,
                ollama_model=self.chat_model,
                ollama_endpoint=self.ollama_endpoint,
                temperature=self.temperature,
                stream=stream,
            )
        except Exception as exc:
            logger.warning("Ollama response failed: %s — using fallback.", exc)
            return fallback_response(self.state.energy_node)

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
