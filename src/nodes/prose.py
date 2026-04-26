from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional

from ..models.llm import OllamaLLM
from ..observability import log_node_output, time_node
from ..state import ConversationState


PROSE_SYSTEM = """You are the final response generator. Produce the user-visible reply only.

Strict constraints:
- must_follow_skeleton: every bullet in the skeleton must be reflected, in order
- must_respect_overlay: tone, stance, and prohibitions in the overlay are binding
- must_not_expose_internal_tags: never emit <think>, <|skeleton|>, or any meta tags
- maintain_persona_consistency

Output the prose only. No preamble, no labels, no tags.
"""


class ProseNode:
    """Streaming prose generator. Emits chunks via on_chunk if provided."""

    def __init__(self, llm: OllamaLLM, on_chunk: Optional[Callable[[str], None]] = None) -> None:
        self.llm = llm
        self.on_chunk = on_chunk

    def __call__(self, state: ConversationState) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        style_overlay = state.get("style_overlay", "")
        skeleton = state.get("brain", {}).get("skeleton", "")
        prompt = (
            f"style_overlay:\n{style_overlay}\n\n"
            f"skeleton:\n{skeleton}\n\n"
            f"user_input:\n{user_input}\n\n"
            "Write the reply now."
        )
        with time_node("prose", {"user_input": user_input}):
            try:
                pieces = []
                for chunk in self.llm.stream(prompt, system=PROSE_SYSTEM, max_tokens=512):
                    pieces.append(chunk)
                    if self.on_chunk is not None:
                        self.on_chunk(chunk)
                prose = "".join(pieces).strip()
            except Exception:
                prose = _fallback_prose(skeleton)
                if self.on_chunk is not None:
                    self.on_chunk(prose)
            prose = _strip_tags(prose)
            log_node_output("prose", {"prose": prose})
        return {"prose": prose}


_BAD_TAGS = ("<think>", "</think>", "<|skeleton|>", "<|/skeleton|>")


def _strip_tags(text: str) -> str:
    out = text
    for tag in _BAD_TAGS:
        out = out.replace(tag, "")
    return out.strip()


def _fallback_prose(skeleton: str) -> str:
    return "I hear you. That lands. Let's leave a little space for it and come back when we can."
