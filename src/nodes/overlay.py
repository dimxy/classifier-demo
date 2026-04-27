from __future__ import annotations

from typing import Any, Dict

from ..config import Persona
from ..models.llm import OllamaLLM
from ..observability import log_node_output, time_node, logger
from ..state import ConversationState


OVERLAY_SYSTEM = """You are a persona style compiler. Given a classifier reading and a persona, emit a short
style overlay (<= 120 tokens) that another model will follow when generating prose.

The overlay MUST contain three labeled sections:
- TONE: a one-line tone definition.
- STANCE: the emotional stance to take.
- PROHIBITIONS: things to avoid.

Apply these persona rules: sharpened_warm, restrained_affection, avoid_oversharing.
Do NOT produce any user-visible prose. Output only the overlay block.
"""


class OverlayNode:
    def __init__(self, llm: OllamaLLM, persona: Persona) -> None:
        self.llm = llm
        self.persona = persona

    def __call__(self, state: ConversationState) -> Dict[str, Any]:
        classifier = {
            "intent": state.get("intent"),
            "emotions": state.get("emotions", []),
            "intensity": state.get("intensity"),
            "shift": state.get("shift"),
            "lead_signal": state.get("lead_signal"),
        }
        prompt = (
            f"classifier:\n{classifier}\n\n"
            f"persona:\n  name: {self.persona.name}\n  rules: {self.persona.rules}\n\n"
            "Emit the overlay now."
        )
        with time_node("overlay", {"classifier": classifier, "persona": self.persona.name}):
            try:
                overlay = self.llm.generate(prompt, system=OVERLAY_SYSTEM, max_tokens=120)
            except Exception as exc:
                logger.error(f"overlay llm.generate: {str(exc)}")
                overlay = _fallback_overlay(self.persona, classifier, str(exc))
            overlay = overlay.strip()
            log_node_output("overlay", {"style_overlay": overlay})
        return {"style_overlay": overlay}


def _fallback_overlay(persona: Persona, classifier: Dict[str, Any], reason: str) -> str:
    return (
        "TONE: warm but restrained, measured cadence.\n"
        "STANCE: acknowledge feeling without mirroring; maintain gentle distance.\n"
        "PROHIBITIONS: do not over-share, do not declare reciprocal emotion, "
        f"respect persona rules {persona.rules}."
    )
