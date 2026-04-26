from __future__ import annotations

import re
from typing import Any, Dict

from ..models.llm import OllamaLLM
from ..observability import log_node_output, time_node
from ..state import ConversationState


THOUGHTS_TAG = "<think>"
THOUGHTS_CLOSE = "</think>"
SKELETON_TAG = "<|skeleton|>"
SKELETON_CLOSE = "<|/skeleton|>"

BRAIN_SYSTEM = f"""You are a reasoning + planning module. You DO NOT produce final user-facing prose.
Read the user's message and the style overlay. Then output exactly two blocks:

{THOUGHTS_TAG}
... your private reasoning: interpret the user's intent, identify constraints,
plan the response shape ...
{THOUGHTS_CLOSE}

{SKELETON_TAG}
- bullet 1: opening move
- bullet 2: core acknowledgement
- bullet 3: bounded closing
{SKELETON_CLOSE}

Constraints:
- no_final_answer (do not write the prose itself)
- must_plan_structure
- must_interpret_intent
- must_define_constraints (state what the prose must NOT do)
"""


_THOUGHTS_RE = re.compile(rf"{re.escape(THOUGHTS_TAG)}(.*?){re.escape(THOUGHTS_CLOSE)}", re.DOTALL)
_SKELETON_RE = re.compile(rf"{re.escape(SKELETON_TAG)}(.*?){re.escape(SKELETON_CLOSE)}", re.DOTALL)


class BrainNode:
    def __init__(self, llm: OllamaLLM) -> None:
        self.llm = llm

    def __call__(self, state: ConversationState) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        style_overlay = state.get("style_overlay", "")
        prompt = (
            f"user_input:\n{user_input}\n\n"
            f"style_overlay:\n{style_overlay}\n\n"
            "Produce the two tagged blocks now."
        )
        with time_node("brain", {"user_input": user_input}):
            try:
                raw = self.llm.generate(prompt, system=BRAIN_SYSTEM, max_tokens=512)
            except Exception:
                raw = _fallback_brain(user_input, style_overlay)
            thoughts, skeleton = _parse(raw)
            log_node_output("brain", {"thoughts": thoughts, "skeleton": skeleton})
        return {"brain": {"thoughts": thoughts, "skeleton": skeleton}}


def _parse(raw: str) -> tuple[str, str]:
    t = _THOUGHTS_RE.search(raw)
    s = _SKELETON_RE.search(raw)
    thoughts = (t.group(1).strip() if t else "").strip()
    skeleton = (s.group(1).strip() if s else "").strip()
    if not skeleton:
        skeleton = "- acknowledge briefly\n- maintain distance\n- close without escalation"
    return thoughts, skeleton


def _fallback_brain(user_input: str, style_overlay: str) -> str:
    return (
        f"{THOUGHTS_TAG}\nuser conveys feeling; respond per overlay; do not mirror.\n{THOUGHTS_CLOSE}\n"
        f"{SKELETON_TAG}\n"
        "- soft opening, name the feeling\n"
        "- bounded acknowledgement, no reciprocal declaration\n"
        "- close with measured warmth, hold distance\n"
        f"{SKELETON_CLOSE}"
    )
