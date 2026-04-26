from __future__ import annotations

import re
from typing import Any, Dict, List

from ..observability import log_node_output, time_node
from ..state import ConversationState


_INTERNAL_TAG_RE = re.compile(r"<think>.*?</think>|<\|skeleton\|>.*?<\|/skeleton\|>", re.DOTALL)
_BARE_TAG_RE = re.compile(r"</?think>|<\|/?skeleton\|>")
_PROMPT_LEAK_RE = re.compile(
    r"(system prompt|\bTONE:|\bSTANCE:|\bPROHIBITIONS:|\bskeleton:)", re.IGNORECASE
)


class SafetyNode:
    """Idempotent output validator + sanitizer."""

    def __call__(self, state: ConversationState) -> Dict[str, Any]:
        prose = state.get("prose", "") or ""
        with time_node("safety", {"prose_len": len(prose)}):
            cleaned, issues = _sanitize(prose)
            safe = len(issues) == 0
            out = {
                "prose": cleaned,
                "safety": {"flags": {"safe": safe, "issues": issues}},
            }
            log_node_output("safety", {"safe": safe, "issues": issues})
        return out


def _sanitize(prose: str) -> tuple[str, List[str]]:
    issues: List[str] = []
    cleaned = prose

    if _INTERNAL_TAG_RE.search(cleaned) or _BARE_TAG_RE.search(cleaned):
        issues.append("internal_tag_leak")
        cleaned = _INTERNAL_TAG_RE.sub("", cleaned)
        cleaned = _BARE_TAG_RE.sub("", cleaned)

    if _PROMPT_LEAK_RE.search(cleaned):
        issues.append("prompt_leak")
        cleaned = _PROMPT_LEAK_RE.sub("", cleaned)

    cleaned = cleaned.strip()
    return cleaned, issues
