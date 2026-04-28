from __future__ import annotations

from typing import Any, Dict, Optional

from ..observability import log_node_output, time_node
from ..persistence.store import ChatStore
from ..state import ConversationState


class PersistNode:
    """Writes the sanitized turn to the persistence store. Idempotent at the
    pipeline level (one write per pipeline run); never raises."""

    def __init__(self, store: Optional[ChatStore]) -> None:
        self.store = store

    def __call__(self, state: ConversationState) -> Dict[str, Any]:
        user_input = state.get("user_input", "") or ""
        prose = state.get("prose", "") or ""
        with time_node("persist", {"user_input": user_input, "prose_len": len(prose)}):
            wrote = False
            if self.store is not None and user_input and prose:
                self.store.write(user_input, prose)
                wrote = True
            log_node_output("persist", {"wrote": wrote})
        return {}
