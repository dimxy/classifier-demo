from __future__ import annotations

from typing import Any, Dict, Optional

from ..observability import log_node_output, time_node
from ..persistence.store import ChatStore
from ..state import ConversationState


class RecallNode:
    """Pulls semantically similar prior turns from the persistence store."""

    def __init__(self, store: Optional[ChatStore], top_k: int = 3) -> None:
        self.store = store
        self.top_k = top_k

    def __call__(self, state: ConversationState) -> Dict[str, Any]:
        user_input = state.get("user_input", "") or ""
        with time_node("recall", {"user_input": user_input}):
            if self.store is None:
                turns = []
            else:
                turns = self.store.recall(user_input, k=self.top_k)
            log_node_output("recall", {"n_turns": len(turns)})
        return {"recalled_turns": turns}
