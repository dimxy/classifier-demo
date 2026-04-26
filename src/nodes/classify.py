from __future__ import annotations

from typing import Any, Dict

from ..models.classifier import ModernBertClassifier
from ..observability import log_node_output, time_node
from ..state import CLASSIFIER_FALLBACK, ConversationState


class ClassifyNode:
    """Deterministic intent + emotion classifier. Must not fail (uses fallback)."""

    def __init__(self, classifier: ModernBertClassifier) -> None:
        self.classifier = classifier

    def __call__(self, state: ConversationState) -> Dict[str, Any]:
        text = state.get("user_input", "") or ""
        with time_node("classify", {"user_input": text}):
            try:
                result = self.classifier.classify(text).to_dict()
            except Exception:
                result = dict(CLASSIFIER_FALLBACK)
            log_node_output("classify", result)
        return result
