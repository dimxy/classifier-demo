from __future__ import annotations

from typing import List, Optional

from ..models.classifier import ModernBertClassifier
from ..observability import logger


class ModernBertEmbedder:
    """Mean-pooled ModernBERT embedder. Reuses the classifier's encoder load.

    Returns None from `embed` when torch/transformers cannot load the model;
    callers must treat this as a graceful no-op signal.
    """

    def __init__(self, classifier: ModernBertClassifier, vector_size: int = 768) -> None:
        self.classifier = classifier
        self.vector_size = vector_size

    @property
    def available(self) -> bool:
        self.classifier._try_load()
        return self.classifier._model is not None and self.classifier._tokenizer is not None

    def embed(self, text: str) -> Optional[List[float]]:
        if not text:
            text = ""
        self.classifier._try_load()
        model = self.classifier._model
        tokenizer = self.classifier._tokenizer
        if model is None or tokenizer is None:
            logger.warning("embedder: encoder unavailable, returning None")
            return None
        try:
            import torch

            with torch.no_grad():
                enc = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512, padding=True
                )
                out = model(**enc)
                hidden = out.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1.0)
                pooled = (summed / counts).squeeze(0)
                return pooled.tolist()
        except Exception as exc:
            logger.error("embedder.embed failed: %s", exc)
            return None
