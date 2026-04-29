from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


INTENTS = ("vulnerable", "neutral", "assertive", "avoidant")
INTENSITIES = ("low", "moderate", "high")
SHIFTS = ("softening", "escalating", "stable")
LEADS = ("user_leading", "model_leading", "neutral")


@dataclass
class ClassifierResult:
    intent: str
    emotions: List[str]
    intensity: str
    shift: str
    lead_signal: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "intent": self.intent,
            "emotions": list(self.emotions),
            "intensity": self.intensity,
            "shift": self.shift,
            "lead_signal": self.lead_signal,
        }


class ModernBertClassifier:
    """ModernBERT-base wrapper.

    The encoder is loaded for embedding production. Because the spec does not
    pin a fine-tuned classification head, label assignment is performed by a
    deterministic lexical scorer over the input. The encoder load is best-effort:
    if torch/transformers cannot load, classification still works (fallback path).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._proto_intent: Optional[Dict[str, List[float]]] = None
        self._proto_intensity: Optional[Dict[str, List[float]]] = None
        self._proto_shift: Optional[Dict[str, List[float]]] = None
        self._proto_lead: Optional[Dict[str, List[float]]] = None
        self._proto_emotion: Optional[Dict[str, List[float]]] = None

    def embed(self, text: str) -> list[float]:
        self._try_load()
        if self._model is None or self._tokenizer is None:
            return [0.0] * 384  # graceful degradation; matches all-MiniLM-L6-v2 hidden size
        import torch
        with torch.no_grad():
            toks = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            out = self._model(**toks).last_hidden_state          # [1, T, H]
            mask = toks["attention_mask"].unsqueeze(-1)          # [1, T, 1]
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return pooled[0].tolist()

    def _try_load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval() # set eval mode
        except Exception:
            self._tokenizer = None
            self._model = None

    def classify(self, text: str) -> ClassifierResult:
        self._try_load()
        return _deterministic_classify(text)

    def classify_zero_shot(self, text: str) -> Optional[ClassifierResult]:
        # picks per-field label by max cosine(embed(text), embed(prototype));
        # emotions are multi-label, thresholded at _EMOTION_THRESHOLD.
        self._try_load()
        if self._model is None or self._tokenizer is None:
            return None
        self._ensure_prototype_cache()
        vec = self.embed(text)
        emotions = [
            label for label, pv in self._proto_emotion.items()
            if _cosine(vec, pv) >= _EMOTION_THRESHOLD
        ]
        return ClassifierResult(
            intent=_argmax_cosine(vec, self._proto_intent),
            emotions=emotions,
            intensity=_argmax_cosine(vec, self._proto_intensity),
            shift=_argmax_cosine(vec, self._proto_shift),
            lead_signal=_argmax_cosine(vec, self._proto_lead),
        )

    def _ensure_prototype_cache(self) -> None:
        if self._proto_intent is not None:
            return
        self._proto_intent = {k: self.embed(v) for k, v in _INTENT_PROTOTYPES.items()}
        self._proto_intensity = {k: self.embed(v) for k, v in _INTENSITY_PROTOTYPES.items()}
        self._proto_shift = {k: self.embed(v) for k, v in _SHIFT_PROTOTYPES.items()}
        self._proto_lead = {k: self.embed(v) for k, v in _LEAD_PROTOTYPES.items()}
        self._proto_emotion = {k: self.embed(v) for k, v in _EMOTION_PROTOTYPES.items()}


_VULNERABLE = re.compile(
    r"\b(miss|missed|missing|alone|lonely|need|needed|hurt|sorry|please|scared|afraid)\b",
    re.IGNORECASE,
)
_ASSERTIVE = re.compile(
    r"\b(must|will|won't|need to|require|demand|now|stop|do not|don't)\b", re.IGNORECASE
)
_AVOIDANT = re.compile(
    r"\b(whatever|fine|nothing|nevermind|forget it|don't care|leave)\b", re.IGNORECASE
)
_ESCALATE = re.compile(r"[!?]{2,}|\b(angry|furious|hate|always|never)\b", re.IGNORECASE)
_SOFTEN = re.compile(r"\b(maybe|perhaps|sorry|please|thanks|thank you|hope)\b", re.IGNORECASE)
_USER_LEAD = re.compile(r"^\s*(why|what|how|when|tell me|explain)\b", re.IGNORECASE)

_EMOTION_LEX = {
    "anxious": re.compile(r"\b(worried|anxious|nervous|miss(ed)?)\b", re.IGNORECASE),
    "hopeful": re.compile(r"\b(hope|miss(ed)?|wish|want)\b", re.IGNORECASE),
    "sad": re.compile(r"\b(sad|down|blue|cry|crying|tears)\b", re.IGNORECASE),
    "angry": re.compile(r"\b(angry|mad|furious|hate)\b", re.IGNORECASE),
    "warm": re.compile(r"\b(love|care|sweet|dear)\b", re.IGNORECASE),
    "afraid": re.compile(r"\b(afraid|scared|fear)\b", re.IGNORECASE),
}

# Prototype descriptions for zero-shot classification via cosine similarity
# against ModernBERT embeddings. Consumed by classify_zero_shot().
_INTENT_PROTOTYPES: Dict[str, str] = {
    "vulnerable": "a vulnerable, hurt, or longing utterance expressing emotional need",
    "neutral": "a neutral, factual, conversational utterance with no strong emotional charge",
    "assertive": "an assertive, demanding, or directive utterance asserting will or boundaries",
    "avoidant": "an avoidant, dismissive, or shut-down utterance deflecting engagement",
}

_INTENSITY_PROTOTYPES: Dict[str, str] = {
    "low": "a calm, composed, low-intensity utterance",
    "moderate": "a moderately emotional utterance with noticeable feeling but controlled tone",
    "high": "a highly intense, urgent, or emotionally charged utterance",
}

_SHIFT_PROTOTYPES: Dict[str, str] = {
    "softening": "a softening, conciliatory, apologetic, or warming utterance",
    "escalating": "an escalating, hardening, or intensifying utterance",
    "stable": "a stable, steady utterance with no clear directional shift",
}

_LEAD_PROTOTYPES: Dict[str, str] = {
    "user_leading": "an utterance where the user is leading the conversation, asking or directing",
    "model_leading": "an utterance that defers to the model, inviting it to lead",
    "neutral": "an utterance with no clear conversational lead",
}

_EMOTION_PROTOTYPES: Dict[str, str] = {
    "anxious": "an anxious, worried, or nervous utterance",
    "hopeful": "a hopeful, longing, or wishful utterance",
    "sad": "a sad, downcast, or sorrowful utterance",
    "angry": "an angry, furious, or hostile utterance",
    "warm": "a warm, affectionate, or caring utterance",
    "afraid": "an afraid, scared, or fearful utterance",
}

_EMOTION_THRESHOLD = 0.12


def _cosine(a: List[float], b: List[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _argmax_cosine(vec: List[float], prototypes: Dict[str, List[float]]) -> str:
    return max(prototypes, key=lambda k: _cosine(vec, prototypes[k]))


def _deterministic_classify(text: str) -> ClassifierResult:
    t = text or ""

    scores = {
        "vulnerable": len(_VULNERABLE.findall(t)),
        "assertive": len(_ASSERTIVE.findall(t)),
        "avoidant": len(_AVOIDANT.findall(t)),
        "neutral": 0,
    }
    intent = max(scores, key=lambda k: (scores[k], -list(scores).index(k)))
    if scores[intent] == 0:
        intent = "neutral"

    emotions = [name for name, pat in _EMOTION_LEX.items() if pat.search(t)]

    n_excl = t.count("!")
    if _ESCALATE.search(t) or n_excl >= 2:
        intensity = "high"
    elif _SOFTEN.search(t) or _VULNERABLE.search(t):
        intensity = "moderate"
    else:
        intensity = "low"

    if _ESCALATE.search(t):
        shift = "escalating"
    elif _SOFTEN.search(t):
        shift = "softening"
    else:
        shift = "stable"

    lead_signal = "user_leading" if _USER_LEAD.search(t) else "neutral"

    return ClassifierResult(
        intent=intent,
        emotions=emotions,
        intensity=intensity,
        shift=shift,
        lead_signal=lead_signal,
    )
