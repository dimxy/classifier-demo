from src.models import classifier as cm
from src.models.classifier import ModernBertClassifier


def test_classify_zero_shot_returns_none_when_encoder_unavailable(monkeypatch):
    c = ModernBertClassifier()
    monkeypatch.setattr(c, "_try_load", lambda: None)
    c._model = None
    c._tokenizer = None
    assert c.classify_zero_shot("anything") is None


def test_classify_zero_shot_picks_winners_via_cosine(monkeypatch):
    c = ModernBertClassifier()
    c._model = object()
    c._tokenizer = object()
    monkeypatch.setattr(c, "_try_load", lambda: None)

    # Texts that should land on vec [1, 0]; everything else lands on [0, 1].
    # Cosine([1,0], [1,0]) = 1, cosine([1,0], [0,1]) = 0 — so the labels
    # whose prototype description is in this set are guaranteed winners.
    matches = {
        "i missed you",
        cm._INTENT_PROTOTYPES["vulnerable"],
        cm._INTENSITY_PROTOTYPES["moderate"],
        cm._SHIFT_PROTOTYPES["softening"],
        cm._LEAD_PROTOTYPES["neutral"],
        cm._EMOTION_PROTOTYPES["anxious"],
        cm._EMOTION_PROTOTYPES["hopeful"],
    }

    def fake_embed(text):
        return [1.0, 0.0] if text in matches else [0.0, 1.0]

    monkeypatch.setattr(c, "embed", fake_embed)

    result = c.classify_zero_shot("i missed you")
    assert result is not None
    assert result.intent == "vulnerable"
    assert result.intensity == "moderate"
    assert result.shift == "softening"
    assert result.lead_signal == "neutral"
    assert set(result.emotions) == {"anxious", "hopeful"}


def test_classify_zero_shot_caches_prototype_embeddings(monkeypatch):
    c = ModernBertClassifier()
    c._model = object()
    c._tokenizer = object()
    monkeypatch.setattr(c, "_try_load", lambda: None)

    calls = []

    def counting_embed(text):
        calls.append(text)
        return [1.0, 0.0]

    monkeypatch.setattr(c, "embed", counting_embed)

    c.classify_zero_shot("first")
    after_first = len(calls)
    c.classify_zero_shot("second")
    after_second = len(calls)

    # Second call should only embed the new input — prototypes are cached.
    assert after_second - after_first == 1
