from src.models.classifier import ModernBertClassifier


def test_classifier_determinism_repeats():
    c = ModernBertClassifier()
    a = c.classify("i missed you").to_dict()
    b = c.classify("i missed you").to_dict()
    assert a == b


def test_example_case_i_missed_you():
    c = ModernBertClassifier()
    out = c.classify("i missed you").to_dict()
    assert out["intent"] == "vulnerable"
    assert "anxious" in out["emotions"]
    assert "hopeful" in out["emotions"]
    assert out["intensity"] == "moderate"
