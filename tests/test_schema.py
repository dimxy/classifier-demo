from src.models.classifier import (
    INTENTS,
    INTENSITIES,
    LEADS,
    SHIFTS,
    ModernBertClassifier,
)


def test_classifier_output_schema():
    c = ModernBertClassifier()
    out = c.classify("hi there").to_dict()
    assert set(out.keys()) == {"intent", "emotions", "intensity", "shift", "lead_signal"}
    assert out["intent"] in INTENTS
    assert out["intensity"] in INTENSITIES
    assert out["shift"] in SHIFTS
    assert out["lead_signal"] in LEADS
    assert isinstance(out["emotions"], list)


def test_safety_flags_shape():
    from src.nodes.safety import SafetyNode

    out = SafetyNode()({"prose": "hello"})
    assert "prose" in out
    flags = out["safety"]["flags"]
    assert set(flags.keys()) == {"safe", "issues"}
    assert isinstance(flags["safe"], bool)
    assert isinstance(flags["issues"], list)
