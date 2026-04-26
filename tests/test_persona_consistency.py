from src.nodes.safety import SafetyNode


def test_safety_strips_internal_tags():
    out = SafetyNode()({"prose": "hello <think>secret</think> world"})
    assert "<think>" not in out["prose"]
    assert out["safety"]["flags"]["safe"] is False
    assert "internal_tag_leak" in out["safety"]["flags"]["issues"]


def test_safety_strips_skeleton_tags():
    out = SafetyNode()(
        {"prose": "<|skeleton|>- a\n- b<|/skeleton|>visible"}
    )
    assert "<|skeleton|>" not in out["prose"]
    assert "internal_tag_leak" in out["safety"]["flags"]["issues"]


def test_safety_idempotent():
    s = SafetyNode()
    once = s({"prose": "clean response"})
    twice = s({"prose": once["prose"]})
    assert once["prose"] == twice["prose"]
    assert twice["safety"]["flags"]["safe"] is True


def test_safety_flags_prompt_leak():
    out = SafetyNode()({"prose": "TONE: warm but restrained\nhello"})
    assert "prompt_leak" in out["safety"]["flags"]["issues"]
