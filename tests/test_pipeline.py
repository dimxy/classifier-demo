"""Full-pipeline integration test using stubbed LLMs (no Ollama required)."""
from __future__ import annotations

from typing import Iterator, List, Optional

import pytest

from src.config import Config, Features, ModelCfg, Persona
from src.graph import build_graph


class StubLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt, system=None, max_tokens=None, stop=None) -> str:
        return self.response

    def stream(self, prompt, system=None, max_tokens=None, stop=None) -> Iterator[str]:
        for word in self.response.split(" "):
            yield word + " "


@pytest.fixture
def cfg() -> Config:
    return Config(
        classifier=ModelCfg("sentence-transformers/all-MiniLM-L6-v2", "cpu"),
        overlay=ModelCfg("qwen2.5:3b", "gpu"),
        brain=ModelCfg("qwen2.5:7b", "gpu"),
        prose=ModelCfg("qwen2.5:7b", "gpu", streaming=True),
        ollama_host="http://localhost:11434",
        features=Features(stream_thoughts=True, enable_safety=True),
        latency_targets_ms={"classify": 10, "overlay": 150, "brain": 4000, "prose_first_token": 500},
        persona=Persona(name="Elena_v1", rules=["warm_but_restrained", "maintain_distance"]),
    )


def test_full_pipeline_execution(cfg, monkeypatch):
    from src import graph as graph_mod

    def fake_llm_factory(model: str, host: str = ""):
        if "3b" in model:
            return StubLLM(
                "TONE: warm but restrained.\nSTANCE: acknowledge.\nPROHIBITIONS: do not mirror."
            )
        return StubLLM(
            "<think>user is vulnerable; do not mirror</think>\n"
            "<|skeleton|>\n- soft opening\n- bounded acknowledgement\n- measured close\n<|/skeleton|>"
        )

    monkeypatch.setattr(graph_mod, "OllamaLLM", fake_llm_factory)

    chunks: List[str] = []
    app = build_graph(cfg, on_prose_chunk=chunks.append)
    final = app.invoke({"user_input": "i missed you"})

    assert final["intent"] == "vulnerable"
    assert isinstance(final["style_overlay"], str) and final["style_overlay"]
    assert "skeleton" in final["brain"]
    assert isinstance(final["prose"], str) and final["prose"]
    assert "<think>" not in final["prose"]
    assert "<|skeleton|>" not in final["prose"]
    flags = final["safety"]["flags"]
    assert flags["safe"] is True


def test_streaming_behavior(cfg, monkeypatch):
    from src import graph as graph_mod

    monkeypatch.setattr(
        graph_mod,
        "OllamaLLM",
        lambda model, host="": StubLLM("hello there friend"),
    )

    received: List[str] = []
    app = build_graph(cfg, on_prose_chunk=received.append)
    app.invoke({"user_input": "hello"})

    assert len(received) >= 2  # got multiple chunks
