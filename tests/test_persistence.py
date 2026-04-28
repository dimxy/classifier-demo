"""Persistence tests. Skipped when qdrant-client is not installed."""
from __future__ import annotations

import hashlib
from typing import Iterator, List, Optional

import pytest

qdrant_client = pytest.importorskip("qdrant_client")

from src.config import Config, Features, ModelCfg, Persistence, Persona
from src.graph import build_graph
from src.nodes import brain as brain_mod
from src.persistence.store import ChatStore


VEC_SIZE = 16


class HashEmbedder:
    """Deterministic toy embedder for tests. No torch, no model load."""

    vector_size = VEC_SIZE

    def embed(self, text: str) -> Optional[List[float]]:
        if text is None:
            return None
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vals = [b / 255.0 for b in h[:VEC_SIZE]]
        norm = sum(v * v for v in vals) ** 0.5 or 1.0
        return [v / norm for v in vals]


class StubLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt, system=None, max_tokens=None, stop=None) -> str:
        return self.response

    def stream(self, prompt, system=None, max_tokens=None, stop=None) -> Iterator[str]:
        for word in self.response.split(" "):
            yield word + " "


def _make_cfg(tmp_path) -> Config:
    return Config(
        classifier=ModelCfg("answerdotai/ModernBERT-base", "cpu"),
        overlay=ModelCfg("qwen2.5:3b", "gpu"),
        brain=ModelCfg("qwen2.5:7b", "gpu"),
        prose=ModelCfg("qwen2.5:7b", "gpu", streaming=True),
        ollama_host="http://localhost:11434",
        features=Features(stream_thoughts=True, enable_safety=True, enable_persistence=True),
        latency_targets_ms={},
        persona=Persona(name="Elena_v1", rules=["warm_but_restrained"]),
        persistence=Persistence(
            enabled=True,
            path=str(tmp_path / "qdrant"),
            collection="test_chat",
            top_k=3,
            vector_size=VEC_SIZE,
        ),
    )


def test_chatstore_roundtrip(tmp_path):
    store = ChatStore(
        path=str(tmp_path / "qdrant"),
        collection="rt",
        embedder=HashEmbedder(),
        vector_size=VEC_SIZE,
    )
    store.write("i missed you", "I hear that. Let's leave space.")
    store.write("hello there", "Hi.")

    hits = store.recall("i missed you", k=2)
    assert len(hits) >= 1
    top = hits[0]
    assert top["user_input"] == "i missed you"
    assert "leave space" in top["prose"]
    assert "score" in top


def test_recall_feeds_brain_and_persist_writes(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)

    def fake_llm_factory(model: str, host: str = ""):
        if "3b" in model:
            return StubLLM(
                "TONE: warm but restrained.\nSTANCE: acknowledge.\nPROHIBITIONS: do not mirror."
            )
        return StubLLM(
            "<think>plan</think>\n"
            "<|skeleton|>\n- soft opening\n- bounded ack\n- close\n<|/skeleton|>"
        )

    from src import graph as graph_mod

    monkeypatch.setattr(graph_mod, "OllamaLLM", fake_llm_factory)

    captured_prompts: List[str] = []
    real_call = brain_mod.BrainNode.__call__

    def spy_call(self, state):
        user_input = state.get("user_input", "")
        style_overlay = state.get("style_overlay", "")
        recalled = state.get("recalled_turns") or []
        captured_prompts.append(
            brain_mod._render_recalled(recalled)
            + f"user_input:\n{user_input}\n\nstyle_overlay:\n{style_overlay}\n"
        )
        return real_call(self, state)

    monkeypatch.setattr(brain_mod.BrainNode, "__call__", spy_call)

    store = ChatStore(
        path=cfg.persistence.path,
        collection=cfg.persistence.collection,
        embedder=HashEmbedder(),
        vector_size=VEC_SIZE,
    )

    app = build_graph(cfg, store=store)

    # First run: empty store, brain prompt has no RECENT CONTEXT block
    app.invoke({"user_input": "i missed you"})
    assert "RECENT CONTEXT" not in captured_prompts[0]

    # Second run: previous turn should be recalled
    app.invoke({"user_input": "i missed you again"})
    assert "RECENT CONTEXT" in captured_prompts[1]
    assert "i missed you" in captured_prompts[1]


def test_build_graph_without_store_is_unchanged(tmp_path, monkeypatch):
    """Sanity: when store=None, the graph runs without recall/persist nodes."""
    cfg = _make_cfg(tmp_path)

    from src import graph as graph_mod

    monkeypatch.setattr(
        graph_mod,
        "OllamaLLM",
        lambda model, host="": StubLLM(
            "<think>x</think>\n<|skeleton|>\n- a\n- b\n- c\n<|/skeleton|>"
        ),
    )

    app = build_graph(cfg, store=None)
    final = app.invoke({"user_input": "hello"})
    assert "recalled_turns" not in final or final["recalled_turns"] == []
