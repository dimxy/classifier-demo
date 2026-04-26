# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Source of truth

[REQS.yaml](REQS.yaml) is the authoritative specification for the application. Keep code aligned with it: state-schema field names, node I/O shapes, graph edges, persona scoping, and constraint lists are all driven by that file. When the spec and code diverge, ask which one should change before editing.

## Commands

Run from repo root using the existing venv at `.venv/`.

- Run tests: `.venv/bin/python -m pytest -q`
- Run one test: `.venv/bin/python -m pytest tests/test_pipeline.py::test_full_pipeline_execution -q`
- Run the pipeline end-to-end (requires Ollama running locally with `qwen2.5:3b` and `qwen2.5:7b` pulled): `.venv/bin/python main.py -i "i missed you"`
- Install full deps (LLM + ModernBERT runtime): `.venv/bin/pip install -r requirements.txt`. The minimal test set only needs `langgraph pyyaml pytest`.

## Architecture

A 5-stage LangGraph pipeline that **separates classification, planning, and expression**. Each stage is a stateless node that reads from and writes back to a single `ConversationState` TypedDict.

```
classify → overlay → brain → prose → safety
```

- **classify** ([src/nodes/classify.py](src/nodes/classify.py)) — deterministic intent/emotion classifier. Constraint: `must_not_fail`; on any exception it returns `CLASSIFIER_FALLBACK` from [src/state.py](src/state.py). The current head ([src/models/classifier.py](src/models/classifier.py)) is a lexical scorer because REQS.yaml does not pin a fine-tuning recipe; the ModernBERT encoder is loaded best-effort for future use. **If you replace the head, do not change the public `classify()` return shape** — every downstream node depends on it.
- **overlay** ([src/nodes/overlay.py](src/nodes/overlay.py)) — compiles the classifier reading + persona into a short style block (TONE / STANCE / PROHIBITIONS). Persona is **static config** loaded from `config.yaml`, not part of dynamic state. Capped at 120 tokens.
- **brain** ([src/nodes/brain.py](src/nodes/brain.py)) — produces two tagged blocks: `<think>...</think>` and `<|skeleton|>...<|/skeleton|>`. **No final answer.** The skeleton is what `prose` follows; the thoughts are optionally streamed on a premium-only channel.
- **prose** ([src/nodes/prose.py](src/nodes/prose.py)) — streaming-only generator. Chunks flow through the `on_prose_chunk` callback wired in [src/graph.py](src/graph.py). Internal tags are stripped before the chunk leaves the node.
- **safety** ([src/nodes/safety.py](src/nodes/safety.py)) — idempotent sanitizer. Strips any leaked internal tags or prompt-section markers and writes `safety.flags.{safe, issues}`. The `flags` shape **must** match `state_schema.safety.flags` in REQS.yaml — when one changes, change the other.

## State conventions

[src/state.py](src/state.py) is the single source of truth for runtime types. Two things are easy to get wrong:

1. **Nested keys**: `brain.thoughts`, `brain.skeleton`, and `safety.flags.*` are nested dicts in state. Returning `{"skeleton": ...}` from a node will silently land in the wrong place.
2. **Partial updates**: nodes return only the keys they own. LangGraph merges these into state — never return the full state dict.

## Models and runtime

Three model surfaces, all swappable:

- `ModernBertClassifier` in [src/models/classifier.py](src/models/classifier.py) — CPU-bound, target latency 10ms.
- `OllamaLLM` in [src/models/llm.py](src/models/llm.py) — single wrapper used by `overlay` (qwen2.5:3b), `brain` (qwen2.5:7b), and `prose` (qwen2.5:7b streaming). Hits a local Ollama server.
- The graph builder ([src/graph.py](src/graph.py)) instantiates these by name. Tests stub them by monkeypatching `graph_mod.OllamaLLM`, so don't import `OllamaLLM` directly into `graph.py`'s callsites in a way that bypasses the module attribute.

## Testing

Integration tests in [tests/test_pipeline.py](tests/test_pipeline.py) run the full LangGraph pipeline with stubbed LLMs — no Ollama required. The stub pattern (a `StubLLM` class with `.generate()` and `.stream()`) is the right way to add new pipeline tests; do not reach for the real Ollama client.

The example case in REQS.yaml (`"i missed you"` → `intent=vulnerable, emotions=[anxious, hopeful], intensity=moderate`) is asserted in [tests/test_classifier_determinism.py](tests/test_classifier_determinism.py). Treat that case as a regression anchor — if you change the classifier head, this test is the contract.
