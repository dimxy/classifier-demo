from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from langgraph.graph import END, StateGraph

from .config import Config, ModelCfg
from .models.classifier import ModernBertClassifier
from .models.llm import LLMProvider, OllamaLLM
from .models.openai_llm import OpenAILLM
from .nodes.brain import BrainNode
from .nodes.classify import ClassifyNode
from .nodes.overlay import OverlayNode
from .nodes.persist import PersistNode
from .nodes.prose import ProseNode
from .nodes.recall import RecallNode
from .nodes.safety import SafetyNode
from .observability import logger
from .persistence.embedder import ModernBertEmbedder
from .persistence.store import ChatStore
from .state import ConversationState


def _build_llm(node_cfg: ModelCfg, config: Config) -> LLMProvider:
    """Pick the LLM provider for a single node based on its config.

    The Ollama branch resolves ``OllamaLLM`` from this module's globals so
    that test fixtures can monkeypatch it.
    """
    provider = (node_cfg.provider or "ollama").lower()
    if provider == "openai":
        if not node_cfg.openai_url:
            raise ValueError(
                f"openai provider selected for model {node_cfg.name!r} "
                "but no openai_url is configured for this node"
            )
        return OpenAILLM(
            model=node_cfg.name,
            base_url=node_cfg.openai_url,
            api_key_env=node_cfg.openai_api_key_env or config.openai_api_key_env,
        )
    if provider == "ollama":
        return OllamaLLM(node_cfg.name, host=config.ollama_host)
    raise ValueError(
        f"unknown LLM provider {node_cfg.provider!r} for model {node_cfg.name!r}; "
        "expected 'ollama' or 'openai'"
    )


def build_graph(
    config: Config,
    on_prose_chunk: Optional[Callable[[str], None]] = None,
    store: Optional[ChatStore] = None,
    classifier: Optional[ModernBertClassifier] = None,
):
    classifier = classifier or ModernBertClassifier(config.classifier.name)
    overlay_llm = _build_llm(config.overlay, config)
    brain_llm = _build_llm(config.brain, config)
    prose_llm = _build_llm(config.prose, config)

    classify_node = ClassifyNode(classifier)
    overlay_node = OverlayNode(overlay_llm, config.persona)
    brain_node = BrainNode(brain_llm)
    prose_node = ProseNode(prose_llm, on_chunk=on_prose_chunk)
    safety_node = SafetyNode()

    g = StateGraph(ConversationState)
    g.add_node("classify", classify_node)
    g.add_node("overlay", overlay_node)
    g.add_node("brain", brain_node)
    g.add_node("prose", prose_node)

    persistence_active = store is not None

    if persistence_active:
        recall_node = RecallNode(store, top_k=config.persistence.top_k)
        persist_node = PersistNode(store)
        g.add_node("recall", recall_node)
        g.add_node("persist", persist_node)

    g.set_entry_point("classify")

    if persistence_active:
        g.add_edge("classify", "overlay")
        g.add_edge("overlay", "recall")
        g.add_edge("recall", "brain")
    else:
        g.add_edge("classify", "overlay")
        g.add_edge("overlay", "brain")

    g.add_edge("brain", "prose")

    if config.features.enable_safety:
        g.add_node("safety", safety_node)
        g.add_edge("prose", "safety")
        if persistence_active:
            g.add_edge("safety", "persist")
            g.add_edge("persist", END)
        else:
            g.add_edge("safety", END)
    else:
        if persistence_active:
            g.add_edge("prose", "persist")
            g.add_edge("persist", END)
        else:
            g.add_edge("prose", END)

    return g.compile()


def _maybe_build_store(
    config: Config, classifier: ModernBertClassifier
) -> Optional[ChatStore]:
    if not config.features.enable_persistence or not config.persistence.enabled:
        return None
    embedder = ModernBertEmbedder(classifier, vector_size=config.persistence.vector_size)
    if not embedder.available:
        logger.warning("persistence: embedder unavailable; running stateless")
        return None
    return ChatStore(
        path=config.persistence.path,
        collection=config.persistence.collection,
        embedder=embedder,
        vector_size=config.persistence.vector_size,
    )


def run(
    user_input: str,
    config: Optional[Config] = None,
    on_prose_chunk: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    cfg = config or Config.load()
    classifier = ModernBertClassifier(cfg.classifier.name)
    store = _maybe_build_store(cfg, classifier)
    app = build_graph(cfg, on_prose_chunk=on_prose_chunk, store=store, classifier=classifier)
    initial: ConversationState = {"user_input": user_input}
    return app.invoke(initial)
