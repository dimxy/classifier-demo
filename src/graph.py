from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from langgraph.graph import END, StateGraph

from .config import Config
from .models.classifier import ModernBertClassifier
from .models.llm import OllamaLLM
from .nodes.brain import BrainNode
from .nodes.classify import ClassifyNode
from .nodes.overlay import OverlayNode
from .nodes.prose import ProseNode
from .nodes.safety import SafetyNode
from .state import ConversationState


def build_graph(
    config: Config,
    on_prose_chunk: Optional[Callable[[str], None]] = None,
):
    classifier = ModernBertClassifier(config.classifier.name)
    overlay_llm = OllamaLLM(config.overlay.name, host=config.ollama_host)
    brain_llm = OllamaLLM(config.brain.name, host=config.ollama_host)
    prose_llm = OllamaLLM(config.prose.name, host=config.ollama_host)

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
    g.add_node("safety", safety_node)

    g.set_entry_point("classify")
    g.add_edge("classify", "overlay")
    g.add_edge("overlay", "brain")
    g.add_edge("brain", "prose")
    if config.features.enable_safety:
        g.add_edge("prose", "safety")
        g.add_edge("safety", END)
    else:
        g.add_edge("prose", END)

    return g.compile()


def run(
    user_input: str,
    config: Optional[Config] = None,
    on_prose_chunk: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    cfg = config or Config.load()
    app = build_graph(cfg, on_prose_chunk=on_prose_chunk)
    initial: ConversationState = {"user_input": user_input}
    return app.invoke(initial)
