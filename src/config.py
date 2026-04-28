from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelCfg:
    name: str
    runtime: str = "cpu"
    streaming: bool = False
    provider: str = "ollama"  # "ollama" | "openai"
    openai_url: Optional[str] = None
    openai_api_key_env: Optional[str] = None  # falls back to Config.openai_api_key_env


@dataclass
class Persona:
    name: str
    rules: List[str] = field(default_factory=list)


@dataclass
class Features:
    stream_thoughts: bool = True
    enable_safety: bool = True
    enable_persistence: bool = True


@dataclass
class Persistence:
    enabled: bool = True
    path: str = "./qdrant_data/"
    collection: str = "chat_turns"
    top_k: int = 3
    vector_size: int = 768


@dataclass
class Config:
    classifier: ModelCfg
    overlay: ModelCfg
    brain: ModelCfg
    prose: ModelCfg
    ollama_host: str
    features: Features
    latency_targets_ms: Dict[str, int]
    persona: Persona
    persistence: Persistence = field(default_factory=Persistence)
    openai_api_key_env: str = "OPENAI_API_KEY"

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        raw: Dict[str, Any] = yaml.safe_load(Path(path).read_text())
        m = raw["models"]
        return cls(
            classifier=ModelCfg(**m["classifier"]),
            overlay=ModelCfg(**m["overlay"]),
            brain=ModelCfg(**m["brain"]),
            prose=ModelCfg(**m["prose"]),
            ollama_host=raw.get("ollama", {}).get("host", "http://localhost:11434"),
            features=Features(**raw.get("features", {})),
            latency_targets_ms=raw.get("latency_targets_ms", {}),
            persona=Persona(**raw["persona"]),
            persistence=Persistence(**raw.get("persistence", {})),
            openai_api_key_env=raw.get("openai", {}).get(
                "api_key_env", "OPENAI_API_KEY"
            ),
        )
