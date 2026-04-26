from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class ModelCfg:
    name: str
    runtime: str = "cpu"
    streaming: bool = False


@dataclass
class Persona:
    name: str
    rules: List[str] = field(default_factory=list)


@dataclass
class Features:
    stream_thoughts: bool = True
    enable_safety: bool = True


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
        )
