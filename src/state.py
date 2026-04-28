from __future__ import annotations

from typing import TypedDict, List, Literal, Optional


Intent = Literal["vulnerable", "neutral", "assertive", "avoidant"]
Intensity = Literal["low", "moderate", "high"]
Shift = Literal["softening", "escalating", "stable"]
LeadSignal = Literal["user_leading", "model_leading", "neutral"]


class BrainState(TypedDict, total=False):
    thoughts: str
    skeleton: str


class SafetyFlags(TypedDict, total=False):
    safe: bool
    issues: List[str]


class SafetyState(TypedDict, total=False):
    flags: SafetyFlags


class RecalledTurn(TypedDict, total=False):
    user_input: str
    prose: str
    score: float


class ConversationState(TypedDict, total=False):
    user_input: str

    intent: Intent
    emotions: List[str]
    intensity: Intensity
    shift: Shift
    lead_signal: LeadSignal

    style_overlay: str

    recalled_turns: List[RecalledTurn]

    brain: BrainState

    prose: str

    safety: SafetyState


CLASSIFIER_FALLBACK = {
    "intent": "neutral",
    "emotions": [],
    "intensity": "moderate",
    "shift": "stable",
    "lead_signal": "neutral",
}
