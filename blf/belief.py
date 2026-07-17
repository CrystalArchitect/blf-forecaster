"""The linguistic belief state (paper Sec. C.1).

The core innovation of BLF: instead of appending all retrieved evidence to an
ever-growing raw context, the agent maintains a compact, semi-structured belief
that the LLM *rewrites* at every step. The belief state consists of:

    - a probability estimate p in [0, 1] for the binary outcome
    - a confidence level (low / medium / high)
    - key evidence for and against the outcome (natural-language summaries)
    - open questions the agent plans to investigate next

Crucially (Sec. C.2), the belief update is not a separate call: it is emitted as
an ``updated_belief`` argument *inside* the tool call the agent makes, so a
single LLM forward pass produces both the next action and the new belief. The
``update_reasoning`` slot forces the model to articulate why its belief changed,
which the paper reports "encourages coherent probabilistic updating".
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BeliefState:
    p: float = 0.5
    confidence: str = "low"
    update_reasoning: str = ""
    key_evidence_for: list[str] = field(default_factory=list)
    key_evidence_against: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)

    @classmethod
    def initial(cls) -> "BeliefState":
        """b_0: no information -> p = 0.5 (Algorithm 1, line 1)."""
        return cls(p=0.5, confidence="low", update_reasoning="Initial (no evidence).")

    @classmethod
    def from_tool_arg(cls, arg: dict) -> "BeliefState":
        """Parse the ``updated_belief`` object carried inside a tool call."""
        return cls(
            p=float(arg.get("p", 0.5)),
            confidence=str(arg.get("confidence", "low")),
            update_reasoning=str(arg.get("update_reasoning", "")),
            key_evidence_for=list(arg.get("key_evidence_for", []) or []),
            key_evidence_against=list(arg.get("key_evidence_against", []) or []),
            open_questions=list(arg.get("open_questions", []) or []),
        )

    def summary(self) -> str:
        return (
            f"p={self.p:.3f} ({self.confidence}) — {self.update_reasoning}"
        )


# JSON-schema fragment describing ``updated_belief``. Every tool the agent can
# call embeds this as a required argument, so each generation yields an action
# *and* a refreshed belief in one shot (Sec. C.2).
UPDATED_BELIEF_SCHEMA: dict = {
    "type": "object",
    "description": (
        "Your rewritten belief AFTER accounting for the evidence you have so "
        "far. Fill this in on every tool call."
    ),
    "properties": {
        "p": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Current probability the event resolves YES.",
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "update_reasoning": {
            "type": "string",
            "description": "Why your belief changed since the previous step.",
        },
        "key_evidence_for": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concise evidence pointing toward YES.",
        },
        "key_evidence_against": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concise evidence pointing toward NO.",
        },
        "open_questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "What you still need to resolve.",
        },
    },
    "required": ["p", "confidence", "update_reasoning"],
}
