"""Tool specifications. A tool with a `describe` function has side effects and must pass the
policy decision point; a tool without one is read-only."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from engine.facts import ActionFacts
from engine.provenance import Trust

Args = dict[str, Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    execute: Callable[[Args], str]
    output_trust: Trust = Trust.SYSTEM
    describe: Callable[[Args], ActionFacts] | None = None

    @property
    def side_effect(self) -> bool:
        return self.describe is not None

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def registry(tools: list[ToolSpec]) -> dict[str, ToolSpec]:
    index: dict[str, ToolSpec] = {}
    for tool in tools:
        if tool.name in index:
            raise ValueError(f"duplicate tool name {tool.name!r}")
        index[tool.name] = tool
    return index
