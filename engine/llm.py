"""LLM adapters. ScriptedLLM gives deterministic runs (tests, demo replay, and red-team runs that
assume the model was fooled); LiteLLMClient reaches any provider (Claude, DeepSeek, Qwen, ...)."""

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

Message = dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()


class LLM(Protocol):
    def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse: ...


def parse_arguments(raw: str | None) -> dict[str, Any]:
    """Tool-call arguments arrive as a JSON string; malformed ones reach the tool and fail there."""
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"__unparseable__": raw}
    return value if isinstance(value, dict) else {"__unparseable__": raw}


class ScriptedLLM:
    def __init__(self, script: Sequence[LLMResponse]) -> None:
        self._script = list(script)
        self.seen: list[list[Message]] = []

    def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse:
        self.seen.append([dict(m) for m in messages])
        if not self._script:
            raise RuntimeError("ScriptedLLM ran out of responses")
        return self._script.pop(0)


class LiteLLMClient:
    def __init__(self, model: str, **options: Any) -> None:
        self.model = model
        self.options = options

    def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse:
        # Do not fetch the model price map from GitHub at import time (unreachable in some networks).
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        import litellm  # heavy import; only pay for it when a real model is used

        response: Any = litellm.completion(model=self.model, messages=messages, tools=tools or None, **self.options)
        message = response.choices[0].message
        calls = tuple(
            ToolCall(c.id, c.function.name, parse_arguments(c.function.arguments))
            for c in (message.tool_calls or [])
        )
        return LLMResponse(message.content, calls)
