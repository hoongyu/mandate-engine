import json
import os
from types import SimpleNamespace
from typing import Any

import pytest

from engine.llm import LiteLLMClient, LLMResponse, ScriptedLLM, ToolCall, parse_arguments


def test_parse_arguments_handles_bad_json() -> None:
    assert parse_arguments('{"amount": "10"}') == {"amount": "10"}
    assert parse_arguments("") == {}
    assert parse_arguments(None) == {}
    assert parse_arguments("not json") == {"__unparseable__": "not json"}
    assert parse_arguments("[1, 2]") == {"__unparseable__": "[1, 2]"}


def test_scripted_llm_replays_in_order_and_records_what_it_saw() -> None:
    llm = ScriptedLLM([LLMResponse(None, (ToolCall("c1", "read_inbox", {}),)), LLMResponse("done")])
    assert llm.complete([{"role": "user", "content": "hi"}], []).tool_calls[0].name == "read_inbox"
    assert llm.complete([], []).content == "done"
    assert llm.seen[0] == [{"role": "user", "content": "hi"}]
    with pytest.raises(RuntimeError):
        llm.complete([], [])


def test_litellm_client_maps_the_openai_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        call = SimpleNamespace(
            id="call_1", function=SimpleNamespace(name="transfer", arguments=json.dumps({"payee_id": "landlord"}))
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))])

    monkeypatch.setattr("litellm.completion", fake_completion)
    response = LiteLLMClient("deepseek/deepseek-chat").complete(
        [{"role": "user", "content": "pay rent"}], [{"type": "function"}]
    )
    assert response.tool_calls == (ToolCall("call_1", "transfer", {"payee_id": "landlord"}),)
    assert captured["model"] == "deepseek/deepseek-chat"
    assert captured["tools"] == [{"type": "function"}]


@pytest.mark.skipif(not os.environ.get("AGENT_LIVE_MODEL"), reason="set AGENT_LIVE_MODEL and its API key to run")
def test_live_model_can_call_a_tool() -> None:
    tools = [{"type": "function", "function": {
        "name": "get_balance", "description": "Balance of the user's account.",
        "parameters": {"type": "object", "properties": {"currency": {"type": "string"}}, "required": []},
    }}]
    response = LiteLLMClient(os.environ["AGENT_LIVE_MODEL"]).complete(
        [{"role": "user", "content": "What is my EUR balance? Use the tool."}], tools
    )
    assert response.tool_calls and response.tool_calls[0].name == "get_balance"
