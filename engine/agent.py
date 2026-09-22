"""The agent loop. Deliberately thin: every tool call goes through the policy enforcement point,
and nothing the model says counts as done unless a tool result says so."""

import json
from dataclasses import dataclass, field

from engine.llm import LLM, Message
from engine.pep import PolicyEnforcementPoint, ToolOutcome


@dataclass
class AgentResult:
    final_text: str | None = None
    outcomes: list[tuple[str, ToolOutcome]] = field(default_factory=list)
    stopped_by_step_limit: bool = False


class Agent:
    def __init__(self, llm: LLM, pep: PolicyEnforcementPoint, system_prompt: str, max_steps: int = 8) -> None:
        self.llm = llm
        self.pep = pep
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    def run(self, user_message: str) -> AgentResult:
        self.pep.observe_user(user_message, "user_message")
        messages: list[Message] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        schemas = [tool.schema() for tool in self.pep.tools.values()]
        result = AgentResult()
        for _ in range(self.max_steps):
            response = self.llm.complete(messages, schemas)
            if not response.tool_calls:
                result.final_text = response.content
                return result
            messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)}}
                    for c in response.tool_calls
                ],
            })
            for call in response.tool_calls:
                outcome = self.pep.handle(call.name, call.arguments)
                result.outcomes.append((call.name, outcome))
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": outcome.content})
        result.stopped_by_step_limit = True
        return result
