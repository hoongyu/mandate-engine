"""Tool-independent, policy-relevant facts about one proposed side-effecting action.

Each side-effecting tool translates its raw arguments into ActionFacts, so the policy decision
point never has to understand tool-specific argument shapes.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ActionFacts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    action: str
    payee_id: str | None = None  # set when the counterparty is a payee saved in the mandate
    payee_category: str | None = None
    counterparty_account: str | None = None
    counterparty_name: str | None = None
    amount_minor: int | None = None
    currency: str | None = None
    traced_args: dict[str, str] = Field(default_factory=dict)  # arg name -> value to trace


class ExecutedAction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    facts: ActionFacts
    executed_at: datetime
