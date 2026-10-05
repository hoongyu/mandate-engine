"""The mandate: what the principal has authorised the agent to do on their own.

Semantics (order-independent, like Cedar's forbid-overrides-permit):
- A clause applies to an action when every condition it states holds.
- Among the clauses that apply, the most restrictive effect wins: DENY > OBSERVE > ASK > ALLOW.
- If none applies, the mandate's default effect is used, and it can never be ALLOW (fail-safe).
"""

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from engine.grade import Grade
from engine.provenance import normalize


class Effect(StrEnum):
    ALLOW = "allow"  # the agent may act on its own
    ASK = "ask"  # a human must approve first
    OBSERVE = "observe"  # shadow run: record what would happen, never execute
    DENY = "deny"  # never, not even with approval


RESTRICTIVENESS: dict[Effect, int] = {
    Effect.ALLOW: 0,
    Effect.ASK: 1,
    Effect.OBSERVE: 2,
    Effect.DENY: 3,
}


class Payee(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    payee_id: str
    name: str
    account: str
    category: str
    verified_via: str


class Clause(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    clause_id: str
    effect: Effect
    actions: frozenset[str]
    payee_in_registry: bool | None = None
    payee_categories: frozenset[str] | None = None
    currencies: frozenset[str] | None = None
    min_amount_minor: int | None = Field(default=None, gt=0)
    max_amount_minor: int | None = Field(default=None, gt=0)
    period_cap_minor: int | None = Field(default=None, gt=0)
    max_count_per_period: int | None = Field(default=None, gt=0)
    period_days: int = Field(default=30, gt=0)
    min_rate_bps: int | None = Field(default=None, ge=0)
    max_rate_bps: int | None = Field(default=None, ge=0)
    min_grade: Grade | None = None
    within_consent: bool | None = None  # applies only to actions inside (True) or outside (False) the owner's consent
    trusted_args: frozenset[str] = frozenset()

    @model_validator(mode="after")
    def _amounts_need_one_currency(self) -> Self:
        amounts = (self.min_amount_minor, self.max_amount_minor, self.period_cap_minor)
        if any(a is not None for a in amounts) and (
            self.currencies is None or len(self.currencies) != 1
        ):
            raise ValueError(f"{self.clause_id}: amount conditions need exactly one currency")
        return self

    @model_validator(mode="after")
    def _rate_bounds_are_ordered(self) -> Self:
        if (
            self.min_rate_bps is not None
            and self.max_rate_bps is not None
            and self.min_rate_bps > self.max_rate_bps
        ):
            raise ValueError(f"{self.clause_id}: min_rate_bps must not exceed max_rate_bps")
        return self

    @property
    def currency(self) -> str:
        """The single currency of a clause that has amount conditions."""
        assert self.currencies is not None and len(self.currencies) == 1
        return next(iter(self.currencies))


class Mandate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mandate_id: str
    version: int = Field(ge=1)
    principal: str
    agent_id: str
    valid_from: datetime
    valid_until: datetime
    revoked: bool = False
    payees: tuple[Payee, ...]
    clauses: tuple[Clause, ...]
    default_effect: Effect = Effect.ASK
    approver: str = "the user"  # who answers an ASK: the user the agent works for, or a body the mandate names

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.valid_from.tzinfo is None or self.valid_until.tzinfo is None:
            raise ValueError("validity datetimes must be timezone-aware")
        if self.valid_until <= self.valid_from:
            raise ValueError("valid_until must be after valid_from")
        if self.default_effect is Effect.ALLOW:
            raise ValueError("default_effect ALLOW would let unmatched actions run unattended")
        for what, ids in (
            ("clause_id", [c.clause_id for c in self.clauses]),
            ("payee_id", [p.payee_id for p in self.payees]),
        ):
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {what}")
        return self

    def is_active(self, now: datetime) -> bool:
        return not self.revoked and self.valid_from <= now < self.valid_until

    def payee(self, payee_id: str) -> Payee | None:
        return next((p for p in self.payees if p.payee_id == payee_id), None)

    def payee_by_account(self, account: str) -> Payee | None:
        key = normalize(account)
        return next((p for p in self.payees if normalize(p.account) == key), None)
