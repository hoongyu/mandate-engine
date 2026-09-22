from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from engine.mandate import Clause, Effect, Mandate, Payee

T0 = datetime(2026, 10, 1, tzinfo=UTC)
LANDLORD = Payee(
    payee_id="landlord",
    name="Hausverwaltung Weber",
    account="DE89 3704 0044 0532 0130 00",
    category="rent",
    verified_via="user_confirmed",
)


def make(**overrides: Any) -> Mandate:
    fields: dict[str, Any] = {
        "mandate_id": "m1",
        "version": 1,
        "principal": "p",
        "agent_id": "a",
        "valid_from": T0,
        "valid_until": T0 + timedelta(days=30),
        "payees": (LANDLORD,),
        "clauses": (),
    }
    fields.update(overrides)
    return Mandate.model_validate(fields)


def test_amount_conditions_need_exactly_one_currency() -> None:
    with pytest.raises(ValidationError):
        Clause(clause_id="C1", effect=Effect.ALLOW, actions=frozenset({"transfer"}), max_amount_minor=100)
    with pytest.raises(ValidationError):
        Clause(
            clause_id="C1",
            effect=Effect.ALLOW,
            actions=frozenset({"transfer"}),
            currencies=frozenset({"EUR", "USD"}),
            period_cap_minor=100,
        )


def test_default_effect_cannot_be_allow() -> None:
    with pytest.raises(ValidationError):
        make(default_effect=Effect.ALLOW)


def test_rejects_duplicate_clause_ids_and_naive_datetimes() -> None:
    clause = Clause(clause_id="C1", effect=Effect.DENY, actions=frozenset({"transfer"}))
    with pytest.raises(ValidationError):
        make(clauses=(clause, clause))
    with pytest.raises(ValidationError):
        make(valid_from=datetime(2026, 10, 1))


def test_is_active_respects_window_and_revocation() -> None:
    mandate = make()
    assert mandate.is_active(T0)
    assert not mandate.is_active(T0 + timedelta(days=30))
    assert not make(revoked=True).is_active(T0)


def test_payee_lookup_by_id_and_by_account() -> None:
    mandate = make()
    assert mandate.payee("landlord") == LANDLORD
    assert mandate.payee_by_account("de89370400440532013000") == LANDLORD
    assert mandate.payee("nobody") is None
