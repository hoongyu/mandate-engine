from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from engine.facts import ActionFacts
from engine.grade import Grade, rank, weakest
from engine.history import ActionHistory
from engine.mandate import Clause, Effect, Mandate, Payee
from engine.pdp import decide

NOW = datetime(2026, 10, 8, 9, 0, tzinfo=UTC)
LENDER = Payee(
    payee_id="lender",
    name="Sandbox Bank",
    account="LENDER-001",
    category="lender",
    verified_via="mandate",
)
ACCEPT_CHEAP = Clause(
    clause_id="L1",
    effect=Effect.ALLOW,
    actions=frozenset({"loan.accept"}),
    payee_in_registry=True,
    currencies=frozenset({"CNY"}),
    max_amount_minor=1_000_000_000,
    max_rate_bps=450,
    min_grade=Grade.SOURCED,
)
DENY_USURY = Clause(
    clause_id="L2", effect=Effect.DENY, actions=frozenset({"loan.accept"}), min_rate_bps=2400
)


def loan_mandate(*clauses: Clause) -> Mandate:
    return Mandate(
        mandate_id="m-fin",
        version=1,
        principal="company:test",
        agent_id="agent:test",
        valid_from=NOW - timedelta(days=1),
        valid_until=NOW + timedelta(days=30),
        payees=(LENDER,),
        clauses=clauses,
    )


def accept(
    rate_bps: int | None, evidence: dict[str, Grade], amount_minor: int = 600_000_000
) -> ActionFacts:
    return ActionFacts(
        action="loan.accept",
        payee_id="lender",
        payee_category="lender",
        amount_minor=amount_minor,
        currency="CNY",
        rate_bps=rate_bps,
        evidence=evidence,
    )


def test_grades_rank_and_weakest() -> None:
    assert rank(Grade.VERIFIED) > rank(Grade.SOURCED) > rank(Grade.USER) > rank(Grade.ESTIMATED)
    assert weakest([Grade.VERIFIED, Grade.USER, Grade.SOURCED]) is Grade.USER


def test_every_grade_has_its_own_rank() -> None:
    assert sorted(rank(g) for g in Grade) == list(range(len(Grade)))


def test_weakest_of_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least one grade"):
        weakest([])


def test_negative_rate_is_rejected() -> None:
    assert ActionFacts(action="loan.accept", rate_bps=0).rate_bps == 0
    with pytest.raises(ValidationError):
        ActionFacts(action="loan.accept", rate_bps=-1)


def test_cheap_loan_on_sourced_evidence_is_allowed() -> None:
    d = decide(
        loan_mandate(ACCEPT_CHEAP, DENY_USURY),
        accept(420, {"revenue": Grade.VERIFIED, "net_burn_monthly": Grade.SOURCED}),
        {},
        ActionHistory(),
        NOW,
    )
    assert d.effect is Effect.ALLOW and d.matched == ("L1",)


def test_rate_above_cap_falls_back_to_ask() -> None:
    d = decide(loan_mandate(ACCEPT_CHEAP), accept(600, {"revenue": Grade.VERIFIED}), {}, ActionHistory(), NOW)
    assert d.effect is Effect.ASK
    assert any("rate 6.00% exceeds 4.50%" in r for r in d.reasons)


def test_estimated_evidence_is_not_enough() -> None:
    d = decide(loan_mandate(ACCEPT_CHEAP), accept(420, {"revenue": Grade.ESTIMATED}), {}, ActionHistory(), NOW)
    assert d.effect is Effect.ASK
    assert any("evidence weaker than sourced: revenue (estimated)" in r for r in d.reasons)


def test_missing_rate_and_missing_evidence_are_explained() -> None:
    d = decide(loan_mandate(ACCEPT_CHEAP), accept(None, {}), {}, ActionHistory(), NOW)
    assert d.effect is Effect.ASK
    assert any("the action has no interest rate" in r for r in d.reasons)
    d = decide(loan_mandate(ACCEPT_CHEAP), accept(420, {}), {}, ActionHistory(), NOW)
    assert d.effect is Effect.ASK
    assert any("the action cites no evidence" in r for r in d.reasons)


def test_usurious_rate_is_denied_even_if_another_clause_allows() -> None:
    permissive = ACCEPT_CHEAP.model_copy(update={"max_rate_bps": 5000})
    d = decide(
        loan_mandate(permissive, DENY_USURY), accept(3000, {"revenue": Grade.VERIFIED}), {}, ActionHistory(), NOW
    )
    assert d.effect is Effect.DENY


@pytest.mark.parametrize(("rate_bps", "effect"), [(450, Effect.ALLOW), (451, Effect.ASK)])
def test_max_rate_is_inclusive(rate_bps: int, effect: Effect) -> None:
    facts = accept(rate_bps, {"revenue": Grade.VERIFIED})
    d = decide(loan_mandate(ACCEPT_CHEAP), facts, {}, ActionHistory(), NOW)
    assert d.effect is effect


@pytest.mark.parametrize(("rate_bps", "effect"), [(2399, Effect.ASK), (2400, Effect.DENY)])
def test_min_rate_is_inclusive(rate_bps: int, effect: Effect) -> None:
    facts = accept(rate_bps, {"revenue": Grade.VERIFIED})
    d = decide(loan_mandate(DENY_USURY), facts, {}, ActionHistory(), NOW)
    assert d.effect is effect


@pytest.mark.parametrize(
    ("grade", "effect"), [(Grade.SOURCED, Effect.ALLOW), (Grade.USER, Effect.ASK)]
)
def test_evidence_grade_boundary(grade: Grade, effect: Effect) -> None:
    # As strings "user" > "sourced", but USER ranks below SOURCED: only rank() may decide.
    facts = accept(420, {"revenue": grade})
    d = decide(loan_mandate(ACCEPT_CHEAP), facts, {}, ActionHistory(), NOW)
    assert d.effect is effect


def test_past_loans_count_toward_the_cap_whatever_their_rate_or_grade() -> None:
    capped = Clause(
        clause_id="L3",
        effect=Effect.ALLOW,
        actions=frozenset({"loan.accept"}),
        currencies=frozenset({"CNY"}),
        period_cap_minor=1_000_000_000,
        max_rate_bps=450,
        min_grade=Grade.SOURCED,
    )
    request = accept(420, {"revenue": Grade.SOURCED}, amount_minor=500_000_000)
    alone = decide(loan_mandate(capped), request, {}, ActionHistory(), NOW)
    assert alone.effect is Effect.ALLOW
    history = ActionHistory()  # CNY 8m earlier, at a rate and on evidence the clause would refuse
    history.record(
        accept(600, {"revenue": Grade.ESTIMATED}, amount_minor=800_000_000), NOW - timedelta(days=2)
    )
    d = decide(loan_mandate(capped), request, {}, history, NOW)
    assert d.effect is Effect.ASK
    assert any("30-day total would be CNY 13,000,000.00" in r for r in d.reasons)


def test_observe_beats_ask_whichever_clause_comes_first() -> None:
    observe = Clause(clause_id="L4", effect=Effect.OBSERVE, actions=frozenset({"loan.accept"}))
    ask = Clause(clause_id="L5", effect=Effect.ASK, actions=frozenset({"loan.accept"}))
    facts = accept(420, {"revenue": Grade.VERIFIED})
    for clauses in ((observe, ask), (ask, observe)):
        d = decide(loan_mandate(*clauses), facts, {}, ActionHistory(), NOW)
        assert d.effect is Effect.OBSERVE
        assert set(d.matched) == {"L4", "L5"}
