from datetime import UTC, datetime, timedelta

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


def accept(rate_bps: int | None, evidence: dict[str, Grade]) -> ActionFacts:
    return ActionFacts(
        action="loan.accept",
        payee_id="lender",
        payee_category="lender",
        amount_minor=600_000_000,
        currency="CNY",
        rate_bps=rate_bps,
        evidence=evidence,
    )


def test_grades_rank_and_weakest() -> None:
    assert rank(Grade.VERIFIED) > rank(Grade.SOURCED) > rank(Grade.USER) > rank(Grade.ESTIMATED)
    assert weakest([Grade.VERIFIED, Grade.USER, Grade.SOURCED]) is Grade.USER


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
    assert any("the action has no interest rate" in r for r in d.reasons)
    d = decide(loan_mandate(ACCEPT_CHEAP), accept(420, {}), {}, ActionHistory(), NOW)
    assert any("the action cites no evidence" in r for r in d.reasons)


def test_usurious_rate_is_denied_even_if_another_clause_allows() -> None:
    permissive = ACCEPT_CHEAP.model_copy(update={"max_rate_bps": 5000})
    d = decide(
        loan_mandate(permissive, DENY_USURY), accept(3000, {"revenue": Grade.VERIFIED}), {}, ActionHistory(), NOW
    )
    assert d.effect is Effect.DENY
