"""Policy decision point: a pure function from (mandate, action, provenance, history, time) to a
decision. It explains every clause it looked at, so an escalation card can say why it is asking."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from engine.facts import ActionFacts
from engine.grade import rank
from engine.history import ActionHistory
from engine.mandate import RESTRICTIVENESS, Clause, Effect, Mandate
from engine.money import format_minor
from engine.provenance import ProvenanceLabel


@dataclass(frozen=True)
class Decision:
    effect: Effect
    matched: tuple[str, ...]
    reasons: tuple[str, ...]


def decide(
    mandate: Mandate,
    facts: ActionFacts,
    labels: dict[str, ProvenanceLabel],
    history: ActionHistory,
    now: datetime,
) -> Decision:
    if not mandate.is_active(now):
        return Decision(Effect.DENY, (), ("mandate is revoked or outside its validity window",))
    matched: list[Clause] = []
    reasons: list[str] = []
    for clause in mandate.clauses:
        if facts.action not in clause.actions:
            continue
        why_not = _why_not(clause, facts, labels, history, now)
        if why_not is None:
            matched.append(clause)
            reasons.append(f"{clause.clause_id} ({clause.effect}) applies")
        else:
            reasons.append(f"{clause.clause_id} ({clause.effect}) does not apply: {why_not}")
    if not matched:
        reasons.append(f"no clause applies; default is {mandate.default_effect}")
        return Decision(mandate.default_effect, (), tuple(reasons))
    effect = max((c.effect for c in matched), key=RESTRICTIVENESS.__getitem__)
    return Decision(effect, tuple(c.clause_id for c in matched), tuple(reasons))


def _out_of_scope(clause: Clause, facts: ActionFacts) -> str | None:
    """Static conditions. They also decide which past actions count toward the clause's caps."""
    if clause.payee_in_registry is not None:
        in_registry = facts.payee_id is not None
        if in_registry != clause.payee_in_registry:
            return "payee is in the registry" if in_registry else "payee is not in the registry"
    if clause.payee_categories is not None and facts.payee_category not in clause.payee_categories:
        return f"payee category {facts.payee_category!r} is not one of {sorted(clause.payee_categories)}"
    if clause.currencies is not None and facts.currency not in clause.currencies:
        return f"currency {facts.currency!r} is not one of {sorted(clause.currencies)}"
    # Unknown consent (None) matches neither True nor False: an action that cannot say whether the
    # owner agreed is never allowed by a clause for consented sharing.
    if clause.within_consent is not None and facts.within_consent is not clause.within_consent:
        if facts.within_consent is None:
            return "the action does not say whether the data owner's consent covers it"
        return "the data owner's consent covers it" if facts.within_consent else "the data owner's consent does not cover it"
    return None


def _pct(bps: int) -> str:
    return f"{Decimal(bps) / 100:.2f}%"


def _rate_why_not(clause: Clause, facts: ActionFacts) -> str | None:
    if clause.min_rate_bps is None and clause.max_rate_bps is None:
        return None
    if facts.rate_bps is None:
        return "the action has no interest rate"
    if clause.min_rate_bps is not None and facts.rate_bps < clause.min_rate_bps:
        return f"rate {_pct(facts.rate_bps)} is below {_pct(clause.min_rate_bps)}"
    if clause.max_rate_bps is not None and facts.rate_bps > clause.max_rate_bps:
        return f"rate {_pct(facts.rate_bps)} exceeds {_pct(clause.max_rate_bps)}"
    return None


def _grade_why_not(clause: Clause, facts: ActionFacts) -> str | None:
    if clause.min_grade is None:
        return None
    if not facts.evidence:
        return "the action cites no evidence"
    weak = sorted(k for k, g in facts.evidence.items() if rank(g) < rank(clause.min_grade))
    if weak:
        listed = ", ".join(f"{k} ({facts.evidence[k]})" for k in weak)
        return f"evidence weaker than {clause.min_grade}: {listed}"
    return None


def _why_not(
    clause: Clause,
    facts: ActionFacts,
    labels: dict[str, ProvenanceLabel],
    history: ActionHistory,
    now: datetime,
) -> str | None:
    """None if the clause applies to this action, otherwise the first reason it does not."""
    scope = _out_of_scope(clause, facts)
    if scope is not None:
        return scope
    amounts = (clause.min_amount_minor, clause.max_amount_minor, clause.period_cap_minor)
    if any(a is not None for a in amounts):
        if facts.amount_minor is None:
            return "the action has no amount"
        cur = clause.currency
        if clause.min_amount_minor is not None and facts.amount_minor < clause.min_amount_minor:
            return f"amount {format_minor(facts.amount_minor, cur)} is below {format_minor(clause.min_amount_minor, cur)}"
        if clause.max_amount_minor is not None and facts.amount_minor > clause.max_amount_minor:
            return f"amount {format_minor(facts.amount_minor, cur)} exceeds {format_minor(clause.max_amount_minor, cur)}"
    # Rate and grade judge this request only: past actions under the clause still count toward its caps.
    rate = _rate_why_not(clause, facts)
    if rate is not None:
        return rate
    grade = _grade_why_not(clause, facts)
    if grade is not None:
        return grade
    for arg in sorted(clause.trusted_args):
        label = labels.get(arg)
        if label is None or not label.trusted:
            return f"{arg} is not from a trusted source ({label.best if label else 'missing'})"
    if clause.period_cap_minor is not None or clause.max_count_per_period is not None:
        past = history.within(
            clause.period_days,
            now,
            lambda f: f.action in clause.actions and _out_of_scope(clause, f) is None,
        )
        if clause.period_cap_minor is not None and facts.amount_minor is not None:
            total = sum(f.amount_minor or 0 for f in past) + facts.amount_minor
            if total > clause.period_cap_minor:
                cur = clause.currency
                return (
                    f"{clause.period_days}-day total would be {format_minor(total, cur)}, "
                    f"cap is {format_minor(clause.period_cap_minor, cur)}"
                )
        if clause.max_count_per_period is not None and len(past) + 1 > clause.max_count_per_period:
            return (
                f"this would be action #{len(past) + 1} in {clause.period_days} days, "
                f"limit is {clause.max_count_per_period}"
            )
    return None
