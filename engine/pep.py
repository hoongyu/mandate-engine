"""Policy enforcement point: sits between the LLM's tool calls and their execution.

The LLM proposes, the mandate disposes. Every call is audited; side-effecting calls execute only
when the policy decision point says ALLOW, or after a human approves an ASK. A call the policy says
to OBSERVE, and every side-effecting call in shadow mode, is recorded but never executed.
"""

import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from engine.audit import AuditLog
from engine.facts import ActionFacts
from engine.history import ActionHistory
from engine.mandate import Effect, Mandate
from engine.pdp import Decision, decide
from engine.provenance import ProvenanceLabel, ProvenanceTracker, Trust
from engine.tools import Args, ToolSpec

Status = Literal["ok", "pending_approval", "denied", "rejected", "error", "observed"]


@dataclass(frozen=True)
class ToolOutcome:
    status: Status
    content: str  # what the model gets back
    escalation_id: str | None = None


@dataclass(frozen=True)
class Escalation:
    escalation_id: str
    tool: str
    args: Args
    facts: ActionFacts
    decision: Decision
    labels: dict[str, ProvenanceLabel]


def _reply(status: Status, **fields: Any) -> str:
    return json.dumps({"status": status, **fields}, ensure_ascii=False)


def _describe_labels(labels: dict[str, ProvenanceLabel]) -> dict[str, Any]:
    return {
        arg: {"best": lab.best, "seen_in": [s.source for s in lab.sightings]}
        for arg, lab in labels.items()
    }


class PolicyEnforcementPoint:
    def __init__(
        self,
        mandate: Mandate,
        tools: dict[str, ToolSpec],
        audit: AuditLog,
        clock: Callable[[], datetime],
        tracker: ProvenanceTracker | None = None,
        history: ActionHistory | None = None,
        shadow: bool = False,
    ) -> None:
        self.mandate = mandate
        self.tools = tools
        self.audit = audit
        self.clock = clock
        self.tracker = tracker if tracker is not None else ProvenanceTracker()
        self.history = history if history is not None else ActionHistory()
        self.pending: dict[str, Escalation] = {}
        self._shadow = shadow
        self._next_id = itertools.count(1)
        for payee in mandate.payees:  # the saved payee registry is trusted by definition
            self.tracker.observe(
                f"{payee.name} {payee.account}",
                Trust.MANDATE,
                f"mandate:{mandate.mandate_id}#{payee.payee_id}",
            )

    def observe_user(self, text: str, source: str) -> None:
        self.tracker.observe(text, Trust.USER, source)
        self.audit.append("user_message", {"source": source, "text": text}, self.clock())

    def handle(self, name: str, args: Args) -> ToolOutcome:
        now = self.clock()
        tool = self.tools.get(name)
        if tool is None:
            return self._reject(name, args, f"unknown tool {name!r}", now)
        if tool.describe is None:
            return self._execute(tool, args, now)
        try:
            facts = tool.describe(args)
        except ValueError as exc:
            return self._reject(name, args, str(exc), now)
        labels = {arg: self.tracker.label(value) for arg, value in facts.traced_args.items()}
        decision = decide(self.mandate, facts, labels, self.history, now)
        self.audit.append(
            "decision",
            {
                "tool": name,
                "args": args,
                "facts": facts.model_dump(mode="json"),
                "effect": decision.effect,
                "matched": list(decision.matched),
                "reasons": list(decision.reasons),
                "provenance": _describe_labels(labels),
                "mandate": self._mandate_ref(),
            },
            now,
        )
        if self.shadow or decision.effect is Effect.OBSERVE:
            return self._observe(name, decision, now)
        if decision.effect is Effect.ALLOW:
            return self._execute(tool, args, now, facts=facts, decision=decision)
        if decision.effect is Effect.DENY:
            return ToolOutcome("denied", _reply("denied", reasons=list(decision.reasons)))
        escalation = Escalation(f"esc-{next(self._next_id)}", name, args, facts, decision, labels)
        self.pending[escalation.escalation_id] = escalation
        self.audit.append(
            "escalation", {"escalation_id": escalation.escalation_id, "tool": name}, now
        )
        approver = self.mandate.approver
        return ToolOutcome(
            "pending_approval",
            _reply(
                "pending_approval",
                escalation_id=escalation.escalation_id,
                message=f"{approver[:1].upper()}{approver[1:]} must approve this first. Do not retry or split it; "
                "tell the user it is waiting for that approval.",
                reasons=list(decision.reasons),
            ),
            escalation.escalation_id,
        )

    def approve(self, escalation_id: str, approver: str) -> ToolOutcome:
        """A human's yes to a pending action, which is a yes to what was asked, not to what it has since
        become. So the action is described again from the stored arguments and decided on the facts of
        now (the stored labels, the history and the mandate as they stand): a consent that lapsed, a
        grade that fell or an amount that grew since the request is refused here, and what runs is
        what the policy lets through now."""
        escalation = self.pending.pop(escalation_id)
        now = self.clock()
        tool = self.tools[escalation.tool]
        describe = tool.describe
        assert describe is not None  # only a tool with a describe hook is ever escalated
        try:
            facts = describe(escalation.args)
        except ValueError as exc:  # it can no longer be described: the data it rests on is gone or refused
            return self._refuse_approval(escalation_id, approver, [str(exc)], now)
        decision = decide(self.mandate, facts, escalation.labels, self.history, now)
        if decision.effect is Effect.DENY:
            return self._refuse_approval(escalation_id, approver, list(decision.reasons), now)
        if self.shadow or decision.effect is Effect.OBSERVE:
            # OBSERVE outranks ASK: a human's yes cannot execute what the policy now only watches.
            return self._observe(
                escalation.tool, decision, now, escalation_id=escalation_id, approver=approver
            )
        self.audit.append(
            "approved",
            {"escalation_id": escalation_id, "approver": approver, "facts_changed": facts != escalation.facts},
            now,
        )
        return self._execute(
            tool, escalation.args, now, facts=facts, decision=decision, approved_by=approver
        )

    def _refuse_approval(
        self, escalation_id: str, approver: str, reasons: list[str], now: datetime
    ) -> ToolOutcome:
        self.audit.append(
            "approval_refused",
            {"escalation_id": escalation_id, "approver": approver, "reasons": reasons},
            now,
        )
        return ToolOutcome("denied", _reply("denied", reasons=reasons))

    def reject(self, escalation_id: str, approver: str) -> None:
        self.pending.pop(escalation_id)
        self.audit.append(
            "rejected_by_human",
            {"escalation_id": escalation_id, "approver": approver},
            self.clock(),
        )

    def revoke(self, by: str) -> None:
        """Kill switch: the mandate stops authorising anything, including pending approvals."""
        self.mandate = self.mandate.model_copy(update={"revoked": True})
        self.audit.append(
            "mandate_revoked", {"mandate": self._mandate_ref(), "by": by}, self.clock()
        )

    @property
    def shadow(self) -> bool:
        """In shadow mode every side-effecting call is recorded and none is executed."""
        return self._shadow

    def set_shadow(self, on: bool, by: str) -> None:
        """Switch shadow mode. It changes what executes, so it is audited, like revocation."""
        self._shadow = on
        self.audit.append("shadow_changed", {"on": on, "by": by}, self.clock())

    def _execute(
        self,
        tool: ToolSpec,
        args: Args,
        now: datetime,
        facts: ActionFacts | None = None,
        decision: Decision | None = None,
        approved_by: str | None = None,
    ) -> ToolOutcome:
        try:
            result = tool.execute(args)
        except Exception as exc:  # a failing tool must not crash the agent loop
            self.audit.append("error", {"tool": tool.name, "args": args, "error": str(exc)}, now)
            return ToolOutcome("error", _reply("error", error=str(exc)))
        record = self.audit.append(
            "executed",
            {
                "tool": tool.name,
                "args": args,
                "result": result,
                "clauses": list(decision.matched) if decision else [],
                "approved_by": approved_by,
            },
            now,
        )
        self.tracker.observe(result, tool.output_trust, f"tool:{tool.name}#{record.seq}")
        if facts is not None:
            self.history.record(facts, now)
        return ToolOutcome("ok", result)

    def _observe(
        self,
        name: str,
        decision: Decision,
        now: datetime,
        escalation_id: str | None = None,
        approver: str | None = None,
    ) -> ToolOutcome:
        """Record what would have happened. Nothing executes and nothing waits for approval."""
        self.audit.append(
            "observed",
            {
                "tool": name,
                "would_be": decision.effect,
                "shadow": self.shadow,
                "matched": list(decision.matched),
                "reasons": list(decision.reasons),
                "escalation_id": escalation_id,
                "approver": approver,
            },
            now,
        )
        return ToolOutcome(
            "observed",
            _reply(
                "observed",
                would_be=decision.effect,
                reasons=list(decision.reasons),
                message=self._observed_message(decision),
            ),
        )

    def _observed_message(self, decision: Decision) -> str:
        """Tell the model why nothing ran: shadow mode, or which clause only observes."""
        if self.shadow:
            return "Shadow run: nothing was executed."
        observing = [
            c.clause_id
            for c in self.mandate.clauses
            if c.clause_id in decision.matched and c.effect is Effect.OBSERVE
        ]
        if not observing:  # no clause applied and the mandate's default effect is OBSERVE
            return "Observed under the mandate's default effect: nothing was executed."
        noun = "clause" if len(observing) == 1 else "clauses"
        return f"Observed under {noun} {', '.join(observing)}: nothing was executed."

    def _reject(self, name: str, args: Args, reason: str, now: datetime) -> ToolOutcome:
        self.audit.append("rejected", {"tool": name, "args": args, "reason": reason}, now)
        return ToolOutcome("rejected", _reply("rejected", reason=reason))

    def _mandate_ref(self) -> str:
        return f"{self.mandate.mandate_id}@v{self.mandate.version}"
