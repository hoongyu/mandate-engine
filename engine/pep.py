"""Policy enforcement point: sits between the LLM's tool calls and their execution.

The LLM proposes, the mandate disposes. Every call is audited; side-effecting calls execute only
when the policy decision point says ALLOW, or after a human approves an ASK.
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

Status = Literal["ok", "pending_approval", "denied", "rejected", "error"]


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
    ) -> None:
        self.mandate = mandate
        self.tools = tools
        self.audit = audit
        self.clock = clock
        self.tracker = tracker if tracker is not None else ProvenanceTracker()
        self.history = history if history is not None else ActionHistory()
        self.pending: dict[str, Escalation] = {}
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
        if decision.effect is Effect.ALLOW:
            return self._execute(tool, args, now, facts=facts, decision=decision)
        if decision.effect is Effect.DENY:
            return ToolOutcome("denied", _reply("denied", reasons=list(decision.reasons)))
        escalation = Escalation(f"esc-{next(self._next_id)}", name, args, facts, decision, labels)
        self.pending[escalation.escalation_id] = escalation
        self.audit.append(
            "escalation", {"escalation_id": escalation.escalation_id, "tool": name}, now
        )
        return ToolOutcome(
            "pending_approval",
            _reply(
                "pending_approval",
                escalation_id=escalation.escalation_id,
                message="The user must approve this first. Do not retry or split it; tell the user it is waiting for their approval.",
                reasons=list(decision.reasons),
            ),
            escalation.escalation_id,
        )

    def approve(self, escalation_id: str, approver: str) -> ToolOutcome:
        escalation = self.pending.pop(escalation_id)
        now = self.clock()
        decision = decide(self.mandate, escalation.facts, escalation.labels, self.history, now)
        if decision.effect is Effect.DENY:
            self.audit.append(
                "approval_refused",
                {
                    "escalation_id": escalation_id,
                    "approver": approver,
                    "reasons": list(decision.reasons),
                },
                now,
            )
            return ToolOutcome("denied", _reply("denied", reasons=list(decision.reasons)))
        self.audit.append("approved", {"escalation_id": escalation_id, "approver": approver}, now)
        return self._execute(
            self.tools[escalation.tool],
            escalation.args,
            now,
            facts=escalation.facts,
            decision=decision,
            approved_by=approver,
        )

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

    def _reject(self, name: str, args: Args, reason: str, now: datetime) -> ToolOutcome:
        self.audit.append("rejected", {"tool": name, "args": args, "reason": reason}, now)
        return ToolOutcome("rejected", _reply("rejected", reason=reason))

    def _mandate_ref(self) -> str:
        return f"{self.mandate.mandate_id}@v{self.mandate.version}"
