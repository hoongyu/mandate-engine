"""Side-effecting actions that actually executed; feeds cumulative caps and velocity limits."""

from collections.abc import Callable
from datetime import datetime, timedelta

from engine.facts import ActionFacts, ExecutedAction


class ActionHistory:
    def __init__(self) -> None:
        self._done: list[ExecutedAction] = []

    def record(self, facts: ActionFacts, at: datetime) -> None:
        self._done.append(ExecutedAction(facts=facts, executed_at=at))

    def within(
        self, days: int, now: datetime, keep: Callable[[ActionFacts], bool]
    ) -> list[ActionFacts]:
        start = now - timedelta(days=days)
        return [e.facts for e in self._done if start < e.executed_at <= now and keep(e.facts)]
