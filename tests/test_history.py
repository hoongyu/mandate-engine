from datetime import UTC, datetime, timedelta

from engine.facts import ActionFacts
from engine.history import ActionHistory

T0 = datetime(2026, 10, 1, tzinfo=UTC)


def test_window_is_rolling_and_filtered() -> None:
    history = ActionHistory()
    rent = ActionFacts(action="transfer", payee_id="landlord", amount_minor=100, currency="EUR")
    other = ActionFacts(action="convert", amount_minor=999, currency="EUR")
    history.record(rent, T0 - timedelta(days=31))
    history.record(rent, T0 - timedelta(days=29))
    history.record(other, T0 - timedelta(days=1))
    recent_transfers = history.within(30, T0, lambda f: f.action == "transfer")
    assert recent_transfers == [rent]


def test_actions_in_the_future_do_not_count() -> None:
    history = ActionHistory()
    history.record(ActionFacts(action="transfer"), T0 + timedelta(seconds=1))
    assert history.within(30, T0, lambda f: True) == []
