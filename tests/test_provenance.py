import pytest

from engine.provenance import ProvenanceTracker, Trust

IBAN = "DE89 3704 0044 0532 0130 00"


def test_value_seen_nowhere_is_unknown() -> None:
    label = ProvenanceTracker().label(IBAN)
    assert label.best is Trust.UNKNOWN
    assert label.sightings == ()
    assert not label.trusted


def test_matching_ignores_spaces_and_case() -> None:
    tracker = ProvenanceTracker()
    tracker.observe("please pay to de89370400440532013000 today", Trust.UNTRUSTED, "email#1")
    label = tracker.label(IBAN)
    assert label.best is Trust.UNTRUSTED
    assert [s.source for s in label.sightings] == ["email#1"]


def test_best_sighting_wins_but_all_are_kept() -> None:
    tracker = ProvenanceTracker()
    tracker.observe(f"attacker repeats {IBAN}", Trust.UNTRUSTED, "email#1")
    tracker.observe(f"Hausverwaltung Weber {IBAN}", Trust.MANDATE, "mandate#landlord")
    label = tracker.label(IBAN)
    assert label.best is Trust.MANDATE
    assert label.trusted
    assert {s.source for s in label.sightings} == {"email#1", "mandate#landlord"}


def test_user_typed_value_is_trusted() -> None:
    tracker = ProvenanceTracker()
    tracker.observe(f"send 20 euros to {IBAN}", Trust.USER, "user#1")
    assert tracker.label(IBAN).best is Trust.USER


def test_too_short_values_are_untraceable() -> None:
    tracker = ProvenanceTracker()
    tracker.observe("DE", Trust.USER, "user#1")
    assert tracker.label("DE").best is Trust.UNKNOWN


def test_unknown_is_not_an_input_label() -> None:
    with pytest.raises(ValueError):
        ProvenanceTracker().observe("x", Trust.UNKNOWN, "?")
