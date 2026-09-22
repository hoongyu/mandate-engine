import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from engine.audit import GENESIS, AuditLog

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)


def test_records_are_chained() -> None:
    log = AuditLog()
    a = log.append("decision", {"effect": "allow"}, T0)
    b = log.append("executed", {"tool": "transfer"}, T0 + timedelta(seconds=1))
    assert a.prev_hash == GENESIS
    assert b.prev_hash == a.hash
    assert log.verify() is None


def test_tampering_in_memory_is_detected() -> None:
    log = AuditLog()
    log.append("decision", {"effect": "ask"}, T0)
    log.append("executed", {"amount": "EUR 850.00"}, T0)
    log._records[0] = log.records()[0].model_copy(update={"payload": {"effect": "allow"}})
    assert log.verify() == 0


def test_persists_to_jsonl_and_reloads(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append("decision", {"effect": "allow", "clauses": ["C1"]}, T0)
    log.append("executed", {"to": "Hausverwaltung Weber"}, T0)
    again = AuditLog(path)
    assert [r.hash for r in again.records()] == [r.hash for r in log.records()]
    assert again.verify() is None


def test_editing_the_file_breaks_the_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append("decision", {"effect": "ask"}, T0)
    log.append("executed", {"amount": "EUR 4,500.00"}, T0)
    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["payload"]["effect"] = "allow"
    lines[0] = json.dumps(first)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="broken at seq 0"):
        AuditLog(path)


def test_payload_is_frozen_to_json_types_and_naive_times_are_refused() -> None:
    log = AuditLog()
    record = log.append("x", {"when": T0, "n": 1}, T0)
    assert record.payload == {"when": T0.isoformat(), "n": 1}
    with pytest.raises(ValueError):
        log.append("x", {}, datetime(2026, 10, 1))
