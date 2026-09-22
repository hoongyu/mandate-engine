"""Append-only, hash-chained audit log. Each record commits to the previous one, so editing or
deleting any past record breaks verification from that point on."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

GENESIS = "0" * 64


class AuditRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int
    ts: datetime
    kind: str
    payload: dict[str, Any]
    prev_hash: str
    hash: str


def _jsonable(obj: object) -> str:
    return obj.isoformat() if isinstance(obj, datetime) else str(obj)


def _digest(seq: int, ts: datetime, kind: str, payload: dict[str, Any], prev_hash: str) -> str:
    body = {"seq": seq, "ts": ts.isoformat(), "kind": kind, "payload": payload, "prev_hash": prev_hash}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._records: list[AuditRecord] = []
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._records.append(AuditRecord.model_validate_json(line))
            broken = self.verify()
            if broken is not None:
                raise ValueError(f"audit log {path} is broken at seq {broken}")

    def append(self, kind: str, payload: dict[str, Any], ts: datetime) -> AuditRecord:
        if ts.tzinfo is None:
            raise ValueError("audit timestamps must be timezone-aware")
        ts = ts.astimezone(UTC)
        frozen: dict[str, Any] = json.loads(json.dumps(payload, default=_jsonable, ensure_ascii=False))
        seq = len(self._records)
        prev = self._records[-1].hash if self._records else GENESIS
        record = AuditRecord(
            seq=seq, ts=ts, kind=kind, payload=frozen, prev_hash=prev,
            hash=_digest(seq, ts, kind, frozen, prev),
        )
        self._records.append(record)
        if self._path is not None:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(record.model_dump_json() + "\n")
        return record

    def records(self) -> tuple[AuditRecord, ...]:
        return tuple(self._records)

    def verify(self) -> int | None:
        """Index of the first record that does not check out, or None if the chain is intact."""
        prev = GENESIS
        for i, r in enumerate(self._records):
            expected = _digest(r.seq, r.ts, r.kind, r.payload, r.prev_hash)
            if r.seq != i or r.prev_hash != prev or r.hash != expected:
                return i
            prev = r.hash
        return None
