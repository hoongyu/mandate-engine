"""Agent identity: a credential a counterparty checks before it accepts an agent's request.

It says which agent acts for which principal, under which mandate version, for which actions, and
until when. Signed with HMAC-SHA256 over a canonical JSON body; a production system would use
asymmetric signatures (listed as a known limitation in docs/SECURITY.md).
"""

import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta

from pydantic import AwareDatetime, BaseModel, ConfigDict

from engine.mandate import Mandate

MIN_KEY_BYTES = 16


class AgentCredential(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    principal: str
    mandate_id: str
    mandate_version: int
    actions: frozenset[str]
    issued_at: AwareDatetime
    expires_at: AwareDatetime
    signature: str = ""

    def signing_bytes(self) -> bytes:
        body = {
            "agent_id": self.agent_id,
            "principal": self.principal,
            "mandate_id": self.mandate_id,
            "mandate_version": self.mandate_version,
            "actions": sorted(self.actions),
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sign(cred: AgentCredential, key: bytes) -> str:
    if len(key) < MIN_KEY_BYTES:
        raise ValueError(f"key must be at least {MIN_KEY_BYTES} bytes")
    return hmac.new(key, cred.signing_bytes(), hashlib.sha256).hexdigest()


def _require_aware(now: datetime) -> None:
    if now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")


def issue(
    mandate: Mandate, actions: frozenset[str], key: bytes, now: datetime, ttl: timedelta
) -> AgentCredential:
    """Sign a credential valid from `now` for `ttl`, but never past the mandate.

    Revocation is deliberately not checked here: the enforcement point checks it on every call,
    and credentials are meant to be short-lived.
    """
    _require_aware(now)
    now = now.astimezone(UTC)
    if ttl <= timedelta(0):
        raise ValueError("ttl must be positive")
    if now < mandate.valid_from:
        raise ValueError(f"mandate {mandate.mandate_id} is not valid before {mandate.valid_from.isoformat()}")
    if now >= mandate.valid_until:
        raise ValueError(f"mandate {mandate.mandate_id} expired at {mandate.valid_until.isoformat()}")
    unsigned = AgentCredential(
        agent_id=mandate.agent_id,
        principal=mandate.principal,
        mandate_id=mandate.mandate_id,
        mandate_version=mandate.version,
        actions=actions,
        issued_at=now,
        expires_at=now + min(ttl, mandate.valid_until - now),  # UTC, and no overflow on a huge ttl
    )
    return unsigned.model_copy(update={"signature": _sign(unsigned, key)})


def check(
    cred: AgentCredential, key: bytes, now: datetime, action: str, principal: str
) -> str | None:
    """None if the credential covers this action for this principal now, else the reason.

    A naive `now` or a weak key is a programming error and raises ValueError.
    """
    _require_aware(now)
    expected = _sign(cred, key)  # a weak key raises before any credential is judged
    # compare_digest raises on a non-ASCII str, and a forged credential must be refused, not crash
    # the counterparty. isascii() reads only the caller's own input, so it leaks nothing about the key.
    if not cred.signature.isascii() or not hmac.compare_digest(cred.signature, expected):
        return "signature does not match"
    if not cred.issued_at <= now < cred.expires_at:
        return "credential is not valid at this time"
    if action not in cred.actions:
        return f"credential does not cover {action!r}"
    if cred.principal != principal:
        return f"credential acts for {cred.principal!r}, not {principal!r}"
    return None
