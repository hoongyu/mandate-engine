"""Where did a value come from?

Values the agent passes to money-moving tools are traced back to the texts it has seen, so an IBAN
that only ever appeared in an email can be told apart from one the user typed or one saved in the
mandate. Matching is on whitespace- and case-normalised text, so no field extraction is needed.
"""

import re
from dataclasses import dataclass, field
from enum import StrEnum


class Trust(StrEnum):
    MANDATE = "mandate"  # saved in the signed mandate (e.g. the payee registry)
    SYSTEM = "system"  # our own systems of record (balances, executed transfers)
    USER = "user"  # typed by the principal in this session
    UNTRUSTED = "untrusted"  # emails, documents, web pages, third-party messages
    UNKNOWN = "unknown"  # seen nowhere: possibly hallucinated


TRUSTED: frozenset[Trust] = frozenset({Trust.MANDATE, Trust.SYSTEM, Trust.USER})
_RANK: dict[Trust, int] = {
    Trust.MANDATE: 4,
    Trust.SYSTEM: 3,
    Trust.USER: 2,
    Trust.UNTRUSTED: 1,
    Trust.UNKNOWN: 0,
}
MIN_TRACEABLE = 6


def normalize(text: str) -> str:
    """Whitespace- and case-insensitive form (IBANs are usually printed in groups of four)."""
    return re.sub(r"\s+", "", text).upper()


@dataclass(frozen=True)
class Sighting:
    trust: Trust
    source: str


@dataclass(frozen=True)
class ProvenanceLabel:
    value: str
    best: Trust
    sightings: tuple[Sighting, ...]

    @property
    def trusted(self) -> bool:
        return self.best in TRUSTED


@dataclass
class ProvenanceTracker:
    _seen: list[tuple[str, Sighting]] = field(default_factory=list, init=False, repr=False)

    def observe(self, text: str, trust: Trust, source: str) -> None:
        if trust is Trust.UNKNOWN:
            raise ValueError("UNKNOWN is a result of tracing, not a label you can assign")
        self._seen.append((normalize(text), Sighting(trust, source)))

    def label(self, value: str) -> ProvenanceLabel:
        needle = normalize(value)
        if len(needle) < MIN_TRACEABLE:
            return ProvenanceLabel(value, Trust.UNKNOWN, ())
        sightings = tuple(s for hay, s in self._seen if needle in hay)
        best = max((s.trust for s in sightings), key=_RANK.__getitem__, default=Trust.UNKNOWN)
        return ProvenanceLabel(value, best, sightings)
