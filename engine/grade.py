"""How well a number is backed by evidence. A higher rank is stronger evidence.

The engine only compares grades; the evidence package decides which grade a number earns.
"""

from collections.abc import Iterable
from enum import StrEnum


class Grade(StrEnum):
    """Compare grades only through rank() and weakest(), never with `<` or `>`.

    StrEnum members compare as plain strings, so `<` and `>` order them alphabetically, not by
    strength: Grade.USER > Grade.SOURCED is True as strings, although SOURCED is the stronger.
    """

    VERIFIED = "verified"  # two independent documents agree, or official market data
    SOURCED = "sourced"  # read from one document, with page and verbatim excerpt
    USER = "user"  # typed in by the principal, unchecked
    ESTIMATED = "estimated"  # produced by a model or an assumption, or in conflict


_RANK: dict[Grade, int] = {
    Grade.ESTIMATED: 0,
    Grade.USER: 1,
    Grade.SOURCED: 2,
    Grade.VERIFIED: 3,
}


def rank(grade: Grade) -> int:
    return _RANK[grade]


def weakest(grades: Iterable[Grade]) -> Grade:
    """A result is only as strong as its weakest input."""
    items = list(grades)
    if not items:
        raise ValueError("weakest() needs at least one grade")
    return min(items, key=rank)
