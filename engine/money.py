"""Money is an integer number of minor units (cents) plus an ISO currency code. Never floats."""

from decimal import Decimal, InvalidOperation

_ZERO_DECIMAL = frozenset({"JPY", "KRW"})


def minor_digits(currency: str) -> int:
    return 0 if currency.upper() in _ZERO_DECIMAL else 2


def to_minor(amount: str | int | float | Decimal, currency: str) -> int:
    """Parse a positive major-unit amount such as "850.00" into minor units (85000)."""
    try:
        value = Decimal(str(amount).strip())
    except InvalidOperation as exc:
        raise ValueError(f"not a number: {amount!r}") from exc
    if not value.is_finite() or value <= 0:
        raise ValueError(f"amount must be a positive number: {amount!r}")
    scaled = value.scaleb(minor_digits(currency))
    if scaled != scaled.to_integral_value():
        raise ValueError(f"too many decimal places for {currency.upper()}: {amount!r}")
    return int(scaled)


def format_minor(amount_minor: int, currency: str) -> str:
    digits = minor_digits(currency)
    major = Decimal(amount_minor).scaleb(-digits)
    return f"{currency.upper()} {major:,.{digits}f}"
