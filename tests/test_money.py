import pytest

from engine.money import format_minor, to_minor


def test_parses_major_units_into_minor_units() -> None:
    assert to_minor("850.00", "EUR") == 85_000
    assert to_minor("850", "EUR") == 85_000
    assert to_minor(850, "EUR") == 85_000
    assert to_minor("0.1", "eur") == 10


def test_zero_decimal_currency() -> None:
    assert to_minor("1000", "JPY") == 1_000


@pytest.mark.parametrize("bad", ["-5", "0", "abc", "NaN", "Infinity", "1.234", ""])
def test_rejects_bad_amounts(bad: str) -> None:
    with pytest.raises(ValueError):
        to_minor(bad, "EUR")


def test_formats_minor_units() -> None:
    assert format_minor(123_456, "EUR") == "EUR 1,234.56"
    assert format_minor(1_000, "JPY") == "JPY 1,000"
