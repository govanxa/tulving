"""Shared test fixtures for Tulving."""

from datetime import UTC, datetime, timedelta

import pytest

CLOCK_START = datetime(2026, 7, 3, 12, 0, 0, tzinfo=UTC)


class FakeClock:
    """Steppable injected clock — decay/touch tests never sleep."""

    def __init__(self, start: datetime = CLOCK_START) -> None:
        self.current = start

    def __call__(self) -> datetime:
        return self.current

    def advance(self, **kwargs: float) -> None:
        """Advance by timedelta(**kwargs), e.g. advance(hours=2)."""
        self.current += timedelta(**kwargs)


@pytest.fixture
def fake_clock() -> FakeClock:
    """A FakeClock starting at a fixed, tz-aware UTC instant."""
    return FakeClock()
