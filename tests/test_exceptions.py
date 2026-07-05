"""Tests for tulving.exceptions — written BEFORE implementation."""

import builtins

import pytest

import tulving.exceptions
from tulving.exceptions import (
    ConfigError,
    MemoryStoreError,
    ScopeError,
    SecurityError,
    StorageError,
    TulvingError,
    VectorIndexError,
)

ALL_SUBCLASSES = (
    MemoryStoreError,
    StorageError,
    VectorIndexError,
    ScopeError,
    SecurityError,
    ConfigError,
)


class TestFailurePaths:
    """Builtin shadowing — the audit's core D6 fix."""

    def test_no_memory_error_defined(self) -> None:
        assert not hasattr(tulving.exceptions, "MemoryError") or (
            tulving.exceptions.MemoryError is builtins.MemoryError
        )
        assert "MemoryError" not in tulving.exceptions.__dict__

    def test_no_index_error_defined(self) -> None:
        assert "IndexError" not in tulving.exceptions.__dict__

    def test_no_builtin_shadowing_at_all(self) -> None:
        """No Tulving-defined symbol may shadow any builtin name."""
        builtin_names = {n for n in dir(builtins) if not n.startswith("_")}
        module_names = {n for n, obj in vars(tulving.exceptions).items() if not n.startswith("_")}
        assert module_names & builtin_names == set()

    def test_tulving_error_does_not_catch_builtin_memory_error(self) -> None:
        with pytest.raises(MemoryError):
            try:
                raise MemoryError("out of memory")
            except TulvingError:  # pragma: no cover - must not be reached
                pytest.fail("TulvingError caught builtin MemoryError")

    def test_tulving_error_does_not_catch_builtin_index_error(self) -> None:
        with pytest.raises(IndexError):
            try:
                raise IndexError("list index out of range")
            except TulvingError:  # pragma: no cover - must not be reached
                pytest.fail("TulvingError caught builtin IndexError")


class TestBoundaryConditions:
    """Sibling isolation and hierarchy shape."""

    def test_scope_error_is_not_security_error(self) -> None:
        assert not issubclass(ScopeError, SecurityError)

    def test_security_error_is_not_scope_error(self) -> None:
        assert not issubclass(SecurityError, ScopeError)

    def test_all_subclasses_inherit_directly_from_tulving_error(self) -> None:
        for exc in ALL_SUBCLASSES:
            assert exc.__bases__ == (TulvingError,)

    def test_tulving_error_is_exception(self) -> None:
        assert issubclass(TulvingError, Exception)

    def test_tulving_error_catchable_by_except_exception(self) -> None:
        with pytest.raises(Exception, match="boom"):
            raise TulvingError("boom")


class TestBasicBehavior:
    """Message round-trip and catch-all contract."""

    def test_message_round_trip(self) -> None:
        assert str(StorageError("disk full")) == "disk full"

    def test_construction_with_no_args_is_legal(self) -> None:
        for exc in (TulvingError, *ALL_SUBCLASSES):
            instance = exc()
            assert isinstance(instance, TulvingError)

    def test_each_subclass_caught_by_tulving_error(self) -> None:
        for exc in ALL_SUBCLASSES:
            with pytest.raises(TulvingError):
                raise exc("caught")
