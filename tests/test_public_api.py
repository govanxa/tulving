"""Tests for tulving/__init__.py — the public API surface (QA addition).

The export inventory is a compatibility contract: removing or renaming a
name in ``__all__`` is a breaking change, not a refactor. These tests pin
the surface exactly so any drift fails loudly.
"""

import importlib.metadata

import tulving

EXPECTED_EXPORTS = {
    # enums
    "ArchiveReason",
    "MatchType",
    "MemoryType",
    "SessionStatus",
    # entry model
    "MemoryEntry",
    "Relationship",
    "SourceInfo",
    # exceptions (full D6 hierarchy)
    "ConfigError",
    "MemoryStoreError",
    "ScopeError",
    "SecurityError",
    "StorageError",
    "TulvingError",
    "VectorIndexError",
    # metadata
    "__version__",
}


class TestBoundaryConditions:
    """Exact export inventory — no more, no fewer."""

    def test_all_exact_inventory(self) -> None:
        assert set(tulving.__all__) == EXPECTED_EXPORTS

    def test_all_is_sorted_and_duplicate_free(self) -> None:
        assert list(tulving.__all__) == sorted(tulving.__all__)
        assert len(tulving.__all__) == len(set(tulving.__all__))

    def test_security_internals_not_exported(self) -> None:
        """security.py is internal (its own docstring): never in __all__."""
        assert "security" not in tulving.__all__
        assert "REDACTED" not in tulving.__all__
        assert "redact_text" not in tulving.__all__


class TestBasicBehavior:
    """Every advertised name resolves; version metadata is consistent."""

    def test_every_export_is_an_attribute(self) -> None:
        for name in tulving.__all__:
            assert getattr(tulving, name) is not None

    def test_star_import_exposes_exactly_all(self) -> None:
        namespace: dict[str, object] = {}
        exec("from tulving import *", namespace)  # deliberate: pins the star-import surface
        imported = {name for name in namespace if not name.startswith("__")} | (
            {"__version__"} if "__version__" in namespace else set()
        )
        assert imported == EXPECTED_EXPORTS

    def test_version_matches_installed_metadata(self) -> None:
        assert tulving.__version__ == importlib.metadata.version("tulving")

    def test_exceptions_reachable_from_package_root(self) -> None:
        """Users catch tulving.TulvingError without importing submodules."""
        assert issubclass(tulving.SecurityError, tulving.TulvingError)
        assert issubclass(tulving.ConfigError, tulving.TulvingError)

    def test_entry_constructible_from_root_exports_only(self) -> None:
        entry = tulving.MemoryEntry(
            id="root-1",
            content="constructed via package root",
            type=tulving.MemoryType.FACT,
            source=tulving.SourceInfo(agent_id="agent-1"),
        )
        assert entry.type is tulving.MemoryType.FACT
