"""Tests for tulving.semantic_index — written BEFORE implementation.

hnswlib runs REAL (core dependency, fast, local). Embeddings are never real
models: HashEmbedder is deterministic and offline; small local adapters cover
non-normalizing, exploding, and wrong-metric cases. Storage is the real
InMemoryBackend (it satisfies the reconciled label/meta/embedding surface).
The .hnsw cache lives under tmp_path; ephemeral (index_path=None) elsewhere.
"""

import logging
import re
import threading
from pathlib import Path
from typing import Any

import pytest

from tulving.adapters.embeddings import HashEmbedder
from tulving.adapters.storage import InMemoryBackend, unpack_embedding
from tulving.exceptions import VectorIndexError
from tulving.semantic_index import ReconcileReport, SemanticIndex

TS = "2026-07-03T12:00:00+00:00"

DIM = 32


def make_row(entry_id: str, content: str) -> dict[str, Any]:
    """A minimal valid MemoryEntry.to_dict()-shaped dict."""
    return {
        "id": entry_id,
        "content": content,
        "type": "fact",
        "source": {"agent_id": "agent-1", "step_id": None, "run_id": None, "workflow_name": None},
        "key": None,
        "tags": [],
        "relationships": [],
        "session_id": None,
        "base_importance": 0.5,
        "created_at": TS,
        "updated_at": TS,
        "last_accessed_at": TS,
        "access_count": 0,
        "archived": False,
        "archive_reason": None,
        "source_entry_ids": [],
        "pinned": False,
    }


def seed(backend: InMemoryBackend, items: dict[str, str]) -> None:
    """Create entry rows so set_embedding has something to attach to."""
    for entry_id, content in items.items():
        backend.create(make_row(entry_id, content))


def seed_and_add(index: SemanticIndex, backend: InMemoryBackend, items: dict[str, str]) -> None:
    seed(backend, items)
    for entry_id, content in items.items():
        index.add(entry_id, content)


def ten_items() -> dict[str, str]:
    return {f"e{i}": f"note number {i} about topic {i % 3}" for i in range(10)}


class SaltedHashEmbedder(HashEmbedder):
    """Same dimension as HashEmbedder, different identity AND different vectors."""

    @property
    def model_id(self) -> str:
        return f"tulving/salted-hash-v1-{self.dimension}"

    def embed(self, text: str) -> list[float]:
        return super().embed("SALT::" + text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class _NonNormalizingAdapter:
    """Returns HashEmbedder vectors scaled by 3.0 and declares normalizes=False."""

    def __init__(self, dimension: int = DIM) -> None:
        self._inner = HashEmbedder(dimension)

    @property
    def model_id(self) -> str:
        return f"tulving/test-nonnorm-v1-{self._inner.dimension}"

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    @property
    def distance_metric(self) -> str:
        return "cosine"

    @property
    def normalizes(self) -> bool:
        return False

    def embed(self, text: str) -> list[float]:
        return [3.0 * v for v in self._inner.embed(text)]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class _ExplodingAdapter:
    """Raises on any embed call; identity is configurable to slip past open()."""

    def __init__(self, dimension: int = DIM, model_id: str | None = None) -> None:
        self._dimension = dimension
        self._model_id = model_id or f"tulving/test-exploding-v1-{dimension}"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def distance_metric(self) -> str:
        return "cosine"

    @property
    def normalizes(self) -> bool:
        return True

    def embed(self, text: str) -> list[float]:
        raise RuntimeError("provider exploded")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("provider exploded")


class _L2Adapter(HashEmbedder):
    """Declares a non-cosine metric — must be refused at open()."""

    @property
    def model_id(self) -> str:
        return f"tulving/test-l2-v1-{self.dimension}"

    @property
    def distance_metric(self) -> str:
        return "l2"


@pytest.fixture
def backend() -> InMemoryBackend:
    return InMemoryBackend()


@pytest.fixture
def index(backend: InMemoryBackend) -> SemanticIndex:
    """An opened ephemeral index over a fresh backend."""
    idx = SemanticIndex(HashEmbedder(DIM), backend)
    idx.open()
    return idx


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestFailurePaths:
    def test_methods_raise_before_open(self, backend: InMemoryBackend) -> None:
        idx = SemanticIndex(HashEmbedder(DIM), backend)
        with pytest.raises(VectorIndexError, match="not open"):
            idx.add("e1", "text")
        with pytest.raises(VectorIndexError, match="not open"):
            idx.add_batch([("e1", "text")])
        with pytest.raises(VectorIndexError, match="not open"):
            idx.remove("e1")
        with pytest.raises(VectorIndexError, match="not open"):
            idx.search("query")
        with pytest.raises(VectorIndexError, match="not open"):
            idx.count()
        with pytest.raises(VectorIndexError, match="not open"):
            idx.reconcile()
        with pytest.raises(VectorIndexError, match="not open"):
            idx.maybe_compact()
        with pytest.raises(VectorIndexError, match="not open"):
            idx.flush()

    def test_close_before_open_is_safe(self, backend: InMemoryBackend) -> None:
        idx = SemanticIndex(HashEmbedder(DIM), backend)
        idx.close()  # never raises
        idx.close()  # idempotent

    def test_methods_raise_after_close(self, index: SemanticIndex) -> None:
        index.close()
        with pytest.raises(VectorIndexError, match="not open"):
            index.search("query")

    def test_constructor_rejects_bad_config(self, backend: InMemoryBackend) -> None:
        adapter = HashEmbedder(DIM)
        with pytest.raises(VectorIndexError):
            SemanticIndex(adapter, backend, overfetch_multiplier=0)
        with pytest.raises(VectorIndexError):
            SemanticIndex(adapter, backend, compaction_threshold=1.5)
        with pytest.raises(VectorIndexError):
            SemanticIndex(adapter, backend, compaction_threshold=0.0)
        with pytest.raises(VectorIndexError):
            SemanticIndex(adapter, backend, initial_capacity=0)
        with pytest.raises(VectorIndexError):
            SemanticIndex(adapter, backend, ef_construction=0)
        with pytest.raises(VectorIndexError):
            SemanticIndex(adapter, backend, m=0)
        with pytest.raises(VectorIndexError):
            SemanticIndex(adapter, backend, ef_search=0)

    def test_constructor_is_cheap(self, backend: InMemoryBackend) -> None:
        """D8: construction never touches the adapter, storage, or disk."""

        class _Untouchable:
            def __getattr__(self, name: str) -> Any:
                raise AssertionError(f"constructor touched adapter.{name}")

        SemanticIndex(_Untouchable(), backend, index_path=Path("does/not/exist.hnsw"))

    def test_non_cosine_metric_refused_at_open(self, backend: InMemoryBackend) -> None:
        idx = SemanticIndex(_L2Adapter(DIM), backend)
        with pytest.raises(VectorIndexError, match="cosine"):
            idx.open()

    def test_search_top_k_zero_rejected(self, index: SemanticIndex) -> None:
        with pytest.raises(VectorIndexError):
            index.search("query", top_k=0)

    def test_add_duplicate_live_id_rejected(
        self, index: SemanticIndex, backend: InMemoryBackend
    ) -> None:
        seed_and_add(index, backend, {"e1": "hello"})
        with pytest.raises(VectorIndexError, match="e1"):
            index.add("e1", "hello again")

    def test_remove_unknown_returns_false(self, index: SemanticIndex) -> None:
        assert index.remove("nope") is False

    def test_remove_twice_second_returns_false(
        self, index: SemanticIndex, backend: InMemoryBackend
    ) -> None:
        seed_and_add(index, backend, {"e1": "hello"})
        assert index.remove("e1") is True
        assert index.remove("e1") is False

    def test_adapter_failure_in_add_wrapped_and_db_unchanged(
        self, backend: InMemoryBackend
    ) -> None:
        idx = SemanticIndex(_ExplodingAdapter(DIM), backend)
        idx.open()
        seed(backend, {"e1": "hello"})
        with pytest.raises(VectorIndexError) as excinfo:
            idx.add("e1", "hello")
        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert backend.get_embedding("e1") is None  # embed precedes any write
        assert backend.load_labels() == []

    def test_adapter_failure_in_search_wrapped(self, backend: InMemoryBackend) -> None:
        # Build with a good adapter, then reopen with an exploder wearing the
        # SAME identity — open() passes, the query embed explodes.
        good = HashEmbedder(DIM)
        idx = SemanticIndex(good, backend)
        idx.open()
        seed_and_add(idx, backend, {"e1": "hello"})
        idx.close()
        evil = _ExplodingAdapter(DIM, model_id=good.model_id)
        idx2 = SemanticIndex(evil, backend)
        idx2.open()
        with pytest.raises(VectorIndexError) as excinfo:
            idx2.search("query")
        assert isinstance(excinfo.value.__cause__, RuntimeError)

    def test_adapter_failure_in_rebuild_re_embed_wrapped(self, backend: InMemoryBackend) -> None:
        idx = SemanticIndex(HashEmbedder(DIM), backend)
        idx.open()
        seed_and_add(idx, backend, {"e1": "hello"})
        idx.close()
        idx2 = SemanticIndex(_ExplodingAdapter(DIM), backend)
        with pytest.raises(VectorIndexError):
            idx2.open()
        with pytest.raises(VectorIndexError) as excinfo:
            idx2.rebuild(re_embed=True)
        assert isinstance(excinfo.value.__cause__, RuntimeError)

    def test_storage_failure_in_add_wrapped(self, index: SemanticIndex) -> None:
        """set_embedding for a nonexistent entry row is a StorageError — wrapped (D6)."""
        from tulving.exceptions import StorageError

        with pytest.raises(VectorIndexError) as excinfo:
            index.add("never-created", "text")  # no seed() — the row does not exist
        assert isinstance(excinfo.value.__cause__, StorageError)

    def test_flush_failure_wrapped(self, backend: InMemoryBackend, tmp_path: Path) -> None:
        """An unwritable cache path surfaces as VectorIndexError, never raw OSError."""
        blocked = tmp_path / "tulving.hnsw"
        blocked.mkdir()  # a directory where the file should go
        idx = SemanticIndex(HashEmbedder(DIM), backend, blocked)
        idx.open()
        seed_and_add(idx, backend, {"e1": "hello"})
        with pytest.raises(VectorIndexError):
            idx.flush()

    def test_index_error_identifier_never_appears(self) -> None:
        """D6: the bare identifier IndexError must not appear in the module."""
        import tulving.semantic_index as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        assert re.search(r"(?<![A-Za-z])IndexError", source) is None


# ---------------------------------------------------------------------------
# Mandatory regression — adapter swap forces rebuild (even same-dimension)
# ---------------------------------------------------------------------------


class TestAdapterSwapRegression:
    def test_same_dimension_model_swap_refused_then_rebuilt(
        self, backend: InMemoryBackend, tmp_path: Path
    ) -> None:
        path = tmp_path / "tulving.hnsw"
        old = HashEmbedder(DIM)
        idx = SemanticIndex(old, backend, path)
        idx.open()
        items = {"e1": "alpha text", "e2": "beta text", "e3": "gamma text"}
        seed_and_add(idx, backend, items)
        idx.flush()
        idx.close()
        old_blobs = {i: backend.get_embedding(i) for i in items}

        new = SaltedHashEmbedder(DIM)
        idx2 = SemanticIndex(new, backend, path)
        with pytest.raises(VectorIndexError) as excinfo:
            idx2.open()
        message = str(excinfo.value)
        assert old.model_id in message
        assert new.model_id in message
        assert "rebuild" in message

        idx2.rebuild(re_embed=True)
        meta = backend.get_meta()
        assert meta["embedding_model_id"] == new.model_id
        assert meta["embedding_dimension"] == DIM
        assert meta["distance_metric"] == "cosine"
        # BLOBs were re-embedded by the NEW adapter.
        for entry_id in items:
            assert backend.get_embedding(entry_id) != old_blobs[entry_id]

        idx3 = SemanticIndex(new, backend, path)
        idx3.open()
        results = idx3.search("alpha text", top_k=1)
        assert results[0][0] == "e1"

    def test_dimension_mismatch_refused(self, backend: InMemoryBackend) -> None:
        idx = SemanticIndex(HashEmbedder(32), backend)
        idx.open()
        idx.close()
        idx2 = SemanticIndex(HashEmbedder(16), backend)
        with pytest.raises(VectorIndexError, match="rebuild"):
            idx2.open()

    def test_fresh_store_writes_identity_to_meta(
        self, backend: InMemoryBackend, index: SemanticIndex
    ) -> None:
        meta = backend.get_meta()
        assert meta["embedding_model_id"] == HashEmbedder(DIM).model_id
        assert meta["embedding_dimension"] == DIM
        assert meta["distance_metric"] == "cosine"

    def test_matching_identity_reopens(self, backend: InMemoryBackend) -> None:
        idx = SemanticIndex(HashEmbedder(DIM), backend)
        idx.open()
        idx.close()
        idx2 = SemanticIndex(HashEmbedder(DIM), backend)
        idx2.open()  # no raise
        assert idx2.count() == 0


# ---------------------------------------------------------------------------
# Mandatory regression — archived vectors never surface
# ---------------------------------------------------------------------------


class TestArchivedExclusion:
    def test_archived_never_surface_full_scan(
        self, index: SemanticIndex, backend: InMemoryBackend
    ) -> None:
        items = ten_items()
        seed_and_add(index, backend, items)
        archived = ["e0", "e4", "e7"]
        for entry_id in archived:
            backend.update(entry_id, {"archived": True, "archive_reason": "superseded"})
        results = index.search("note number 4 about topic 1", top_k=10)
        ids = [entry_id for entry_id, _ in results]
        assert set(ids) == set(items) - set(archived)
        assert not set(ids) & set(archived)

    def test_two_nearest_archived_overfetch_recovers(
        self, index: SemanticIndex, backend: InMemoryBackend
    ) -> None:
        items = ten_items()
        seed_and_add(index, backend, items)
        query = "note number 4 about topic 1"
        embedder = HashEmbedder(DIM)
        qvec = embedder.embed(query)
        by_similarity = sorted(
            items, key=lambda i: dot(qvec, embedder.embed(items[i])), reverse=True
        )
        nearest_two = by_similarity[:2]
        for entry_id in nearest_two:
            backend.update(entry_id, {"archived": True, "archive_reason": "superseded"})
        results = index.search(query, top_k=2)
        assert len(results) == 2
        returned = {entry_id for entry_id, _ in results}
        assert not returned & set(nearest_two)


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


class TestBoundaryConditions:
    def test_search_empty_index_returns_empty(self, index: SemanticIndex) -> None:
        assert index.search("anything") == []

    def test_search_top_k_larger_than_live(
        self, index: SemanticIndex, backend: InMemoryBackend
    ) -> None:
        seed_and_add(index, backend, {"e1": "one", "e2": "two"})
        results = index.search("one", top_k=50)
        assert len(results) == 2

    def test_add_batch_empty_is_noop(self, index: SemanticIndex) -> None:
        assert index.add_batch([]) == []
        assert index.count() == 0

    def test_count_and_fraction_on_empty(self, index: SemanticIndex) -> None:
        assert index.count() == 0
        assert index.tombstone_fraction == 0.0
        assert index.needs_compaction is False

    def test_open_is_idempotent(self, index: SemanticIndex, backend: InMemoryBackend) -> None:
        seed_and_add(index, backend, {"e1": "one"})
        index.open()
        assert index.count() == 1
        assert index.search("one", top_k=1)[0][0] == "e1"


# ---------------------------------------------------------------------------
# hnswlib realities
# ---------------------------------------------------------------------------


class TestOverfetch:
    def test_underfill_returns_short_with_single_escalation(
        self,
        index: SemanticIndex,
        backend: InMemoryBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        items = {f"e{i}": f"document {i}" for i in range(20)}
        seed_and_add(index, backend, items)
        allowed = {"e3", "e11", "e17"}

        rounds: list[list[str]] = []
        original = index._filter_active

        def counting(ids: list[str]) -> set[str]:
            rounds.append(list(ids))
            return original(ids)

        monkeypatch.setattr(index, "_filter_active", counting)
        results = index.search("document 3", top_k=5, filter_fn=lambda i: i in allowed)
        assert {entry_id for entry_id, _ in results} == allowed
        assert len(results) == 3  # short results are normal, never loop
        assert len(rounds) <= 2  # one escalation at most

    def test_no_escalation_when_first_fetch_fills(
        self,
        index: SemanticIndex,
        backend: InMemoryBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        items = {f"e{i}": f"document {i}" for i in range(20)}
        seed_and_add(index, backend, items)
        rounds: list[list[str]] = []
        original = index._filter_active

        def counting(ids: list[str]) -> set[str]:
            rounds.append(list(ids))
            return original(ids)

        monkeypatch.setattr(index, "_filter_active", counting)
        results = index.search("document 3", top_k=3)
        assert len(results) == 3
        assert len(rounds) == 1


class TestTombstoneCompaction:
    def test_threshold_crossing_and_compaction(
        self, index: SemanticIndex, backend: InMemoryBackend
    ) -> None:
        items = {f"e{i}": f"entry text {i}" for i in range(12)}
        seed_and_add(index, backend, items)
        for entry_id in ("e0", "e3", "e6", "e9"):
            index.remove(entry_id)
        assert index.count() == 8
        assert index.tombstone_fraction == pytest.approx(4 / 12)
        assert index.needs_compaction is True

        assert index.maybe_compact() is True
        assert index.count() == 8
        assert index.tombstone_fraction == 0.0
        # Labels are dense 0..7 and all live after the rebuild.
        rows = backend.load_labels()
        assert [label for label, _, _ in rows] == list(range(8))
        assert all(not tombstoned for _, _, tombstoned in rows)
        # Every surviving UUID still resolves via search.
        for entry_id in set(items) - {"e0", "e3", "e6", "e9"}:
            assert index.search(items[entry_id], top_k=1)[0][0] == entry_id

    def test_maybe_compact_below_threshold_is_noop(
        self, index: SemanticIndex, backend: InMemoryBackend
    ) -> None:
        items = {f"e{i}": f"entry text {i}" for i in range(12)}
        seed_and_add(index, backend, items)
        index.remove("e0")  # 1/12 < 0.25
        assert index.needs_compaction is False
        assert index.maybe_compact() is False

    def test_removed_entries_do_not_surface(
        self, index: SemanticIndex, backend: InMemoryBackend
    ) -> None:
        items = {f"e{i}": f"entry text {i}" for i in range(6)}
        seed_and_add(index, backend, items)
        index.remove("e2")
        results = index.search("entry text 2", top_k=6)
        assert "e2" not in [entry_id for entry_id, _ in results]

    def test_re_add_after_remove(self, index: SemanticIndex, backend: InMemoryBackend) -> None:
        seed_and_add(index, backend, {"e1": "original text"})
        index.remove("e1")
        index.add("e1", "replacement text")  # store update path: remove + add
        assert index.count() == 1
        assert index.search("replacement text", top_k=1)[0][0] == "e1"


class TestCapacityGrowth:
    def test_geometric_growth_never_errors(
        self, backend: InMemoryBackend, caplog: pytest.LogCaptureFixture
    ) -> None:
        idx = SemanticIndex(HashEmbedder(DIM), backend, initial_capacity=4)
        idx.open()
        singles = {f"s{i}": f"single {i}" for i in range(10)}
        batch = {f"b{i}": f"batched {i}" for i in range(10)}
        with caplog.at_level(logging.INFO, logger="tulving.semantic_index"):
            seed_and_add(idx, backend, singles)
            seed(backend, batch)
            idx.add_batch(list(batch.items()))
        assert idx.count() == 20
        for entry_id, text in {**singles, **batch}.items():
            assert idx.search(text, top_k=1)[0][0] == entry_id
        assert idx._index.get_max_elements() >= 20
        assert any("capacity" in record.message.lower() for record in caplog.records)


class TestScores:
    def test_scores_clamped_and_self_match_near_one(
        self, index: SemanticIndex, backend: InMemoryBackend
    ) -> None:
        items = ten_items()
        seed_and_add(index, backend, items)
        results = index.search(items["e5"], top_k=10)
        assert results[0][0] == "e5"
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)
        for _, score in results:
            assert 0.0 <= score <= 1.0
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)


class TestNormalization:
    def test_non_normalizing_adapter_persists_unit_vectors(self, backend: InMemoryBackend) -> None:
        idx = SemanticIndex(_NonNormalizingAdapter(DIM), backend)
        idx.open()
        seed_and_add(idx, backend, {"e1": "hello world"})
        blob = backend.get_embedding("e1")
        assert blob is not None
        vector = unpack_embedding(blob)
        assert dot(vector, vector) == pytest.approx(1.0, abs=1e-5)

    def test_geometry_identical_to_normalized_twin(self) -> None:
        items = ten_items()
        backend_a, backend_b = InMemoryBackend(), InMemoryBackend()
        idx_a = SemanticIndex(HashEmbedder(DIM), backend_a)
        idx_b = SemanticIndex(_NonNormalizingAdapter(DIM), backend_b)
        idx_a.open()
        idx_b.open()
        seed_and_add(idx_a, backend_a, items)
        seed_and_add(idx_b, backend_b, items)
        results_a = idx_a.search("note number 4 about topic 1", top_k=5)
        results_b = idx_b.search("note number 4 about topic 1", top_k=5)
        assert [entry_id for entry_id, _ in results_a] == [entry_id for entry_id, _ in results_b]
        for (_, score_a), (_, score_b) in zip(results_a, results_b, strict=True):
            assert score_a == pytest.approx(score_b, abs=1e-5)


# ---------------------------------------------------------------------------
# ADR-015 — cache semantics & reconcile
# ---------------------------------------------------------------------------


class TestCacheSemantics:
    def test_flush_close_reopen_loads_from_file(
        self, backend: InMemoryBackend, tmp_path: Path
    ) -> None:
        path = tmp_path / "tulving.hnsw"
        idx = SemanticIndex(HashEmbedder(DIM), backend, path)
        idx.open()
        items = ten_items()
        seed_and_add(idx, backend, items)
        baseline = idx.search("note number 4 about topic 1", top_k=5)
        idx.close()  # close flushes
        assert path.exists()
        assert not list(tmp_path.glob("*.tmp"))  # atomic write leaves no temp file

        idx2 = SemanticIndex(HashEmbedder(DIM), backend, path)
        idx2.open()
        assert idx2.search("note number 4 about topic 1", top_k=5) == baseline
        report = idx2.reconcile()
        assert report.full_rebuild is False
        assert report.added_from_blobs == 0
        assert report.dropped_orphans == 0

    def test_deleted_cache_file_rebuilds_from_blobs(
        self, backend: InMemoryBackend, tmp_path: Path
    ) -> None:
        path = tmp_path / "tulving.hnsw"
        idx = SemanticIndex(HashEmbedder(DIM), backend, path)
        idx.open()
        items = ten_items()
        seed_and_add(idx, backend, items)
        baseline = idx.search("note number 4 about topic 1", top_k=5)
        idx.close()
        path.unlink()

        idx2 = SemanticIndex(HashEmbedder(DIM), backend, path)
        idx2.open()
        report = idx2.reconcile()
        assert report.full_rebuild is True
        results = idx2.search("note number 4 about topic 1", top_k=5)
        assert [entry_id for entry_id, _ in results] == [entry_id for entry_id, _ in baseline]
        for (_, score), (_, base_score) in zip(results, baseline, strict=True):
            assert score == pytest.approx(base_score, abs=1e-6)

    def test_corrupt_cache_file_rebuilds_with_warning(
        self, backend: InMemoryBackend, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "tulving.hnsw"
        idx = SemanticIndex(HashEmbedder(DIM), backend, path)
        idx.open()
        items = ten_items()
        seed_and_add(idx, backend, items)
        baseline = idx.search("note number 4 about topic 1", top_k=5)
        idx.close()
        path.write_bytes(b"this is not an hnsw index at all")

        idx2 = SemanticIndex(HashEmbedder(DIM), backend, path)
        with caplog.at_level(logging.WARNING, logger="tulving.semantic_index"):
            idx2.open()  # NEVER a user-facing error (cache semantics)
        assert any(record.levelno == logging.WARNING for record in caplog.records)
        results = idx2.search("note number 4 about topic 1", top_k=5)
        assert [entry_id for entry_id, _ in results] == [entry_id for entry_id, _ in baseline]


class TestReconcile:
    def test_crash_after_db_write_readded_from_blob(
        self, backend: InMemoryBackend, tmp_path: Path
    ) -> None:
        path = tmp_path / "tulving.hnsw"
        idx = SemanticIndex(HashEmbedder(DIM), backend, path)
        idx.open()
        seed_and_add(idx, backend, {"e1": "one", "e2": "two"})
        idx.close()
        # Simulate a crash between the DB write and the index insert: BLOB and
        # label row exist, the vector never reached the graph.
        from tulving.adapters.storage import pack_embedding

        seed(backend, {"e3": "three"})
        backend.set_embedding("e3", pack_embedding(HashEmbedder(DIM).embed("three")))
        backend.insert_labels([(2, "e3")])

        idx2 = SemanticIndex(HashEmbedder(DIM), backend, path)
        idx2.open()
        report = idx2.reconcile()
        assert report.added_from_blobs == 1
        assert idx2.search("three", top_k=1)[0][0] == "e3"

    def test_blob_without_label_row_readded(self, backend: InMemoryBackend, tmp_path: Path) -> None:
        path = tmp_path / "tulving.hnsw"
        idx = SemanticIndex(HashEmbedder(DIM), backend, path)
        idx.open()
        seed_and_add(idx, backend, {"e1": "one"})
        idx.close()
        from tulving.adapters.storage import pack_embedding

        seed(backend, {"e2": "two"})
        backend.set_embedding("e2", pack_embedding(HashEmbedder(DIM).embed("two")))

        idx2 = SemanticIndex(HashEmbedder(DIM), backend, path)
        idx2.open()
        report = idx2.reconcile()
        assert report.added_from_blobs == 1
        assert idx2.search("two", top_k=1)[0][0] == "e2"

    def test_orphan_label_dropped(self, backend: InMemoryBackend, tmp_path: Path) -> None:
        path = tmp_path / "tulving.hnsw"
        idx = SemanticIndex(HashEmbedder(DIM), backend, path)
        idx.open()
        seed_and_add(idx, backend, {"e1": "one"})
        idx.close()
        backend.insert_labels([(1, "ghost")])  # mapping row with no DB embedding

        idx2 = SemanticIndex(HashEmbedder(DIM), backend, path)
        idx2.open()
        report = idx2.reconcile()
        assert report.dropped_orphans == 1
        assert idx2.search("one", top_k=5) == [("e1", pytest.approx(1.0, abs=1e-5))]

    def test_reconcile_idempotent(self, backend: InMemoryBackend, tmp_path: Path) -> None:
        path = tmp_path / "tulving.hnsw"
        idx = SemanticIndex(HashEmbedder(DIM), backend, path)
        idx.open()
        seed_and_add(idx, backend, ten_items())
        idx.close()
        path.unlink()  # force a full rebuild on the next open

        idx2 = SemanticIndex(HashEmbedder(DIM), backend, path)
        idx2.open()
        first = idx2.reconcile()
        assert first.full_rebuild is True
        second = idx2.reconcile()
        assert second == ReconcileReport(
            added_from_blobs=0,
            dropped_orphans=0,
            full_rebuild=False,
            compacted=False,
            tombstone_fraction=0.0,
        )

    def test_reconcile_triggers_compaction_over_threshold(self, backend: InMemoryBackend) -> None:
        idx = SemanticIndex(HashEmbedder(DIM), backend)
        idx.open()
        items = {f"e{i}": f"entry {i}" for i in range(12)}
        seed_and_add(idx, backend, items)
        for entry_id in ("e0", "e3", "e6", "e9"):
            idx.remove(entry_id)
        report = idx.reconcile()
        assert report.compacted is True
        assert idx.tombstone_fraction == 0.0


class TestRebuildCrashWindow:
    def test_stale_cache_discarded_before_new_mapping_commits(
        self,
        backend: InMemoryBackend,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """DB-HIGH-1 regression: a crash between replace_labels and flush()
        must never leave the OLD graph on disk under the NEW mapping — the
        cache file is unlinked BEFORE the mapping commit, so every crash
        ordering leaves a missing cache that rebuilds cleanly from BLOBs."""
        path = tmp_path / "tulving.hnsw"
        items = ten_items()
        idx = SemanticIndex(HashEmbedder(DIM), backend, path)
        idx.open()
        seed_and_add(idx, backend, items)
        idx.flush()
        assert path.exists()
        # Simulate the crash: rebuild runs, but the trailing flush never does.
        monkeypatch.setattr(idx, "flush", lambda: None)
        idx.rebuild()
        assert not path.exists()  # stale cache was gone before replace_labels
        assert not list(tmp_path.glob("*.tmp"))
        # Recovery: next open rebuilds from BLOBs; results are correct.
        idx2 = SemanticIndex(HashEmbedder(DIM), backend, path)
        idx2.open()
        report = idx2.reconcile()
        assert report.full_rebuild is True
        assert idx2.search(items["e4"], top_k=1)[0][0] == "e4"


class TestReadOnly:
    def _populate(
        self, backend: InMemoryBackend, path: Path
    ) -> tuple[dict[str, str], list[tuple[str, float]]]:
        """Writer pass: populate, flush, close; return items + a baseline search."""
        items = ten_items()
        writer = SemanticIndex(HashEmbedder(DIM), backend, path)
        writer.open()
        seed_and_add(writer, backend, items)
        baseline = writer.search("note number 4 about topic 1", top_k=5)
        writer.close()
        return items, baseline

    def test_clean_cache_serves_searches(self, backend: InMemoryBackend, tmp_path: Path) -> None:
        path = tmp_path / "tulving.hnsw"
        _, baseline = self._populate(backend, path)
        reader = SemanticIndex(HashEmbedder(DIM), backend, path, read_only=True)
        reader.open()
        assert reader.search("note number 4 about topic 1", top_k=5) == baseline
        assert reader.count() == 10

    def test_missing_cache_disables_without_writing(
        self, backend: InMemoryBackend, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "tulving.hnsw"
        self._populate(backend, path)
        path.unlink()
        meta_before = backend.get_meta()
        labels_before = backend.load_labels()
        reader = SemanticIndex(HashEmbedder(DIM), backend, path, read_only=True)
        with caplog.at_level(logging.WARNING, logger="tulving.semantic_index"):
            reader.open()  # never raises, never rebuilds
        assert any(record.levelno == logging.WARNING for record in caplog.records)
        assert reader.search("anything") == []  # documented disabled behavior
        assert reader.count() == 0
        assert not path.exists()  # no .hnsw created
        assert not list(tmp_path.glob("*.tmp"))
        assert backend.get_meta() == meta_before  # no meta writes
        assert backend.load_labels() == labels_before  # no label writes

    def test_corrupt_cache_disables_without_writing(
        self, backend: InMemoryBackend, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "tulving.hnsw"
        self._populate(backend, path)
        path.write_bytes(b"garbage")
        reader = SemanticIndex(HashEmbedder(DIM), backend, path, read_only=True)
        with caplog.at_level(logging.WARNING, logger="tulving.semantic_index"):
            reader.open()
        assert reader.search("anything") == []
        assert path.read_bytes() == b"garbage"  # untouched — read-only never repairs

    def test_stale_cache_disables(self, backend: InMemoryBackend, tmp_path: Path) -> None:
        """A live mapping label missing from the graph = divergent cache."""
        path = tmp_path / "tulving.hnsw"
        self._populate(backend, path)
        from tulving.adapters.storage import pack_embedding

        seed(backend, {"extra": "extra text"})
        backend.set_embedding("extra", pack_embedding(HashEmbedder(DIM).embed("extra text")))
        backend.insert_labels([(10, "extra")])  # in the mapping, not in the cache
        reader = SemanticIndex(HashEmbedder(DIM), backend, path, read_only=True)
        reader.open()
        assert reader.search("extra text") == []
        assert reader.count() == 0

    def test_identity_mismatch_disables_and_keeps_meta(
        self, backend: InMemoryBackend, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "tulving.hnsw"
        self._populate(backend, path)
        meta_before = backend.get_meta()
        reader = SemanticIndex(SaltedHashEmbedder(DIM), backend, path, read_only=True)
        with caplog.at_level(logging.WARNING, logger="tulving.semantic_index"):
            reader.open()  # disabled, not raised (read-only never blocks startup)
        assert any(record.levelno == logging.WARNING for record in caplog.records)
        assert reader.search("anything") == []
        assert backend.get_meta() == meta_before  # identity NOT overwritten

    def test_fresh_store_read_only_is_empty_not_disabled(self, backend: InMemoryBackend) -> None:
        reader = SemanticIndex(HashEmbedder(DIM), backend, read_only=True)
        reader.open()
        assert reader.search("anything") == []
        assert reader.count() == 0
        meta = backend.get_meta()
        assert meta["embedding_model_id"] is None  # fresh store: nothing written

    def test_write_methods_raise_in_read_only(
        self, backend: InMemoryBackend, tmp_path: Path
    ) -> None:
        path = tmp_path / "tulving.hnsw"
        self._populate(backend, path)
        reader = SemanticIndex(HashEmbedder(DIM), backend, path, read_only=True)
        reader.open()
        with pytest.raises(VectorIndexError, match="read-only"):
            reader.add("e-new", "text")
        with pytest.raises(VectorIndexError, match="read-only"):
            reader.add_batch([("e-new", "text")])
        with pytest.raises(VectorIndexError, match="read-only"):
            reader.remove("e1")
        with pytest.raises(VectorIndexError, match="read-only"):
            reader.rebuild()
        with pytest.raises(VectorIndexError, match="read-only"):
            reader.rebuild(re_embed=True)
        with pytest.raises(VectorIndexError, match="read-only"):
            reader.flush()
        with pytest.raises(VectorIndexError, match="read-only"):
            reader.maybe_compact()
        with pytest.raises(VectorIndexError, match="read-only"):
            reader.reconcile()
        reader.close()  # close never writes and never raises in read-only


class TestRebuildPagination:
    def test_re_embed_rebuild_crosses_list_page_boundary(
        self,
        backend: InMemoryBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """rebuild(re_embed=True) must pick up every active entry across
        multiple storage.list() pages (CR-L5; page size shrunk to stay fast)."""
        import tulving.semantic_index as module

        monkeypatch.setattr(module, "_LIST_PAGE", 7)
        items = {f"e{i:03d}": f"paginated entry {i}" for i in range(20)}  # 3 pages of 7
        idx = SemanticIndex(HashEmbedder(DIM), backend)
        idx.open()
        seed(backend, items)
        idx.add_batch(list(items.items()))
        idx.rebuild(re_embed=True)
        assert idx.count() == 20
        rows = backend.load_labels()
        assert [label for label, _, _ in rows] == list(range(20))  # dense relabel
        for entry_id in ("e000", "e007", "e013", "e019"):  # spans all pages
            assert idx.search(items[entry_id], top_k=1)[0][0] == entry_id


class TestBatchParity:
    def test_batch_blobs_byte_identical_to_single_adds(self) -> None:
        texts = {"e1": "alpha", "e2": "beta", "e3": "gamma"}
        backend_single, backend_batch = InMemoryBackend(), InMemoryBackend()
        idx_single = SemanticIndex(HashEmbedder(DIM), backend_single)
        idx_batch = SemanticIndex(HashEmbedder(DIM), backend_batch)
        idx_single.open()
        idx_batch.open()
        seed_and_add(idx_single, backend_single, texts)
        seed(backend_batch, texts)
        vectors = idx_batch.add_batch(list(texts.items()))
        assert len(vectors) == 3
        for entry_id in texts:
            assert backend_single.get_embedding(entry_id) == backend_batch.get_embedding(entry_id)

    def test_batch_created_entries_are_searchable(
        self, index: SemanticIndex, backend: InMemoryBackend
    ) -> None:
        texts = {"e1": "alpha content", "e2": "beta content", "e3": "gamma content"}
        seed(backend, texts)
        index.add_batch(list(texts.items()))
        assert index.count() == 3
        for entry_id, text in texts.items():
            assert index.search(text, top_k=1)[0][0] == entry_id

    def test_batch_duplicate_live_id_rejected_before_any_write(
        self, index: SemanticIndex, backend: InMemoryBackend
    ) -> None:
        seed_and_add(index, backend, {"e1": "existing"})
        seed(backend, {"e2": "new"})
        with pytest.raises(VectorIndexError):
            index.add_batch([("e2", "new"), ("e1", "existing")])
        assert index.count() == 1
        assert backend.get_embedding("e2") is None


# ---------------------------------------------------------------------------
# Concurrency smoke (exclusive-lock correctness, not performance)
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_parallel_adds_and_searches(
        self, index: SemanticIndex, backend: InMemoryBackend
    ) -> None:
        threads = 8
        per_thread = 5
        errors: list[BaseException] = []

        def worker(worker_id: int) -> None:
            try:
                for i in range(per_thread):
                    entry_id = f"w{worker_id}-e{i}"
                    backend.create(make_row(entry_id, f"worker {worker_id} item {i}"))
                    index.add(entry_id, f"worker {worker_id} item {i}")
                    index.search(f"worker {worker_id} item {i}", top_k=3)
            except BaseException as exc:  # smoke test collects everything
                errors.append(exc)

        pool = [threading.Thread(target=worker, args=(w,)) for w in range(threads)]
        for thread in pool:
            thread.start()
        for thread in pool:
            thread.join()
        assert errors == []
        assert index.count() == threads * per_thread
        for worker_id in range(threads):
            for i in range(per_thread):
                text = f"worker {worker_id} item {i}"
                assert index.search(text, top_k=1)[0][0] == f"w{worker_id}-e{i}"
