"""Tests for tulving.export.formats — written BEFORE implementation (step 15).

Covers the JSON round-trip: security/path validation, the envelope schema,
untrusted-input parsing, ID remapping, D1 dedup semantics, re-embed/reuse,
atomicity, and the full round-trip contract (§TDD Test Plan 1-30).

Uses an ``InMemoryBackend``-backed store and a deterministic recording
embedder — no real models, no network.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from tulving.adapters.storage import InMemoryBackend, SQLiteBackend, pack_embedding
from tulving.entry import MemoryEntry, Relationship, SourceInfo
from tulving.enums import ArchiveReason, MemoryType
from tulving.exceptions import MemoryStoreError, SecurityError, StorageError
from tulving.export import ImportReport as ReexportedImportReport
from tulving.export.formats import ImportReport, MemoryExporter, MemoryImporter
from tulving.security import compile_explicit_patterns
from tulving.store import MemoryStore

# --------------------------------------------------------------------------- #
# Test doubles & helpers
# --------------------------------------------------------------------------- #


class RecordingEmbedder:
    """Deterministic embedder that records every ``embed_batch`` call.

    Equal text -> equal vector. Satisfies the exporter/importer's local
    ``_EmbedderLike`` protocol structurally (embed_batch + the three
    identity properties).
    """

    def __init__(
        self,
        *,
        dimension: int = 4,
        model_id: str = "fake-model-v1",
        distance_metric: str = "cosine",
    ) -> None:
        self._dimension = dimension
        self._model_id = model_id
        self._distance_metric = distance_metric
        self.batch_calls: list[list[str]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def distance_metric(self) -> str:
        return self._distance_metric

    def _vector(self, text: str) -> list[float]:
        seed = sum(ord(ch) for ch in text) or 1
        return [((seed * (i + 1)) % 97) / 97.0 for i in range(self._dimension)]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls.append(list(texts))
        return [self._vector(text) for text in texts]


class RaisingEmbedder(RecordingEmbedder):
    """Embedder whose ``embed_batch`` always raises (pre-transaction failure)."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding backend exploded")


def make_store() -> tuple[MemoryStore, InMemoryBackend]:
    backend = InMemoryBackend()
    return MemoryStore(backend), backend


def make_sqlite_store(tmp_path: Path, name: str = "db") -> tuple[MemoryStore, SQLiteBackend]:
    """A real SQLite-backed store (WAL, BEGIN IMMEDIATE) for on-disk atomicity."""
    backend = SQLiteBackend(tmp_path / f"{name}.db")
    return MemoryStore(backend), backend


def seed_meta(backend: InMemoryBackend, embedder: RecordingEmbedder) -> None:
    backend.set_meta(
        {
            "embedding_model_id": embedder.model_id,
            "embedding_dimension": embedder.dimension,
            "distance_metric": embedder.distance_metric,
        }
    )


def by_content(store: MemoryStore, *, include_archived: bool = True) -> dict[str, MemoryEntry]:
    entries = store.list(include_archived=include_archived, limit=1000)
    return {entry.content: entry for entry in entries}


def make_entry(**overrides: Any) -> MemoryEntry:
    """A fully-specified entry usable with ``store.restore`` in tests."""
    base: dict[str, Any] = {
        "id": "seed-id",
        "content": "seed content",
        "type": MemoryType.FACT,
        "source": SourceInfo(agent_id="agent-x"),
    }
    base.update(overrides)
    return MemoryEntry(**base)


# --------------------------------------------------------------------------- #
# TestSecurity — path validation & redaction (100% branch coverage target)
# --------------------------------------------------------------------------- #


class TestSecurity:
    def test_reexport_is_same_object(self) -> None:
        """The package re-export is the module class (import contract)."""
        assert ReexportedImportReport is ImportReport

    def test_traversal_leaf_rejected(self, tmp_path: Path) -> None:
        store, _ = make_store()
        exporter = MemoryExporter(store, allowed_root=tmp_path)
        with pytest.raises(SecurityError):
            exporter.to_json(str(tmp_path / ".." / "evil.json"))

    def test_directory_outside_root_rejected(self, tmp_path: Path, monkeypatch) -> None:
        """A relative dir resolving outside the allowed root is refused."""
        store, _ = make_store()
        root = tmp_path / "root"
        root.mkdir()
        monkeypatch.chdir(tmp_path)  # cwd is the PARENT of the allowed root
        exporter = MemoryExporter(store, allowed_root=root)
        with pytest.raises(SecurityError):
            exporter.to_json("a/b.json")  # resolves under cwd, not root

    @pytest.mark.parametrize(
        "leaf",
        ["bad name.json", "dots.in.name.json", "con.json", "x" * 129 + ".json"],
    )
    def test_leaf_whitelist_rejects(self, tmp_path: Path, leaf: str) -> None:
        store, _ = make_store()
        exporter = MemoryExporter(store, allowed_root=tmp_path)
        with pytest.raises(SecurityError):
            exporter.to_json(str(tmp_path / leaf))

    def test_leaf_and_bare_name_same_file(self, tmp_path: Path) -> None:
        store, _ = make_store()
        store.create(content="c", type=MemoryType.FACT, source=SourceInfo(agent_id="a"))
        exporter = MemoryExporter(store, allowed_root=tmp_path)
        exporter.to_json(str(tmp_path / "backup.json"))
        assert (tmp_path / "backup.json").is_file()
        (tmp_path / "backup.json").unlink()
        exporter.to_json(str(tmp_path / "backup"))  # suffix appended after validation
        assert (tmp_path / "backup.json").is_file()

    def test_blank_root_refused(self, tmp_path: Path) -> None:
        store, _ = make_store()
        exporter = MemoryExporter(store, allowed_root="")
        with pytest.raises(SecurityError):
            exporter.to_json(str(tmp_path / "backup.json"))

    def test_default_root_is_cwd(self, tmp_path: Path, monkeypatch) -> None:
        store, _ = make_store()
        store.create(content="c", type=MemoryType.FACT, source=SourceInfo(agent_id="a"))
        monkeypatch.chdir(tmp_path)
        exporter = MemoryExporter(store)  # allowed_root defaults to cwd == tmp_path
        exporter.to_json("backup.json")
        assert (tmp_path / "backup.json").is_file()
        outside = tmp_path.parent / "outside_dir"
        outside.mkdir(exist_ok=True)
        with pytest.raises(SecurityError):
            exporter.to_json(str(outside / "backup.json"))

    def test_redaction_default_on(self, tmp_path: Path) -> None:
        """Default-sensitive key + secret-SHAPED content -> whole-masked
        (v0.2 softening, D-v02-7: the key alone is no longer enough).

        The secret is embedded inside an unrelated sentence and the
        SURROUNDING sentence is asserted absent too (test review MAJOR):
        surgical redact_text alone would strip only the secret substring and
        leave the sentence text intact, so this can only pass via true
        whole-body masking."""
        secret_value = "sk-" + "y" * 24
        sentence = f"the rotation doc says {secret_value} expires monthly"
        store, _ = make_store()
        store.create(
            content="the token is sk-abcdefghijklmnopqrstuvwxyz012345",
            type=MemoryType.FACT,
            source=SourceInfo(agent_id="a"),
        )
        store.create(
            content=sentence,
            type=MemoryType.FACT,
            key="service_api_key",
            source=SourceInfo(agent_id="a"),
        )
        exporter = MemoryExporter(store, allowed_root=tmp_path)
        path = tmp_path / "out.json"
        exporter.to_json(str(path))
        raw = path.read_text(encoding="utf-8")
        assert "sk-abcdefghijklmnopqrstuvwxyz012345" not in raw
        assert secret_value not in raw
        assert "rotation doc says" not in raw
        assert "expires monthly" not in raw
        assert "[REDACTED]" in raw
        envelope = json.loads(raw)
        assert envelope["redacted"] is True
        keyed = next(e for e in envelope["entries"] if e["key"] == "service_api_key")
        assert keyed["content"] == "[REDACTED]"

    def test_default_sensitive_key_prose_passes_through_and_keeps_embedding(
        self, tmp_path: Path
    ) -> None:
        """The reported bug fix: prose under a default-sensitive-named key is
        no longer whole-masked, and (vector-consistency rule) its embedding
        is now RETAINED because the exported content is verbatim."""
        store, backend = make_store()
        embedder = RecordingEmbedder()
        seed_meta(backend, embedder)
        prose = store.create(
            content="auth token TTL is 15 min",
            type=MemoryType.FACT,
            key="fact:auth-ttl",
            source=SourceInfo(agent_id="a"),
            embedding=pack_embedding(embedder._vector("prose")),
        )
        secret_value = "sk-" + "z" * 24
        secret = store.create(
            content=secret_value,
            type=MemoryType.FACT,
            key="api_key:x",
            source=SourceInfo(agent_id="a"),
            embedding=pack_embedding(embedder._vector("secret")),
        )
        exporter = MemoryExporter(store, allowed_root=tmp_path)
        path = tmp_path / "out.json"
        exporter.to_json(str(path), include_embeddings=True)
        envelope = json.loads(path.read_text(encoding="utf-8"))
        entries = {e["id"]: e for e in envelope["entries"]}
        assert entries[prose.id]["content"] == "auth token TTL is 15 min"
        assert entries[prose.id]["embedding"] is not None
        assert entries[secret.id]["content"] == "[REDACTED]"
        assert entries[secret.id]["embedding"] is None

    def test_user_declared_key_overrides_default_overlap(self, tmp_path: Path) -> None:
        """D-v02-7 Q3 (mandatory): a user-declared pattern masks unconditionally
        even when the key ALSO matches a built-in default, and even for prose."""
        store, _ = make_store()
        store.create(
            content="rotate quarterly, no value here",
            type=MemoryType.FACT,
            key="auth-prod-token",
            source=SourceInfo(agent_id="a"),
        )
        explicit = compile_explicit_patterns(["auth-prod"])
        exporter = MemoryExporter(store, allowed_root=tmp_path, explicit_key_patterns=explicit)
        path = tmp_path / "out.json"
        exporter.to_json(str(path))
        envelope = json.loads(path.read_text(encoding="utf-8"))
        keyed = next(e for e in envelope["entries"] if e["key"] == "auth-prod-token")
        assert keyed["content"] == "[REDACTED]"

    def test_opt_out_exports_verbatim(self, tmp_path: Path) -> None:
        store, _ = make_store()
        store.create(
            content="the token is sk-abcdefghijklmnopqrstuvwxyz012345",
            type=MemoryType.FACT,
            source=SourceInfo(agent_id="a"),
        )
        exporter = MemoryExporter(store, allowed_root=tmp_path)
        path = tmp_path / "out.json"
        exporter.to_json(str(path), include_sensitive=True)
        raw = path.read_text(encoding="utf-8")
        assert "sk-abcdefghijklmnopqrstuvwxyz012345" in raw
        envelope = json.loads(raw)
        assert envelope["redacted"] is False

        # Importing a non-redacted file warns nothing about placeholders.
        dest, _ = make_store()
        report = MemoryImporter(dest, embedder=RecordingEmbedder()).from_json(str(path))
        assert not any("redact" in w.lower() for w in report.warnings)

    def test_vector_consistency_rule(self, tmp_path: Path) -> None:
        store, backend = make_store()
        embedder = RecordingEmbedder()
        seed_meta(backend, embedder)
        secret = store.create(
            content="leak sk-abcdefghijklmnopqrstuvwxyz012345 here",
            type=MemoryType.FACT,
            source=SourceInfo(agent_id="a"),
            embedding=pack_embedding(embedder._vector("x")),
        )
        clean = store.create(
            content="perfectly ordinary content",
            type=MemoryType.FACT,
            source=SourceInfo(agent_id="a"),
            embedding=pack_embedding(embedder._vector("y")),
        )
        # A sensitive-KEY-masked entry (secret-SHAPED content, v0.2 softening):
        # to_safe_dict blanks the whole content, so its vector must also be
        # dropped (S3 — locks the key-masked branch).
        key_masked = store.create(
            content="sk-" + "q" * 24,
            type=MemoryType.FACT,
            key="service_api_key",
            source=SourceInfo(agent_id="a"),
            embedding=pack_embedding(embedder._vector("z")),
        )
        exporter = MemoryExporter(store, allowed_root=tmp_path)
        path = tmp_path / "out.json"
        exporter.to_json(str(path), include_embeddings=True)
        envelope = json.loads(path.read_text(encoding="utf-8"))
        entries = {e["id"]: e for e in envelope["entries"]}
        assert entries[secret.id]["embedding"] is None  # content-scan changed content
        assert entries[key_masked.id]["embedding"] is None  # key-masked -> vector dropped
        assert entries[clean.id]["embedding"] is not None

    def test_atomic_write_leaves_no_file_on_failure(self, tmp_path: Path, monkeypatch) -> None:
        store, _ = make_store()
        store.create(content="c", type=MemoryType.FACT, source=SourceInfo(agent_id="a"))
        exporter = MemoryExporter(store, allowed_root=tmp_path)
        path = tmp_path / "out.json"

        import tulving.export.formats as formats_mod

        def boom(*_args: Any, **_kwargs: Any) -> str:
            raise RuntimeError("serialization blew up")

        monkeypatch.setattr(formats_mod.json, "dumps", boom)
        with pytest.raises(RuntimeError):
            exporter.to_json(str(path))
        assert not path.exists()


# --------------------------------------------------------------------------- #
# TestEnvelope — shape & untrusted input
# --------------------------------------------------------------------------- #


class TestEnvelope:
    def test_envelope_shape(self, tmp_path: Path) -> None:
        store, backend = make_store()
        embedder = RecordingEmbedder()
        seed_meta(backend, embedder)
        store.create(content="a", type=MemoryType.FACT, source=SourceInfo(agent_id="a"))
        store.create(content="b", type=MemoryType.FACT, source=SourceInfo(agent_id="a"))
        exporter = MemoryExporter(store, allowed_root=tmp_path)

        plain = tmp_path / "plain.json"
        exporter.to_json(str(plain))
        env = json.loads(plain.read_text(encoding="utf-8"))
        assert env["format"] == "tulving-export"
        assert env["format_version"] == 1
        assert env["schema_version"] == backend.get_meta()["schema_version"]
        assert env["entry_count"] == len(env["entries"]) == 2
        assert env["embeddings"] is None
        parsed = datetime.fromisoformat(env["exported_at"])
        assert parsed.tzinfo is not None

        with_vec = tmp_path / "vec.json"
        exporter.to_json(str(with_vec), include_embeddings=True)
        env2 = json.loads(with_vec.read_text(encoding="utf-8"))
        assert env2["embeddings"] == {
            "model_id": embedder.model_id,
            "dimension": embedder.dimension,
            "distance_metric": embedder.distance_metric,
        }

    def test_malformed_files(self, tmp_path: Path) -> None:
        dest, _ = make_store()
        importer = MemoryImporter(dest)

        not_json = tmp_path / "notjson.json"
        not_json.write_text("this is not json {", encoding="utf-8")
        with pytest.raises(MemoryStoreError):
            importer.from_json(str(not_json))

        toplevel_list = tmp_path / "list.json"
        toplevel_list.write_text("[]", encoding="utf-8")
        with pytest.raises(MemoryStoreError):
            importer.from_json(str(toplevel_list))

        wrong_format = tmp_path / "wrong.json"
        wrong_format.write_text(json.dumps({"format": "nope", "entries": []}), encoding="utf-8")
        with pytest.raises(MemoryStoreError):
            importer.from_json(str(wrong_format))

        with pytest.raises(StorageError):
            importer.from_json(str(tmp_path / "does_not_exist.json"))

    def test_deeply_nested_json_rejected(self, tmp_path: Path) -> None:
        """S1: deeply-nested input raises RecursionError inside json.loads —
        it must map to MemoryStoreError, never escape untyped, write nothing."""
        dest, _ = make_store()
        importer = MemoryImporter(dest)
        bomb = tmp_path / "bomb.json"
        bomb.write_text("[" * 6000 + "]" * 6000, encoding="utf-8")
        with pytest.raises(MemoryStoreError):
            importer.from_json(str(bomb))
        assert dest.count(include_archived=True) == 0

    def test_version_gates(self, tmp_path: Path) -> None:
        dest, backend = make_store()
        importer = MemoryImporter(dest)
        store_sv = backend.get_meta()["schema_version"]

        future_fmt = tmp_path / "fmt.json"
        future_fmt.write_text(
            json.dumps(
                {
                    "format": "tulving-export",
                    "format_version": 2,
                    "schema_version": store_sv,
                    "entries": [],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(MemoryStoreError):
            importer.from_json(str(future_fmt))

        future_schema = tmp_path / "schema.json"
        future_schema.write_text(
            json.dumps(
                {
                    "format": "tulving-export",
                    "format_version": 1,
                    "schema_version": store_sv + 1,
                    "entries": [],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(MemoryStoreError):
            importer.from_json(str(future_schema))

        equal = tmp_path / "equal.json"
        equal.write_text(
            json.dumps(
                {
                    "format": "tulving-export",
                    "format_version": 1,
                    "schema_version": store_sv,
                    "entries": [],
                }
            ),
            encoding="utf-8",
        )
        report = importer.from_json(str(equal))  # equal passes
        assert report.entries_imported == 0

    def test_invalid_entries_skip_not_abort(self, tmp_path: Path) -> None:
        src, _ = make_store()
        good = src.create(content="good one", type=MemoryType.FACT, source=SourceInfo(agent_id="a"))
        good_dict = good.to_dict()
        bogus_type = dict(good_dict)
        bogus_type["id"] = "b1"
        bogus_type["type"] = "not-a-real-type"
        missing_content = dict(good_dict)
        missing_content["id"] = "b2"
        missing_content.pop("content")

        envelope = {
            "format": "tulving-export",
            "format_version": 1,
            "schema_version": 1,
            "entries": [good_dict, "i-am-not-a-dict", bogus_type, missing_content],
        }
        path = tmp_path / "mixed.json"
        path.write_text(json.dumps(envelope), encoding="utf-8")

        dest, _ = make_store()
        report = MemoryImporter(dest, embedder=RecordingEmbedder()).from_json(str(path))
        assert report.entries_imported == 1
        assert report.entries_skipped == 3
        assert len(report.errors) == 3
        assert dest.count() == 1

    def test_bad_conflict_mode_before_read(self, tmp_path: Path) -> None:
        dest, _ = make_store()
        importer = MemoryImporter(dest)
        # File never even has to exist: the mode is validated first.
        with pytest.raises(ValueError):
            importer.from_json(str(tmp_path / "missing.json"), on_key_conflict="merge")


# --------------------------------------------------------------------------- #
# TestIdRemapping — mandatory audit area
# --------------------------------------------------------------------------- #


class TestIdRemapping:
    def test_fresh_ids(self, tmp_path: Path) -> None:
        src, _ = make_store()
        for i in range(3):
            src.create(content=f"e{i}", type=MemoryType.FACT, source=SourceInfo(agent_id="a"))
        old_ids = {e.id for e in src.list(limit=100)}
        path = tmp_path / "out.json"
        MemoryExporter(src, allowed_root=tmp_path).to_json(str(path))

        dest, _ = make_store()
        pre = dest.create(content="pre", type=MemoryType.FACT, source=SourceInfo(agent_id="a"))
        MemoryImporter(dest, embedder=RecordingEmbedder()).from_json(str(path))
        new_ids = {e.id for e in dest.list(limit=100)} - {pre.id}
        assert new_ids.isdisjoint(old_ids)
        for new_id in new_ids:
            assert new_id.isalnum() and len(new_id) == len(pre.id)

    def test_references_rewritten_consistently(self, tmp_path: Path) -> None:
        src, _ = make_store()
        b = src.create(content="B", type=MemoryType.FACT, source=SourceInfo(agent_id="a"))
        src.create(
            content="A",
            type=MemoryType.FACT,
            source=SourceInfo(agent_id="a"),
            relationships=[Relationship(target_id=b.id, relationship_type="relates_to")],
        )
        v1 = src.create(content="V", type=MemoryType.FACT, key="k", source=SourceInfo(agent_id="a"))
        src.create(content="V2", type=MemoryType.FACT, key="k", source=SourceInfo(agent_id="a"))
        src.create(
            content="SUM",
            type=MemoryType.SUMMARY,
            source=SourceInfo(agent_id="a"),
            _allow_summary=True,
            _source_entry_ids=[b.id, v1.id],
        )

        path = tmp_path / "out.json"
        MemoryExporter(src, allowed_root=tmp_path).to_json(str(path), include_archived=True)

        dest, _ = make_store()
        report = MemoryImporter(dest, embedder=RecordingEmbedder()).from_json(str(path))
        entries = by_content(dest)
        new_b = entries["B"].id
        new_v1 = entries["V"].id

        rel = entries["A"].relationships[0]
        assert rel.target_id == new_b
        summary = entries["SUM"]
        assert summary.source_entry_ids == [new_b, new_v1]
        v2 = entries["V2"]
        supersedes = next(r for r in v2.relationships if r.relationship_type == "supersedes")
        assert supersedes.target_id == new_v1
        # A->B (1) + summary sources (2) + supersede back-link (1) = 4 rewrites.
        assert report.id_remappings == 4

    def test_dangling_reference_dropped(self, tmp_path: Path) -> None:
        src, _ = make_store()
        b = src.create(content="B", type=MemoryType.FACT, source=SourceInfo(agent_id="a"))
        src.create(
            content="A",
            type=MemoryType.FACT,
            source=SourceInfo(agent_id="a"),
            relationships=[Relationship(target_id=b.id, relationship_type="relates_to")],
        )
        # Export ONLY A by forgetting/excluding B from the file: hand-build it.
        a_entry = next(e for e in src.list(limit=100) if e.content == "A")
        envelope = {
            "format": "tulving-export",
            "format_version": 1,
            "schema_version": 1,
            "entries": [a_entry.to_dict()],  # target B absent from the file
        }
        path = tmp_path / "out.json"
        path.write_text(json.dumps(envelope), encoding="utf-8")

        dest, _ = make_store()
        report = MemoryImporter(dest, embedder=RecordingEmbedder()).from_json(str(path))
        imported = by_content(dest)["A"]
        assert imported.relationships == []
        assert any(b.id in err for err in report.errors)
        assert report.entries_imported == 1

    def test_reference_to_skipped_entry_dropped(self, tmp_path: Path) -> None:
        src, _ = make_store()
        b = src.create(content="B", type=MemoryType.FACT, source=SourceInfo(agent_id="a"))
        a = src.create(
            content="A",
            type=MemoryType.FACT,
            source=SourceInfo(agent_id="a"),
            relationships=[Relationship(target_id=b.id, relationship_type="relates_to")],
        )
        b_broken = b.to_dict()
        b_broken["type"] = "bogus-type"  # B is structurally invalid -> skipped
        envelope = {
            "format": "tulving-export",
            "format_version": 1,
            "schema_version": 1,
            "entries": [a.to_dict(), b_broken],
        }
        path = tmp_path / "out.json"
        path.write_text(json.dumps(envelope), encoding="utf-8")

        dest, _ = make_store()
        report = MemoryImporter(dest, embedder=RecordingEmbedder()).from_json(str(path))
        assert report.entries_skipped == 1
        assert by_content(dest)["A"].relationships == []
        assert any(b.id in err for err in report.errors)

    def test_duplicate_old_id_skipped_not_aborted(self, tmp_path: Path) -> None:
        """C1: two survivors sharing an old id -> the second is skipped and
        recorded (not a PK-collision abort of the whole import)."""
        e1 = make_entry(id="shared", content="first-of-dup")
        e2 = make_entry(id="shared", content="second-of-dup")
        envelope = {
            "format": "tulving-export",
            "format_version": 1,
            "schema_version": 1,
            "entries": [e1.to_dict(), e2.to_dict()],
        }
        path = tmp_path / "dup.json"
        path.write_text(json.dumps(envelope), encoding="utf-8")

        dest, _ = make_store()
        report = MemoryImporter(dest, embedder=RecordingEmbedder()).from_json(str(path))
        assert report.entries_imported == 1
        assert report.entries_skipped == 1
        assert any("duplicate id" in err for err in report.errors)
        assert dest.count(include_archived=True) == 1
        assert by_content(dest, include_archived=False).popitem()[0] == "first-of-dup"


# --------------------------------------------------------------------------- #
# TestDedup — D1 semantics
# --------------------------------------------------------------------------- #


class TestDedup:
    def _export_single_key(self, tmp_path: Path, content: str) -> Path:
        src, _ = make_store()
        src.create(content=content, type=MemoryType.FACT, key="k", source=SourceInfo(agent_id="a"))
        path = tmp_path / "out.json"
        MemoryExporter(src, allowed_root=tmp_path).to_json(str(path), include_sensitive=True)
        return path

    def test_skip_default(self, tmp_path: Path) -> None:
        path = self._export_single_key(tmp_path, "imported-value")
        dest, _ = make_store()
        live = dest.create(
            content="live-value", type=MemoryType.FACT, key="k", source=SourceInfo(agent_id="a")
        )
        report = MemoryImporter(dest, embedder=RecordingEmbedder()).from_json(str(path))
        assert report.entries_skipped == 1
        assert report.entries_imported == 0
        held = dest.get_by_key("k", touch=False)
        assert held is not None
        assert held.id == live.id
        assert held.content == "live-value"

    def test_supersede(self, tmp_path: Path) -> None:
        path = self._export_single_key(tmp_path, "imported-value")
        dest, _ = make_store()
        live = dest.create(
            content="live-value", type=MemoryType.FACT, key="k", source=SourceInfo(agent_id="a")
        )
        report = MemoryImporter(dest, embedder=RecordingEmbedder()).from_json(
            str(path), on_key_conflict="supersede"
        )
        assert report.entries_imported == 1
        old = dest.get_by_id(live.id, touch=False)
        assert old is not None
        assert old.archived is True
        assert old.archive_reason is ArchiveReason.SUPERSEDED
        new_live = dest.get_by_key("k", touch=False)
        assert new_live is not None
        assert new_live.content == "imported-value"
        supersedes = next(r for r in new_live.relationships if r.relationship_type == "supersedes")
        assert supersedes.target_id == live.id

    def test_supersede_against_previously_archived_key(self, tmp_path: Path) -> None:
        """Audit regression #18: an archived row holding ``k`` never blocks."""
        path = self._export_single_key(tmp_path, "imported-value")
        dest, _ = make_store()
        # First live+archived history on k, then a new live holder.
        dest.create(content="v1", type=MemoryType.FACT, key="k", source=SourceInfo(agent_id="a"))
        dest.create(content="v2", type=MemoryType.FACT, key="k", source=SourceInfo(agent_id="a"))
        report = MemoryImporter(dest, embedder=RecordingEmbedder()).from_json(
            str(path), on_key_conflict="supersede"
        )
        assert report.entries_imported == 1
        new_live = dest.get_by_key("k", touch=False)
        assert new_live is not None and new_live.content == "imported-value"
        # Two archived rows now hold k (v1, v2) — the partial unique index allows it.
        archived = [
            e for e in dest.list(include_archived=True, limit=100) if e.key == "k" and e.archived
        ]
        assert len(archived) == 2

    def test_keyless_always_import(self, tmp_path: Path) -> None:
        src, _ = make_store()
        src.create(content="dup", type=MemoryType.FACT, source=SourceInfo(agent_id="a"))
        src.create(content="dup", type=MemoryType.FACT, source=SourceInfo(agent_id="a"))
        path = tmp_path / "out.json"
        MemoryExporter(src, allowed_root=tmp_path).to_json(str(path), include_sensitive=True)

        dest, _ = make_store()
        report = MemoryImporter(dest, embedder=RecordingEmbedder()).from_json(str(path))
        assert report.entries_imported == 2
        assert dest.count() == 2

    def test_archived_keyed_never_conflicts(self, tmp_path: Path) -> None:
        src, _ = make_store()
        e = src.create(
            content="archived-k", type=MemoryType.FACT, key="k", source=SourceInfo(agent_id="a")
        )
        src.archive(e.id, ArchiveReason.EVICTED)
        path = tmp_path / "out.json"
        MemoryExporter(src, allowed_root=tmp_path).to_json(
            str(path), include_archived=True, include_sensitive=True
        )

        dest, _ = make_store()
        dest.create(
            content="live-k", type=MemoryType.FACT, key="k", source=SourceInfo(agent_id="a")
        )
        report = MemoryImporter(dest, embedder=RecordingEmbedder()).from_json(str(path))
        assert report.entries_imported == 1  # archived import untouched by the policy
        assert report.entries_skipped == 0
        live = dest.get_by_key("k", touch=False)
        assert live is not None and live.content == "live-k"

    def test_intra_file_duplicate_live_keys(self, tmp_path: Path) -> None:
        """First (file order) wins; the second follows on_key_conflict."""
        e1 = make_entry(id="f1", content="first", key="dup")
        e2 = make_entry(id="f2", content="second", key="dup")
        envelope = {
            "format": "tulving-export",
            "format_version": 1,
            "schema_version": 1,
            "entries": [e1.to_dict(), e2.to_dict()],
        }
        path = tmp_path / "out.json"
        path.write_text(json.dumps(envelope), encoding="utf-8")

        dest, _ = make_store()
        report = MemoryImporter(dest, embedder=RecordingEmbedder()).from_json(str(path))
        assert report.entries_imported == 1
        assert report.entries_skipped == 1
        held = dest.get_by_key("dup", touch=False)
        assert held is not None and held.content == "first"


# --------------------------------------------------------------------------- #
# TestReembedReuse
# --------------------------------------------------------------------------- #


class TestReembedReuse:
    def test_default_reembeds_and_indexes(self, tmp_path: Path) -> None:
        src, _ = make_store()
        for i in range(3):
            src.create(content=f"c{i}", type=MemoryType.FACT, source=SourceInfo(agent_id="a"))
        path = tmp_path / "out.json"
        MemoryExporter(src, allowed_root=tmp_path).to_json(str(path))  # no vectors

        dest, backend = make_store()
        embedder = RecordingEmbedder()
        seed_meta(backend, embedder)
        report = MemoryImporter(dest, embedder=embedder).from_json(str(path))
        assert report.entries_reembedded == 3
        assert report.embeddings_reused == 0
        assert len(embedder.batch_calls) == 1
        assert set(embedder.batch_calls[0]) == {"c0", "c1", "c2"}
        # restore persisted the BLOB (searchable after reconcile).
        for entry in dest.list(limit=100):
            vec = dest.get_embedding(entry.id)
            assert vec is not None and len(vec) == embedder.dimension

    def test_exact_match_reuse(self, tmp_path: Path) -> None:
        src, src_backend = make_store()
        embedder = RecordingEmbedder()
        seed_meta(src_backend, embedder)
        for i in range(2):
            src.create(
                content=f"c{i}",
                type=MemoryType.FACT,
                source=SourceInfo(agent_id="a"),
                embedding=pack_embedding(embedder._vector(f"c{i}")),
            )
        path = tmp_path / "out.json"
        MemoryExporter(src, allowed_root=tmp_path).to_json(str(path), include_embeddings=True)

        dest, dest_backend = make_store()
        seed_meta(dest_backend, embedder)  # identical meta
        importer_embedder = RecordingEmbedder()
        report = MemoryImporter(dest, embedder=importer_embedder).from_json(str(path))
        assert report.embeddings_reused == 2
        assert report.entries_reembedded == 0
        assert importer_embedder.batch_calls == []  # embedder never called
        # BLOBs byte-equal the exported vectors.
        src_by_content = by_content(src)
        for content, entry in by_content(dest).items():
            expected = src.get_embedding(src_by_content[content].id)
            assert dest.get_embedding(entry.id) == expected

    def test_any_mismatch_reembeds(self, tmp_path: Path) -> None:
        src, src_backend = make_store()
        exporter_embedder = RecordingEmbedder(model_id="model-A")
        seed_meta(src_backend, exporter_embedder)
        src.create(
            content="c0",
            type=MemoryType.FACT,
            source=SourceInfo(agent_id="a"),
            embedding=pack_embedding(exporter_embedder._vector("c0")),
        )
        src.create(
            content="c1",
            type=MemoryType.FACT,
            source=SourceInfo(agent_id="a"),
            embedding=pack_embedding(exporter_embedder._vector("c1")),
        )
        path = tmp_path / "out.json"
        MemoryExporter(src, allowed_root=tmp_path).to_json(str(path), include_embeddings=True)

        # Same dimension, DIFFERENT model_id -> all re-embed (adapter-swap rule).
        dest, dest_backend = make_store()
        other = RecordingEmbedder(model_id="model-B")
        seed_meta(dest_backend, other)
        report = MemoryImporter(dest, embedder=other).from_json(str(path))
        assert report.entries_reembedded == 2
        assert report.embeddings_reused == 0

    def test_distance_metric_mismatch_reembeds(self, tmp_path: Path) -> None:
        """Reuse-gate #24: same model_id + dimension but a DIFFERENT
        distance_metric invalidates the whole reuse (metric is part of vector
        identity — cosine vectors are not L2 vectors)."""
        src, src_backend = make_store()
        embedder = RecordingEmbedder(dimension=4, model_id="model-A", distance_metric="cosine")
        seed_meta(src_backend, embedder)
        for i in range(2):
            src.create(
                content=f"c{i}",
                type=MemoryType.FACT,
                source=SourceInfo(agent_id="a"),
                embedding=pack_embedding(embedder._vector(f"c{i}")),
            )
        path = tmp_path / "out.json"
        MemoryExporter(src, allowed_root=tmp_path).to_json(str(path), include_embeddings=True)

        dest, dest_backend = make_store()
        # Identical model_id AND dimension — ONLY the metric differs.
        other = RecordingEmbedder(dimension=4, model_id="model-A", distance_metric="l2")
        seed_meta(dest_backend, other)
        report = MemoryImporter(dest, embedder=other).from_json(str(path))
        assert report.entries_reembedded == 2
        assert report.embeddings_reused == 0

    def test_embeddings_block_dimension_mismatch_reembeds(self, tmp_path: Path) -> None:
        """Reuse-gate #24: the envelope's declared dimension differing from the
        store meta's dimension fails the gate for every entry (block-level
        mismatch, distinct from a per-entry wrong-length vector)."""
        good = make_entry(id="g0", content="c0")
        good_dict = good.to_dict()
        good_dict["embedding"] = [0.1, 0.2, 0.3, 0.4]  # 4 floats, matches the BLOCK's 4
        envelope = {
            "format": "tulving-export",
            "format_version": 1,
            "schema_version": 1,
            # Block declares dimension 4; the destination store's meta is dim 8.
            "embeddings": {"model_id": "model-A", "dimension": 4, "distance_metric": "cosine"},
            "entries": [good_dict],
        }
        path = tmp_path / "out.json"
        path.write_text(json.dumps(envelope), encoding="utf-8")

        dest, dest_backend = make_store()
        seed_meta(dest_backend, RecordingEmbedder(dimension=8, model_id="model-A"))
        report = MemoryImporter(
            dest, embedder=RecordingEmbedder(dimension=8, model_id="model-A")
        ).from_json(str(path))
        assert report.embeddings_reused == 0
        assert report.entries_reembedded == 1

    def test_reused_embedding_is_byte_exact(self, tmp_path: Path) -> None:
        """A reused vector is the exact same BLOB bytes on both sides — stronger
        than float-list equality (guards a pack/round encoding drift)."""
        src, src_backend = make_store()
        embedder = RecordingEmbedder(dimension=4, model_id="model-A")
        seed_meta(src_backend, embedder)
        src.create(
            content="c0",
            type=MemoryType.FACT,
            source=SourceInfo(agent_id="a"),
            embedding=pack_embedding(embedder._vector("c0")),
        )
        path = tmp_path / "out.json"
        MemoryExporter(src, allowed_root=tmp_path).to_json(str(path), include_embeddings=True)

        dest, dest_backend = make_store()
        seed_meta(dest_backend, embedder)
        report = MemoryImporter(dest, embedder=embedder).from_json(str(path))
        assert report.embeddings_reused == 1
        src_entry = by_content(src)["c0"]
        dest_entry = by_content(dest)["c0"]
        src_bytes = src_backend.get_embedding(src_entry.id)
        dest_bytes = dest_backend.get_embedding(dest_entry.id)
        assert src_bytes is not None
        assert dest_bytes == src_bytes  # bytes == bytes, not just float-list ==

    def test_wrong_length_vector_reembeds_only_that_entry(self, tmp_path: Path) -> None:
        embedder = RecordingEmbedder(dimension=4, model_id="model-A")
        good = make_entry(id="g", content="good")
        bad = make_entry(id="b", content="bad")
        good_dict = good.to_dict()
        good_dict["embedding"] = embedder._vector("good")  # length 4
        bad_dict = bad.to_dict()
        bad_dict["embedding"] = [0.1, 0.2]  # wrong length
        envelope = {
            "format": "tulving-export",
            "format_version": 1,
            "schema_version": 1,
            "embeddings": {
                "model_id": "model-A",
                "dimension": 4,
                "distance_metric": "cosine",
            },
            "entries": [good_dict, bad_dict],
        }
        path = tmp_path / "out.json"
        path.write_text(json.dumps(envelope), encoding="utf-8")

        dest, dest_backend = make_store()
        seed_meta(dest_backend, embedder)  # matches model-A / dim 4 / cosine
        report = MemoryImporter(dest, embedder=embedder).from_json(str(path))
        assert report.embeddings_reused == 1
        assert report.entries_reembedded == 1

    def test_malicious_non_numeric_embedding_reembeds(self, tmp_path: Path) -> None:
        """S2: a non-numeric vector of the EXACT declared length must not crash
        the reuse coercion — the entry re-embeds instead."""
        embedder = RecordingEmbedder(dimension=4, model_id="model-A")
        good = make_entry(id="g", content="good")
        evil = make_entry(id="e", content="evil")
        good_dict = good.to_dict()
        good_dict["embedding"] = embedder._vector("good")  # valid length-4 floats
        evil_dict = evil.to_dict()
        evil_dict["embedding"] = ["a", "b", "c", "d"]  # exact length, non-numeric
        envelope = {
            "format": "tulving-export",
            "format_version": 1,
            "schema_version": 1,
            "embeddings": {"model_id": "model-A", "dimension": 4, "distance_metric": "cosine"},
            "entries": [good_dict, evil_dict],
        }
        path = tmp_path / "evil.json"
        path.write_text(json.dumps(envelope), encoding="utf-8")

        dest, dest_backend = make_store()
        seed_meta(dest_backend, embedder)  # meta matches -> reuse path is taken
        report = MemoryImporter(dest, embedder=embedder).from_json(str(path))
        assert report.embeddings_reused == 1  # only the valid one reused
        assert report.entries_reembedded == 1  # the malicious one re-embedded
        assert dest.count() == 2

    def test_no_embedder_imports_without_vectors(self, tmp_path: Path) -> None:
        src, _ = make_store()
        src.create(content="c0", type=MemoryType.FACT, source=SourceInfo(agent_id="a"))
        path = tmp_path / "out.json"
        MemoryExporter(src, allowed_root=tmp_path).to_json(str(path))  # no vectors

        dest, _ = make_store()
        report = MemoryImporter(dest, embedder=None).from_json(str(path))
        assert report.entries_reembedded == 0
        assert report.embeddings_reused == 0
        assert len([w for w in report.warnings if "embedding adapter" in w]) == 1
        entry = dest.list(limit=10)[0]
        assert dest.get_embedding(entry.id) is None


# --------------------------------------------------------------------------- #
# TestAtomicityAndRoundTrip
# --------------------------------------------------------------------------- #


class TestAtomicityAndRoundTrip:
    def test_all_or_nothing_write(self, tmp_path: Path, monkeypatch) -> None:
        src, _ = make_store()
        for i in range(4):
            src.create(content=f"c{i}", type=MemoryType.FACT, source=SourceInfo(agent_id="a"))
        path = tmp_path / "out.json"
        MemoryExporter(src, allowed_root=tmp_path).to_json(str(path))

        dest, _ = make_store()
        importer = MemoryImporter(dest, embedder=RecordingEmbedder())
        real_restore = dest.restore
        calls = {"n": 0}

        def failing_restore(entry: MemoryEntry, **kwargs: Any) -> MemoryEntry:
            calls["n"] += 1
            if calls["n"] == 3:
                raise StorageError("disk full on the third row")
            return real_restore(entry, **kwargs)

        monkeypatch.setattr(dest, "restore", failing_restore)
        with pytest.raises(StorageError):
            importer.from_json(str(path))
        assert dest.count(include_archived=True) == 0  # rolled back entirely

    def test_sqlite_all_or_nothing_write(self, tmp_path: Path, monkeypatch) -> None:
        """Regression #26 on a REAL SQLite backend: a mid-batch restore failure
        must ROLLBACK the open BEGIN IMMEDIATE transaction so ZERO rows commit to
        disk — proves the real transaction path, not the InMemory snapshot."""
        src, _ = make_store()
        for i in range(4):
            src.create(content=f"c{i}", type=MemoryType.FACT, source=SourceInfo(agent_id="a"))
        path = tmp_path / "out.json"
        MemoryExporter(src, allowed_root=tmp_path).to_json(str(path))

        dest, dest_backend = make_sqlite_store(tmp_path)
        try:
            importer = MemoryImporter(dest, embedder=RecordingEmbedder())
            real_restore = dest.restore
            calls = {"n": 0}

            def failing_restore(entry: MemoryEntry, **kwargs: Any) -> MemoryEntry:
                calls["n"] += 1
                if calls["n"] == 3:
                    raise StorageError("disk full on the third row (real sqlite)")
                # The first two rows really wrote inside the open transaction.
                return real_restore(entry, **kwargs)

            monkeypatch.setattr(dest, "restore", failing_restore)
            with pytest.raises(StorageError):
                importer.from_json(str(path))
            # Same connection: the ROLLBACK undid the two committed-in-txn rows.
            assert dest.count(include_archived=True) == 0
            # And it is genuinely durable on disk: a fresh connection agrees.
            reopened = SQLiteBackend(dest_backend.db_path)
            try:
                assert MemoryStore(reopened).count(include_archived=True) == 0
            finally:
                reopened.close()
        finally:
            dest_backend.close()

    def test_embedding_failure_pre_transaction(self, tmp_path: Path) -> None:
        src, _ = make_store()
        src.create(content="c0", type=MemoryType.FACT, source=SourceInfo(agent_id="a"))
        path = tmp_path / "out.json"
        MemoryExporter(src, allowed_root=tmp_path).to_json(str(path))

        dest, _ = make_store()
        importer = MemoryImporter(dest, embedder=RaisingEmbedder())
        with pytest.raises(StorageError):
            importer.from_json(str(path))
        assert dest.count(include_archived=True) == 0

    def test_full_round_trip_contract(self, tmp_path: Path) -> None:
        src, _ = make_store()
        b = src.create(
            content="B target",
            type=MemoryType.FACT,
            source=SourceInfo(agent_id="agent-1", step_id="s1", run_id="r1", workflow_name="wf"),
            tags=["alpha", "beta"],
        )
        a = src.create(
            content="A with everything",
            type=MemoryType.DECISION,
            key="decision-key",
            source=SourceInfo(agent_id="agent-1"),
            tags=["gamma"],
            base_importance=0.9,
            relationships=[Relationship(target_id=b.id, relationship_type="relates_to")],
            pinned=True,
        )
        # give A a nonzero access_count
        src.get_by_id(a.id, touch=True)
        src.get_by_id(a.id, touch=True)
        src.create(
            content="rollup",
            type=MemoryType.SUMMARY,
            source=SourceInfo(agent_id="agent-1"),
            _allow_summary=True,
            _source_entry_ids=[a.id, b.id],
        )
        archived = src.create(
            content="to be archived", type=MemoryType.FACT, source=SourceInfo(agent_id="agent-1")
        )
        src.archive(archived.id, ArchiveReason.EVICTED)

        path = tmp_path / "out.json"
        MemoryExporter(src, allowed_root=tmp_path).to_json(
            str(path), include_archived=True, include_sensitive=True
        )

        dest, _ = make_store()
        report = MemoryImporter(dest, embedder=RecordingEmbedder()).from_json(str(path))
        assert report.entries_imported == 4

        original = by_content(src)
        imported = by_content(dest)
        assert set(original) == set(imported)

        fields = [
            "content",
            "type",
            "key",
            "tags",
            "base_importance",
            "session_id",
            "created_at",
            "updated_at",
            "last_accessed_at",
            "access_count",
            "archived",
            "archive_reason",
            "pinned",
        ]
        for content, orig in original.items():
            imp = imported[content]
            for field_name in fields:
                assert getattr(imp, field_name) == getattr(orig, field_name), (content, field_name)
            assert imp.source.agent_id == orig.source.agent_id
            assert imp.source.step_id == orig.source.step_id

        # graph isomorphism under the old->new id map (keyed on content)
        new_b = imported["B target"].id
        assert imported["A with everything"].relationships[0].target_id == new_b
        new_a = imported["A with everything"].id
        assert imported["rollup"].source_entry_ids == [new_a, new_b]
        # access_count byte-preserved (import is not access)
        assert (
            imported["A with everything"].access_count == original["A with everything"].access_count
        )
        assert imported["A with everything"].access_count == 2

    def test_redacted_round_trip_documented_exception(self, tmp_path: Path) -> None:
        src, _ = make_store()
        src.create(
            content="password=hunter2xyz please",
            type=MemoryType.FACT,
            source=SourceInfo(agent_id="a"),
        )
        path = tmp_path / "out.json"
        MemoryExporter(src, allowed_root=tmp_path).to_json(str(path))  # default redaction

        dest, _ = make_store()
        report = MemoryImporter(dest, embedder=RecordingEmbedder()).from_json(str(path))
        imported = dest.list(limit=10)[0]
        assert "hunter2xyz" not in imported.content
        assert "[REDACTED]" in imported.content
        assert any("redact" in w.lower() for w in report.warnings)

    def test_empty_store_and_empty_file(self, tmp_path: Path) -> None:
        src, _ = make_store()
        path = tmp_path / "empty.json"
        MemoryExporter(src, allowed_root=tmp_path).to_json(str(path))
        env = json.loads(path.read_text(encoding="utf-8"))
        assert env["entry_count"] == 0
        assert env["entries"] == []

        dest, _ = make_store()
        report = MemoryImporter(dest, embedder=RecordingEmbedder()).from_json(str(path))
        assert report.entries_imported == 0
        assert report.entries_skipped == 0
        assert report.entries_reembedded == 0
        assert report.embeddings_reused == 0
        assert report.id_remappings == 0
        assert report.errors == []


# --------------------------------------------------------------------------- #
# TestRestoreStandalone — C4: restore() supersede is atomic without an ambient txn
# --------------------------------------------------------------------------- #


class TestRestoreStandalone:
    def test_standalone_supersede_restore_rolls_back_on_failure(self, monkeypatch) -> None:
        store, backend = make_store()
        live = store.create(
            content="live", type=MemoryType.FACT, key="k", source=SourceInfo(agent_id="a")
        )
        incoming = make_entry(id="imported", content="incoming", key="k")

        def failing_create(entry: Any, *, embedding: Any = None) -> None:
            raise StorageError("insert failed after the archive write")

        monkeypatch.setattr(backend, "create", failing_create)
        with pytest.raises(StorageError):
            store.restore(incoming, supersede_live_key=True)  # no ambient transaction()

        # The archive of the live holder must have rolled back — no orphaned key.
        held = store.get_by_key("k", touch=False)
        assert held is not None
        assert held.id == live.id
        assert held.archived is False


# --------------------------------------------------------------------------- #
# TestQAHardening — QA-added coverage: emission failure paths, untrusted-input
# gates, dangling summary sources, pagination, and the reserved-key caveat.
# --------------------------------------------------------------------------- #


class TestQAHardening:
    def test_atomic_write_os_error_wrapped_as_storage_error(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """`os.replace` failing (byte-level I/O) maps to StorageError and cleans
        up the ``.tmp`` — never a leaked OSError, never a half file (covers the
        _atomic_write OSError branch)."""
        store, _ = make_store()
        store.create(content="c", type=MemoryType.FACT, source=SourceInfo(agent_id="a"))
        exporter = MemoryExporter(store, allowed_root=tmp_path)
        path = tmp_path / "out.json"

        import tulving.export.formats as formats_mod

        def boom(*_args: Any, **_kwargs: Any) -> None:
            raise OSError("replace across devices failed")

        monkeypatch.setattr(formats_mod.os, "replace", boom)
        with pytest.raises(StorageError):
            exporter.to_json(str(path))
        assert not path.exists()
        assert not (tmp_path / "out.json.tmp").exists()  # tmp cleaned up

    def test_entries_not_a_list_rejected(self, tmp_path: Path) -> None:
        """A valid envelope whose ``entries`` is not a list is untrusted-input
        garbage → MemoryStoreError, nothing written (covers the entries-type gate)."""
        dest, backend = make_store()
        sv = backend.get_meta()["schema_version"]
        bad = tmp_path / "bad.json"
        bad.write_text(
            json.dumps(
                {
                    "format": "tulving-export",
                    "format_version": 1,
                    "schema_version": sv,
                    "entries": {"not": "a list"},
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(MemoryStoreError):
            MemoryImporter(dest).from_json(str(bad))
        assert dest.count(include_archived=True) == 0

    def test_dangling_source_entry_id_dropped(self, tmp_path: Path) -> None:
        """A SUMMARY whose ``source_entry_ids`` names an id absent from the file
        drops that source and records it — the source-link twin of the dangling
        relationship drop (covers the source_entry_ids dangling branch)."""
        src, _ = make_store()
        b = src.create(content="B", type=MemoryType.FACT, source=SourceInfo(agent_id="a"))
        src.create(
            content="SUM",
            type=MemoryType.SUMMARY,
            source=SourceInfo(agent_id="a"),
            _allow_summary=True,
            _source_entry_ids=[b.id],
        )
        summary = next(e for e in src.list(limit=100) if e.content == "SUM")
        envelope = {
            "format": "tulving-export",
            "format_version": 1,
            "schema_version": 1,
            "entries": [summary.to_dict()],  # source target B absent from the file
        }
        path = tmp_path / "sum.json"
        path.write_text(json.dumps(envelope), encoding="utf-8")

        dest, _ = make_store()
        report = MemoryImporter(dest, embedder=RecordingEmbedder()).from_json(str(path))
        imported = by_content(dest)["SUM"]
        assert imported.source_entry_ids == []
        assert any(b.id in err for err in report.errors)
        assert report.entries_imported == 1

    def test_export_paginates_beyond_one_page(self, tmp_path: Path) -> None:
        """More than one _PAGE of entries exports fully (covers the pagination
        continuation in _iter_entries)."""
        src, _ = make_store()
        items = [
            {"content": f"e{i}", "type": MemoryType.FACT, "source": SourceInfo(agent_id="a")}
            for i in range(501)
        ]
        src.batch_create(items)
        path = tmp_path / "big.json"
        MemoryExporter(src, allowed_root=tmp_path).to_json(str(path))
        env = json.loads(path.read_text(encoding="utf-8"))
        assert env["entry_count"] == 501
        assert len(env["entries"]) == 501

    def test_reserved_session_key_round_trips_via_restore(self, tmp_path: Path) -> None:
        """4b: a lifecycle ``session:``-keyed marker — which callers can never
        store() (SEC-SEV-001) — round-trips through export→import because the
        importer persists via restore(), which bypasses the create-path guard.
        A resume note therefore survives a backup/restore cycle."""
        marker = make_entry(id="m1", content="resume note for agent-x", key="session:agent-x:run-1")
        envelope = {
            "format": "tulving-export",
            "format_version": 1,
            "schema_version": 1,
            "entries": [marker.to_dict()],
        }
        path = tmp_path / "session.json"
        path.write_text(json.dumps(envelope), encoding="utf-8")

        dest, _ = make_store()
        report = MemoryImporter(dest, embedder=RecordingEmbedder()).from_json(str(path))
        assert report.entries_imported == 1
        held = dest.get_by_key("session:agent-x:run-1", touch=False)
        assert held is not None
        assert held.content == "resume note for agent-x"
