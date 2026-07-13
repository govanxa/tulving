"""Tests for tulving.adapters.embeddings — written BEFORE implementation.

Constraints (blueprint-embeddings.md): no test imports torch, downloads a
model, or touches the network. ``sentence_transformers`` and ``openai`` are
always mocked/monkeypatched (they are NOT installed in the dev environment);
``HashEmbedder`` is exercised for real — determinism is its job.
"""

from __future__ import annotations

import dataclasses
import math
import sys
import threading
from typing import Any, ClassVar

import pytest

from tulving.adapters.embeddings import (
    _MAX_BATCH,
    _OPENAI_MODEL_DIMENSIONS,
    EmbeddingAdapter,
    EmbeddingIdentity,
    HashEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
)
from tulving.exceptions import ConfigError, SecurityError

INLINE_KEY = "sk-inline123456789012345"

# Pinned golden vector: HashEmbedder(8).embed("tulving"). Computed from the
# blueprint-pinned algorithm (shake_128 -> 4-byte big-endian chunks -> affine
# [-1, 1] -> L2 normalize). If this test EVER breaks, the vector math changed
# and model_id must be bumped hash-embedder-v1 -> v2.
GOLDEN_TULVING_8 = [
    0.5001475472501232,
    -0.10347686485483314,
    0.5051344033050725,
    -0.1451008626020262,
    -0.1553222206761857,
    -0.32049221312640563,
    0.2468182591088105,
    0.5245669068377548,
]


# ---------------------------------------------------------------------------
# Fakes for the gated optional dependencies (never the real packages)
# ---------------------------------------------------------------------------


class FakeEncodedArray:
    """Mimics the numpy array sentence-transformers returns from encode()."""

    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def tolist(self) -> list[list[float]]:
        return [list(row) for row in self._rows]


class FakeSentenceTransformer:
    """Records construction and encode() calls; returns constant vectors."""

    instances: ClassVar[list[FakeSentenceTransformer]] = []

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.encode_calls: list[tuple[list[str], dict[str, Any]]] = []
        FakeSentenceTransformer.instances.append(self)

    def get_sentence_embedding_dimension(self) -> int:
        return 384

    def encode(self, texts: list[str], **kwargs: Any) -> FakeEncodedArray:
        self.encode_calls.append((list(texts), dict(kwargs)))
        return FakeEncodedArray([[0.6, 0.8, 0.0] for _ in texts])


class FakeSentenceTransformersModule:
    """Stands in for the ``sentence_transformers`` module in sys.modules."""

    SentenceTransformer = FakeSentenceTransformer


@pytest.fixture
def fake_st(monkeypatch: pytest.MonkeyPatch) -> type[FakeSentenceTransformer]:
    """Install a fake sentence_transformers module; reset recorded instances."""
    FakeSentenceTransformer.instances = []
    monkeypatch.setitem(sys.modules, "sentence_transformers", FakeSentenceTransformersModule())
    return FakeSentenceTransformer


class FakeSentenceTransformerNewAPI:
    """Only exposes the new (>=5.x) ``get_embedding_dimension`` name.

    Deliberately has NO ``get_sentence_embedding_dimension`` attribute at all,
    so that any code path touching the old name fails honestly with
    AttributeError instead of silently succeeding.
    """

    instances: ClassVar[list[FakeSentenceTransformerNewAPI]] = []

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.encode_calls: list[tuple[list[str], dict[str, Any]]] = []
        FakeSentenceTransformerNewAPI.instances.append(self)

    def get_embedding_dimension(self) -> int:
        return 384

    def encode(self, texts: list[str], **kwargs: Any) -> FakeEncodedArray:
        self.encode_calls.append((list(texts), dict(kwargs)))
        return FakeEncodedArray([[0.6, 0.8, 0.0] for _ in texts])


class FakeSentenceTransformersModuleNewAPI:
    """Fake module whose model only speaks the new dimension API."""

    SentenceTransformer = FakeSentenceTransformerNewAPI


@pytest.fixture
def fake_st_new_api(monkeypatch: pytest.MonkeyPatch) -> type[FakeSentenceTransformerNewAPI]:
    """Install a fake sentence_transformers module exposing only the new API name."""
    FakeSentenceTransformerNewAPI.instances = []
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", FakeSentenceTransformersModuleNewAPI()
    )
    return FakeSentenceTransformerNewAPI


class FakeEmbeddingDatum:
    def __init__(self, index: int, embedding: list[float]) -> None:
        self.index = index
        self.embedding = embedding


class FakeEmbeddingsResponse:
    def __init__(self, data: list[FakeEmbeddingDatum]) -> None:
        self.data = data


class FakeOpenAIEmbeddings:
    """Returns per-input vectors with SHUFFLED .index order (reversed) to
    prove the adapter reassembles by index, not arrival order."""

    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.raise_on_create: Exception | None = None

    def create(self, *, model: str, input: list[str]) -> FakeEmbeddingsResponse:
        self.create_calls.append({"model": model, "input": list(input)})
        if self.raise_on_create is not None:
            raise self.raise_on_create
        data = [
            FakeEmbeddingDatum(index=i, embedding=[float(i), float(i) + 0.5])
            for i in range(len(input))
        ]
        return FakeEmbeddingsResponse(list(reversed(data)))


class FakeOpenAIClient:
    instances: ClassVar[list[FakeOpenAIClient]] = []

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.embeddings = FakeOpenAIEmbeddings()
        FakeOpenAIClient.instances.append(self)


class FakeOpenAIModule:
    OpenAI = FakeOpenAIClient


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> type[FakeOpenAIClient]:
    """Install a fake openai module + env key; reset recorded instances."""
    FakeOpenAIClient.instances = []
    monkeypatch.setitem(sys.modules, "openai", FakeOpenAIModule())
    monkeypatch.setenv("OPENAI_API_KEY", "env-key-from-monkeypatch")
    return FakeOpenAIClient


def _block_module(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Force ``import name`` to raise ImportError even if installed."""
    monkeypatch.setitem(sys.modules, name, None)


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestFailurePaths:
    def test_local_embedder_without_extra_raises_config_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _block_module(monkeypatch, "sentence_transformers")
        with pytest.raises(ConfigError) as excinfo:
            LocalEmbedder()
        assert "tulving[local]" in str(excinfo.value)

    def test_openai_embedder_without_extra_raises_config_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _block_module(monkeypatch, "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "env-key-set")
        with pytest.raises(ConfigError) as excinfo:
            OpenAIEmbedder()
        assert "tulving[openai]" in str(excinfo.value)

    def test_openai_embedder_missing_env_key_raises_config_error(
        self, monkeypatch: pytest.MonkeyPatch, fake_openai: type[FakeOpenAIClient]
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ConfigError) as excinfo:
            OpenAIEmbedder()
        assert "OPENAI_API_KEY" in str(excinfo.value)
        assert fake_openai.instances == []  # no client constructed

    def test_openai_embedder_blank_env_key_raises_config_error(
        self, monkeypatch: pytest.MonkeyPatch, fake_openai: type[FakeOpenAIClient]
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "   ")
        with pytest.raises(ConfigError) as excinfo:
            OpenAIEmbedder()
        assert "OPENAI_API_KEY" in str(excinfo.value)
        assert fake_openai.instances == []

    def test_openai_embedder_unknown_model_raises_config_error(
        self, fake_openai: type[FakeOpenAIClient]
    ) -> None:
        with pytest.raises(ConfigError):
            OpenAIEmbedder(model="text-embedding-nonexistent")
        assert fake_openai.instances == []

    @pytest.mark.parametrize("dimension", [1, 0, -3])
    def test_hash_embedder_dimension_below_two_raises_config_error(self, dimension: int) -> None:
        with pytest.raises(ConfigError):
            HashEmbedder(dimension=dimension)

    def test_openai_provider_exception_propagates_unwrapped(
        self, fake_openai: type[FakeOpenAIClient]
    ) -> None:
        embedder = OpenAIEmbedder()
        boom = RuntimeError("rate limited")
        fake_openai.instances[0].embeddings.raise_on_create = boom
        with pytest.raises(RuntimeError) as excinfo:
            embedder.embed("text")
        assert excinfo.value is boom


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


class TestSecurity:
    def test_inline_api_key_raises_security_error(self) -> None:
        with pytest.raises(SecurityError):
            OpenAIEmbedder(api_key=INLINE_KEY)

    def test_inline_key_rejection_message_never_echoes_key(self) -> None:
        with pytest.raises(SecurityError) as excinfo:
            OpenAIEmbedder(api_key=INLINE_KEY)
        assert INLINE_KEY not in str(excinfo.value)

    def test_inline_key_rejected_before_import_and_env_checks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rejection is constructor step 1: fires even with openai missing
        AND the env var set."""
        _block_module(monkeypatch, "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "env-key-set")
        with pytest.raises(SecurityError):
            OpenAIEmbedder(api_key=INLINE_KEY)

    def test_module_import_needs_no_optional_dependencies(self) -> None:
        """No module-level import of the optional deps — asserted on the
        SOURCE (ast), so the test cannot fail spuriously when openai/torch
        happen to be installed and already imported in the session."""
        import ast
        import inspect

        import tulving.adapters.embeddings as embeddings_module

        forbidden = {"sentence_transformers", "torch", "openai"}
        tree = ast.parse(inspect.getsource(embeddings_module))
        for node in tree.body:  # top level only; gated in-function imports are fine
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
                assert not (roots & forbidden), f"module-level import of {roots & forbidden}"
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                assert root not in forbidden, f"module-level import from {root}"

    def test_sk_shaped_model_value_never_echoed_in_error(
        self, fake_openai: type[FakeOpenAIClient]
    ) -> None:
        """Realistic misuse: a credential mis-pasted into the model slot must
        never be echoed by the unknown-model ConfigError."""
        mispasted = "sk-mispasted12345678901234567890"
        with pytest.raises(ConfigError) as excinfo:
            OpenAIEmbedder(model=mispasted)
        assert mispasted not in str(excinfo.value)
        assert "known models" in str(excinfo.value)

    def test_openai_constructor_rejects_positional_arguments(self) -> None:
        """Keyword-only params: a key passed positionally can never land in
        the model slot."""
        with pytest.raises(TypeError):
            OpenAIEmbedder("sk-positional1234567890123456")  # type: ignore[misc]

    def test_local_constructor_rejects_positional_arguments(self) -> None:
        with pytest.raises(TypeError):
            LocalEmbedder("sk-positional1234567890123456")  # type: ignore[misc]

    def test_local_dimension_error_redacts_sk_shaped_model_name(
        self,
        fake_st: type[FakeSentenceTransformer],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The dimension-unavailable ConfigError names the model, but only
        after redact_secrets — an sk-shaped name never leaks."""
        monkeypatch.setattr(
            FakeSentenceTransformer,
            "get_sentence_embedding_dimension",
            lambda self: None,
        )
        sk_name = "sk-notreallyamodel123456789012345"
        embedder = LocalEmbedder(model_name=sk_name)
        with pytest.raises(ConfigError) as excinfo:
            _ = embedder.dimension
        assert sk_name not in str(excinfo.value)


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


class TestBoundaryConditions:
    def test_hash_embedder_minimum_dimension_two_is_valid(self) -> None:
        vector = HashEmbedder(dimension=2).embed("x")
        assert len(vector) == 2

    def test_hash_embedder_default_dimension_is_32(self) -> None:
        """QA addition: the blueprint pins the default at 32; changing it
        silently changes model_id for every default-constructed embedder."""
        embedder = HashEmbedder()
        assert embedder.dimension == 32
        assert embedder.model_id == "tulving/hash-embedder-v1-32"
        assert len(embedder.embed("default")) == 32

    def test_hash_embedder_embed_batch_empty_returns_empty(self) -> None:
        assert HashEmbedder(8).embed_batch([]) == []

    def test_hash_embedder_empty_string_embeds_fine(self) -> None:
        vector = HashEmbedder(8).embed("")
        assert len(vector) == 8
        assert math.isclose(math.dist(vector, [0.0] * 8), 1.0, rel_tol=1e-9)

    def test_hash_embedder_unicode_content(self) -> None:
        a = HashEmbedder(8).embed("héllo wörld é")
        b = HashEmbedder(8).embed("héllo wörld é")
        assert a == b

    def test_openai_embed_batch_empty_returns_empty_without_api_call(
        self, fake_openai: type[FakeOpenAIClient]
    ) -> None:
        embedder = OpenAIEmbedder()
        assert embedder.embed_batch([]) == []
        assert fake_openai.instances[0].embeddings.create_calls == []

    def test_openai_batch_chunking_at_max_batch_plus_three(
        self, fake_openai: type[FakeOpenAIClient]
    ) -> None:
        embedder = OpenAIEmbedder()
        texts = [f"text-{i}" for i in range(_MAX_BATCH + 3)]
        vectors = embedder.embed_batch(texts)
        calls = fake_openai.instances[0].embeddings.create_calls
        assert len(calls) == 2
        assert len(calls[0]["input"]) == _MAX_BATCH
        assert len(calls[1]["input"]) == 3
        assert len(vectors) == _MAX_BATCH + 3
        # Fake returns data reversed by .index — order must be restored per chunk.
        assert vectors[0] == [0.0, 0.5]
        assert vectors[1] == [1.0, 1.5]
        assert vectors[_MAX_BATCH] == [0.0, 0.5]  # first element of second chunk
        assert vectors[_MAX_BATCH + 2] == [2.0, 2.5]

    def test_openai_batch_of_exactly_max_batch_makes_one_call(
        self, fake_openai: type[FakeOpenAIClient]
    ) -> None:
        """QA addition: exact chunk boundary — no empty trailing request."""
        embedder = OpenAIEmbedder()
        vectors = embedder.embed_batch([f"text-{i}" for i in range(_MAX_BATCH)])
        calls = fake_openai.instances[0].embeddings.create_calls
        assert len(calls) == 1
        assert len(calls[0]["input"]) == _MAX_BATCH
        assert len(vectors) == _MAX_BATCH


# ---------------------------------------------------------------------------
# Basic behavior — HashEmbedder (real execution)
# ---------------------------------------------------------------------------


class TestHashEmbedderBehavior:
    def test_deterministic_across_instances(self) -> None:
        assert HashEmbedder(16).embed("same text") == HashEmbedder(16).embed("same text")

    def test_golden_vector_pins_the_algorithm(self) -> None:
        vector = HashEmbedder(8).embed("tulving")
        assert vector == pytest.approx(GOLDEN_TULVING_8, rel=1e-12, abs=1e-12)

    def test_dimension_respected(self) -> None:
        for dimension in (2, 8, 32, 384):
            assert len(HashEmbedder(dimension).embed("text")) == dimension

    def test_vectors_are_unit_normalized(self) -> None:
        vector = HashEmbedder(32).embed("normalize me")
        norm = math.sqrt(sum(component**2 for component in vector))
        assert math.isclose(norm, 1.0, rel_tol=1e-9)

    def test_embed_batch_equals_per_item_embed(self) -> None:
        embedder = HashEmbedder(8)
        assert embedder.embed_batch(["a", "b"]) == [embedder.embed("a"), embedder.embed("b")]

    def test_different_texts_give_different_vectors(self) -> None:
        embedder = HashEmbedder(8)
        assert embedder.embed("alpha") != embedder.embed("beta")

    def test_model_id_versioned_and_embeds_dimension(self) -> None:
        assert HashEmbedder(16).model_id != HashEmbedder(32).model_id
        assert HashEmbedder(32).model_id == "tulving/hash-embedder-v1-32"

    def test_identity_properties(self) -> None:
        embedder = HashEmbedder(8)
        assert embedder.dimension == 8
        assert embedder.distance_metric == "cosine"
        assert embedder.normalizes is True

    def test_runtime_protocol_conformance(self) -> None:
        assert isinstance(HashEmbedder(8), EmbeddingAdapter)


# ---------------------------------------------------------------------------
# Basic behavior — LocalEmbedder (mocked sentence_transformers)
# ---------------------------------------------------------------------------


class TestLocalEmbedderBehavior:
    def test_constructor_does_not_load_model(self, fake_st: type[FakeSentenceTransformer]) -> None:
        LocalEmbedder()
        assert fake_st.instances == []  # D8: cheap construction

    def test_model_id_is_the_model_name_without_loading(
        self, fake_st: type[FakeSentenceTransformer]
    ) -> None:
        embedder = LocalEmbedder(model_name="org/custom-model")
        assert embedder.model_id == "org/custom-model"
        assert fake_st.instances == []

    def test_default_model_name(self, fake_st: type[FakeSentenceTransformer]) -> None:
        assert LocalEmbedder().model_id == "sentence-transformers/all-MiniLM-L6-v2"

    def test_model_loaded_exactly_once_across_warmup_and_embeds(
        self, fake_st: type[FakeSentenceTransformer]
    ) -> None:
        embedder = LocalEmbedder()
        embedder.warmup()
        embedder.embed("one")
        embedder.embed("two")
        assert len(fake_st.instances) == 1

    def test_dimension_triggers_load_and_returns_model_value(
        self, fake_st: type[FakeSentenceTransformer]
    ) -> None:
        embedder = LocalEmbedder()
        assert embedder.dimension == 384
        assert len(fake_st.instances) == 1

    def test_dimension_uses_old_api_name_when_only_that_is_present(
        self, fake_st: type[FakeSentenceTransformer]
    ) -> None:
        """Compat shim: old sentence-transformers exposing only
        ``get_sentence_embedding_dimension`` (no ``get_embedding_dimension``
        at all) must still resolve the dimension."""
        assert not hasattr(FakeSentenceTransformer, "get_embedding_dimension")
        embedder = LocalEmbedder()
        assert embedder.dimension == 384

    def test_dimension_prefers_new_api_name_when_present(
        self, fake_st_new_api: type[FakeSentenceTransformerNewAPI]
    ) -> None:
        """Compat shim: new sentence-transformers exposing only
        ``get_embedding_dimension`` (no ``get_sentence_embedding_dimension``
        at all, so touching the old name would raise AttributeError) must
        resolve the dimension without a FutureWarning."""
        assert not hasattr(FakeSentenceTransformerNewAPI, "get_sentence_embedding_dimension")
        embedder = LocalEmbedder()
        assert embedder.dimension == 384
        assert len(fake_st_new_api.instances) == 1

    def test_dimension_none_raises_config_error_naming_model(
        self,
        fake_st: type[FakeSentenceTransformer],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Exotic models can report no dimension: ConfigError, not TypeError."""
        monkeypatch.setattr(
            FakeSentenceTransformer,
            "get_sentence_embedding_dimension",
            lambda self: None,
        )
        embedder = LocalEmbedder(model_name="org/exotic-model")
        with pytest.raises(ConfigError) as excinfo:
            _ = embedder.dimension
        assert "org/exotic-model" in str(excinfo.value)

    def test_embed_routes_through_encode_with_normalize(
        self, fake_st: type[FakeSentenceTransformer]
    ) -> None:
        embedder = LocalEmbedder()
        vector = embedder.embed("hello")
        assert vector == [0.6, 0.8, 0.0]
        assert all(isinstance(component, float) for component in vector)
        texts, kwargs = fake_st.instances[0].encode_calls[0]
        assert texts == ["hello"]
        assert kwargs.get("normalize_embeddings") is True

    def test_embed_batch_order_preserving(self, fake_st: type[FakeSentenceTransformer]) -> None:
        embedder = LocalEmbedder()
        vectors = embedder.embed_batch(["a", "b", "c"])
        assert len(vectors) == 3
        texts, _ = fake_st.instances[0].encode_calls[0]
        assert texts == ["a", "b", "c"]

    def test_concurrent_first_use_loads_once(self, fake_st: type[FakeSentenceTransformer]) -> None:
        embedder = LocalEmbedder()
        barrier = threading.Barrier(4)

        def race() -> None:
            barrier.wait()
            embedder.embed("race")

        threads = [threading.Thread(target=race) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(fake_st.instances) == 1

    def test_double_checked_locking_skips_load_when_another_thread_won(
        self, fake_st: type[FakeSentenceTransformer]
    ) -> None:
        """Deterministic race: a lock stand-in installs the model on acquire,
        proving the inner None-check prevents a second load."""
        embedder = LocalEmbedder()
        winner = fake_st("winner-model")

        class WinnerLock:
            def __enter__(self) -> None:
                embedder._st_model = winner  # simulates the losing thread's view

            def __exit__(self, *args: object) -> None:
                return None

        embedder._load_lock = WinnerLock()  # type: ignore[assignment]
        embedder.warmup()
        assert len(fake_st.instances) == 1  # only the winner; no second load
        assert embedder.embed("x") == [0.6, 0.8, 0.0]

    def test_identity_properties(self, fake_st: type[FakeSentenceTransformer]) -> None:
        embedder = LocalEmbedder()
        assert embedder.distance_metric == "cosine"
        assert embedder.normalizes is True

    def test_runtime_protocol_conformance(self, fake_st: type[FakeSentenceTransformer]) -> None:
        assert isinstance(LocalEmbedder(), EmbeddingAdapter)


# ---------------------------------------------------------------------------
# Basic behavior — OpenAIEmbedder (mocked openai)
# ---------------------------------------------------------------------------


class TestOpenAIEmbedderBehavior:
    def test_embed_returns_mocked_vector(self, fake_openai: type[FakeOpenAIClient]) -> None:
        embedder = OpenAIEmbedder()
        assert embedder.embed("hello") == [0.0, 0.5]

    def test_client_constructed_with_env_key(self, fake_openai: type[FakeOpenAIClient]) -> None:
        OpenAIEmbedder()
        assert fake_openai.instances[0].api_key == "env-key-from-monkeypatch"

    def test_known_model_dimension_table(self, fake_openai: type[FakeOpenAIClient]) -> None:
        assert OpenAIEmbedder().dimension == 1536
        assert OpenAIEmbedder(model="text-embedding-3-large").dimension == 3072
        assert OpenAIEmbedder(model="text-embedding-ada-002").dimension == 1536
        assert _OPENAI_MODEL_DIMENSIONS["text-embedding-3-small"] == 1536

    def test_model_id_is_the_model_name(self, fake_openai: type[FakeOpenAIClient]) -> None:
        assert OpenAIEmbedder().model_id == "text-embedding-3-small"
        assert OpenAIEmbedder(model="text-embedding-3-large").model_id == "text-embedding-3-large"

    def test_identity_properties(self, fake_openai: type[FakeOpenAIClient]) -> None:
        embedder = OpenAIEmbedder()
        assert embedder.distance_metric == "cosine"
        assert embedder.normalizes is True

    def test_api_called_with_configured_model(self, fake_openai: type[FakeOpenAIClient]) -> None:
        OpenAIEmbedder(model="text-embedding-3-large").embed("x")
        assert fake_openai.instances[0].embeddings.create_calls[0]["model"] == (
            "text-embedding-3-large"
        )

    def test_runtime_protocol_conformance(self, fake_openai: type[FakeOpenAIClient]) -> None:
        assert isinstance(OpenAIEmbedder(), EmbeddingAdapter)


# ---------------------------------------------------------------------------
# EmbeddingIdentity
# ---------------------------------------------------------------------------


class TestEmbeddingIdentity:
    def test_from_adapter_round_trips_the_triple(self) -> None:
        identity = EmbeddingIdentity.from_adapter(HashEmbedder(8))
        assert identity.model_id == "tulving/hash-embedder-v1-8"
        assert identity.dimension == 8
        assert identity.distance_metric == "cosine"

    def test_frozen_assignment_raises(self) -> None:
        identity = EmbeddingIdentity.from_adapter(HashEmbedder(8))
        with pytest.raises(dataclasses.FrozenInstanceError):
            identity.dimension = 16  # type: ignore[misc]

    def test_equality_and_hash_by_value(self) -> None:
        a = EmbeddingIdentity(model_id="m", dimension=4, distance_metric="cosine")
        b = EmbeddingIdentity(model_id="m", dimension=4, distance_metric="cosine")
        c = EmbeddingIdentity(model_id="other", dimension=4, distance_metric="cosine")
        assert a == b
        assert hash(a) == hash(b)
        assert a != c

    def test_detects_same_dimension_model_swap(self) -> None:
        """The whole point of model identity: same dimension, different model
        -> different identity (semantic index must force rebuild)."""
        a = EmbeddingIdentity(model_id="model-a", dimension=384, distance_metric="cosine")
        b = EmbeddingIdentity(model_id="model-b", dimension=384, distance_metric="cosine")
        assert a != b
