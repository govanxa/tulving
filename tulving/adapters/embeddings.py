"""Embedding adapters: protocol, identity, and the v0.1 implementations.

Model identity is load-bearing: vectors from different models are mutually
incompatible EVEN AT THE SAME DIMENSION. Every adapter declares a stable
``model_id``; the semantic index persists it in the meta table and refuses
to load against a different identity without an explicit ``rebuild()``.

Optional dependencies are gated inside constructors (D9): importing this
module on a core-only install always succeeds. Provider/runtime failures
propagate as-is — the semantic index (the only in-repo caller) wraps them
into ``VectorIndexError``; this module raises only ``ConfigError`` and
``SecurityError`` (D6).
"""

from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable

from tulving.exceptions import ConfigError
from tulving.security import credential_from_env, redact_secrets, reject_inline_credential

_MAX_U32: Final[int] = 2**32 - 1


@runtime_checkable
class EmbeddingAdapter(Protocol):
    """Anything that turns text into fixed-dimension vectors and declares its identity."""

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts efficiently. Order-preserving; ``len(out) == len(texts)``."""

    @property
    def model_id(self) -> str:
        """Stable identity of the exact model (e.g. 'sentence-transformers/all-MiniLM-L6-v2').

        MUST differ between any two models whose vector spaces differ.
        """

    @property
    def dimension(self) -> int:
        """Vector dimensionality."""

    @property
    def distance_metric(self) -> str:
        """Metric the vectors are meant for. v0.1: 'cosine' for all shipped adapters."""

    @property
    def normalizes(self) -> bool:
        """True if returned vectors are unit-normalized.

        If False, the semantic index normalizes before persistence/insertion
        (cosine requires it).
        """


@dataclass(frozen=True)
class EmbeddingIdentity:
    """The (model_id, dimension, distance_metric) triple persisted in the meta table."""

    model_id: str
    dimension: int
    distance_metric: str

    @classmethod
    def from_adapter(cls, adapter: EmbeddingAdapter) -> EmbeddingIdentity:
        """Package an adapter's identity triple for the semantic index to persist/compare.

        Args:
            adapter: The active embedding adapter.

        Returns:
            A frozen, value-compared identity triple.
        """
        return cls(
            model_id=adapter.model_id,
            dimension=adapter.dimension,
            distance_metric=adapter.distance_metric,
        )


class HashEmbedder:
    """Deterministic hash-based vectors for TESTS — no semantic meaning whatsoever.

    Identical text -> identical vector; similar text -> unrelated vector.
    Similarity search over these vectors is meaningless except that exact
    text equality implies vector equality. Never use as a real embedder;
    ``Memory`` never selects it implicitly.

    Algorithm (pinned by golden test — any change requires bumping the
    ``hash-embedder-v1`` version tag in ``model_id``):
    ``shake_128(utf-8 text).digest(4 * dimension)`` split into 4-byte
    big-endian chunks, mapped affinely to ``[-1.0, 1.0]``, then L2-normalized.
    """

    def __init__(self, dimension: int = 32) -> None:
        """Create a hash embedder.

        Args:
            dimension: Vector dimensionality; must be at least 2 (hnswlib
                needs sane dimensions — keeps tests honest).

        Raises:
            ConfigError: If ``dimension`` is below 2.
        """
        if dimension < 2:
            raise ConfigError("HashEmbedder dimension must be at least 2")
        self._dimension = dimension

    @property
    def model_id(self) -> str:
        """Versioned identity; embeds the dimension so different sizes never mix."""
        return f"tulving/hash-embedder-v1-{self._dimension}"

    @property
    def dimension(self) -> int:
        """Vector dimensionality (constructor argument)."""
        return self._dimension

    @property
    def distance_metric(self) -> str:
        """Always 'cosine'."""
        return "cosine"

    @property
    def normalizes(self) -> bool:
        """Vectors are L2-normalized; always True."""
        return True

    def embed(self, text: str) -> list[float]:
        """Embed one text as a deterministic unit-normalized hash vector.

        Args:
            text: Any string (empty allowed).

        Returns:
            A unit-normalized vector of ``dimension`` floats.
        """
        digest = hashlib.shake_128(text.encode("utf-8")).digest(4 * self._dimension)
        components = [
            int.from_bytes(digest[i : i + 4], "big") / _MAX_U32 * 2.0 - 1.0
            for i in range(0, len(digest), 4)
        ]
        norm = math.sqrt(sum(component * component for component in components))
        if norm == 0.0:  # pragma: no cover - unreachable in practice; guarded anyway
            components[0] = 1.0
            norm = 1.0
        return [component / norm for component in components]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed each text independently; order-preserving.

        Args:
            texts: Texts to embed (empty list allowed).

        Returns:
            One vector per input text, in input order.
        """
        return [self.embed(text) for text in texts]


class LocalEmbedder:
    """sentence-transformers embedder. Requires ``pip install tulving[local]``.

    First use downloads the model from HuggingFace (~90 MB for the default) —
    so "no network" holds only AFTER the first-run fetch, and the first
    ``embed()``/``dimension`` access is slow. Call ``warmup()`` to front-load
    this; opening a store with a cold ``LocalEmbedder`` is slow the first time
    because the semantic index reads ``dimension`` during open.

    Input longer than the model's sequence limit is silently truncated by
    sentence-transformers; v0.1 adds no countermeasures.
    """

    DEFAULT_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, *, model_name: str = DEFAULT_MODEL) -> None:
        """Validate the optional dependency; never load the model (D8).

        Parameters are keyword-only so a credential accidentally passed
        positionally can never land in the model slot (and thence into an
        error message).

        Args:
            model_name: HuggingFace model identifier — this IS the stable
                ``model_id`` persisted in the meta table.

        Raises:
            ConfigError: If ``sentence-transformers`` is not installed.
        """
        try:
            import sentence_transformers
        except ImportError as exc:
            raise ConfigError(
                "LocalEmbedder requires the 'sentence-transformers' package; "
                "install it with: pip install tulving[local]"
            ) from exc
        self._st: Any = sentence_transformers
        self._model_name = model_name
        self._st_model: Any | None = None
        self._load_lock = threading.Lock()

    def warmup(self) -> None:
        """Force model load/download now instead of on first embed."""
        self._model()

    def _model(self) -> Any:
        """Lazy singleton model load, lock-guarded against concurrent first calls."""
        if self._st_model is None:
            with self._load_lock:
                if self._st_model is None:
                    self._st_model = self._st.SentenceTransformer(self._model_name)
        return self._st_model

    @property
    def model_id(self) -> str:
        """The HuggingFace identifier — the stable model identity. Never loads."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Vector dimensionality. NOTE: first access triggers model load/download.

        Raises:
            ConfigError: If the loaded model does not report a fixed dimension
                (exotic models can return None). The model name in the message
                is passed through ``redact_secrets`` first.
        """
        value = self._model().get_sentence_embedding_dimension()
        if value is None:
            raise ConfigError(
                f"model '{redact_secrets(self._model_name)}' does not report a "
                "sentence-embedding dimension; LocalEmbedder requires a model "
                "with a fixed dimension"
            )
        return int(value)

    @property
    def distance_metric(self) -> str:
        """Always 'cosine'."""
        return "cosine"

    @property
    def normalizes(self) -> bool:
        """Vectors are encoded with ``normalize_embeddings=True``; always True."""
        return True

    def embed(self, text: str) -> list[float]:
        """Embed one text. First call may trigger the model load/download.

        Args:
            text: The text to embed.

        Returns:
            A unit-normalized vector of plain Python floats.
        """
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts in one encoder pass; order-preserving.

        Args:
            texts: Texts to embed.

        Returns:
            One unit-normalized vector per input text, in input order.
        """
        vectors: list[list[float]] = self._model().encode(texts, normalize_embeddings=True).tolist()
        return vectors


_OPENAI_MODEL_DIMENSIONS: Final[dict[str, int]] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
_MAX_BATCH: Final[int] = 512


class OpenAIEmbedder:
    """OpenAI embeddings. Requires ``pip install tulving[openai]`` and ``OPENAI_API_KEY``.

    Sends memory content to the OpenAI API by design — do not use it for
    content that must never leave the machine. Provider exceptions (rate
    limits, network, over-length input) propagate unwrapped.
    """

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        """Validate credentials and config; open no network connection (D8).

        Parameters are keyword-only so a credential accidentally passed
        positionally can never land in the model slot (and thence into an
        error message).

        Args:
            model: One of the known OpenAI embedding models (the dimension
                must be knowable offline).
            api_key: MUST be None. Exists only to be rejected loudly —
                credentials come from the environment (ADR-010 #4).

        Raises:
            SecurityError: If ``api_key`` is passed inline (checked FIRST).
            ConfigError: If ``openai`` is not installed, ``model`` is unknown
                (the message never echoes the unknown value — it could be a
                mis-pasted credential), or ``OPENAI_API_KEY`` is unset/blank.
        """
        reject_inline_credential(api_key, adapter_name="OpenAIEmbedder")
        try:
            import openai
        except ImportError as exc:
            raise ConfigError(
                "OpenAIEmbedder requires the 'openai' package; "
                "install it with: pip install tulving[openai]"
            ) from exc
        if model not in _OPENAI_MODEL_DIMENSIONS:
            known = ", ".join(sorted(_OPENAI_MODEL_DIMENSIONS))
            raise ConfigError(f"unknown OpenAI embedding model; known models: {known}")
        key = credential_from_env("OPENAI_API_KEY", adapter_name="OpenAIEmbedder")
        self._model_name = model
        self._client: Any = openai.OpenAI(api_key=key)

    @property
    def model_id(self) -> str:
        """The OpenAI model name — the stable model identity."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Vector dimensionality from the known-model table (no network call)."""
        return _OPENAI_MODEL_DIMENSIONS[self._model_name]

    @property
    def distance_metric(self) -> str:
        """Always 'cosine'."""
        return "cosine"

    @property
    def normalizes(self) -> bool:
        """OpenAI returns unit vectors; always True."""
        return True

    def embed(self, text: str) -> list[float]:
        """Embed one text via the OpenAI API.

        Args:
            text: The text to embed.

        Returns:
            A unit-normalized vector of plain Python floats.
        """
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts, chunked at the API's per-request input limit.

        Reassembles results by each datum's ``.index`` (the API does not
        guarantee arrival order), preserving input order across chunks.

        Args:
            texts: Texts to embed (empty list returns empty without an API call).

        Returns:
            One vector per input text, in input order.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _MAX_BATCH):
            chunk = texts[start : start + _MAX_BATCH]
            response = self._client.embeddings.create(model=self._model_name, input=chunk)
            ordered = sorted(response.data, key=lambda datum: int(datum.index))
            vectors.extend([float(value) for value in datum.embedding] for datum in ordered)
        return vectors
