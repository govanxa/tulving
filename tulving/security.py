"""Security primitives: redaction, leaf-name validation, path containment.

Tulving does NOT encrypt at rest in v0.1 (ADR-010 revised): do not store
secrets you cannot afford on disk. Redaction protects *outgoing* text
(curated context, exports, MCP responses); raw values remain readable via
direct ``memory.get()``.

Internal module — not exported from ``tulving/__init__.py``. Consumers
import as ``from tulving.security import ...``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

from tulving.exceptions import ConfigError, SecurityError

REDACTED: Final[str] = "[REDACTED]"

# Key-NAME patterns (ADR-010 list). Matched as whole alphanumeric segments,
# case-insensitive: boundaries are non-alphanumeric characters or string
# edges, so "api_key" matches "key" but "monkey"/"keyboard" do not.
DEFAULT_SENSITIVE_KEY_PATTERNS: Final[tuple[str, ...]] = (
    "password",
    "passwd",
    "secret",
    "token",
    "key",
    "credential",
    "auth",
    "api_key",
    "apikey",
    "private",
)

# Token-SHAPE patterns for content scanning (D10): (name, regex, replacement).
# Anchored on structure, not dictionary words — safe against prose
# false-positives. No generic high-entropy detector in v0.1 (git SHAs and
# UUIDs must never be redacted).
_SECRET_SHAPES: Final[tuple[tuple[str, re.Pattern[str], str], ...]] = (
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), REDACTED),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"), REDACTED),
    # OpenAI/Anthropic style
    ("sk_api_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), REDACTED),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), REDACTED),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
        REDACTED,
    ),
    (
        "private_key_pem",
        # \Z alternative: a truncated PEM block missing its END line is still caught.
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----"
            r"(?:.|\n)*?"
            r"(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)"
        ),
        REDACTED,
    ),
    (
        "bearer_header",
        re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE),
        REDACTED,
    ),
    # Generic assignment: password=..., api_key: "...". Redacts the VALUE only
    # so the redacted output remains readable.
    (
        "kv_assignment",
        re.compile(
            r"(?i)(?P<prefix>(?<![a-z0-9])(?:password|passwd|secret|token|api_key|apikey"
            r"|credential|auth)(?![a-z0-9])\s*[:=]\s*[\"']?)(?P<value>[^\s\"',;]{4,})"
        ),
        rf"\g<prefix>{REDACTED}",
    ),
)

# Windows reserved device names — invalid/dangerous as filenames even when
# they pass the character whitelist. Rejected on ALL platforms so exported
# artifacts stay portable.
_WINDOWS_RESERVED: Final[frozenset[str]] = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }
)

_LEAF_RE: Final[re.Pattern[str]] = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


def compile_key_patterns(
    extra_patterns: Iterable[str] | None = None,
) -> tuple[re.Pattern[str], ...]:
    """Compile sensitive-key patterns with alphanumeric-boundary lookarounds.

    Each pattern ``p`` becomes ``(?<![a-zA-Z0-9])p(?![a-zA-Z0-9])`` (matched
    case-insensitively) with ``p`` escaped via ``re.escape()`` — user-supplied
    extras are treated as literals, never as regex. Underscore, hyphen, dot,
    and string edges therefore count as boundaries ("api_key" matches "key");
    adjacent letters do not ("monkey" does not match "key").

    Known limitation: camelCase ("authToken") does not match — callers pass
    their own ``sensitive_keys`` for that convention.

    Args:
        extra_patterns: Literal key names to compile in addition to
            ``DEFAULT_SENSITIVE_KEY_PATTERNS``.

    Returns:
        Compiled patterns for the defaults plus any extras.
    """
    names: tuple[str, ...] = DEFAULT_SENSITIVE_KEY_PATTERNS
    if extra_patterns is not None:
        names += tuple(extra_patterns)
    return tuple(
        re.compile(rf"(?<![a-zA-Z0-9]){re.escape(name)}(?![a-zA-Z0-9])", re.IGNORECASE)
        for name in names
    )


_DEFAULT_KEY_PATTERNS: Final[tuple[re.Pattern[str], ...]] = compile_key_patterns()


def _compile_labelled(
    patterns: Sequence[re.Pattern[str]],
) -> tuple[re.Pattern[str], ...]:
    """Derive ``<key> [:=] <value>`` value-masking regexes from key patterns."""
    return tuple(
        re.compile(
            rf"(?P<prefix>{pattern.pattern}\s*[:=]\s*[\"']?)(?P<value>[^\s\"',;]{{4,}})",
            re.IGNORECASE,
        )
        for pattern in patterns
    )


_DEFAULT_LABELLED: Final[tuple[re.Pattern[str], ...]] = _compile_labelled(_DEFAULT_KEY_PATTERNS)


def is_sensitive_key(
    key: str,
    patterns: Sequence[re.Pattern[str]] | None = None,
) -> bool:
    """Return True if any compiled sensitive-key pattern matches ``key``.

    Args:
        key: The key name to test.
        patterns: Compiled patterns; defaults to ``compile_key_patterns()``
            over ``DEFAULT_SENSITIVE_KEY_PATTERNS`` (compiled once at module
            level). Callers with ``Memory(sensitive_keys=[...])`` pass their
            augmented set.

    Returns:
        Whether the key is judged sensitive.
    """
    if patterns is None:
        patterns = _DEFAULT_KEY_PATTERNS
    return any(pattern.search(key) for pattern in patterns)


def redact_secrets(text: str) -> str:
    """Content-level scan (D10): replace every token-shape match with REDACTED.

    For ``kv_assignment`` only the value group is replaced (the label stays,
    so redacted output remains readable). Pure function; returns ``text``
    unchanged (same object) when nothing matches. Never raises. Matched
    secret values are never logged.

    Args:
        text: Arbitrary outgoing text.

    Returns:
        The text with every secret-shaped substring replaced by ``REDACTED``.
    """
    result = text
    for _name, shape, replacement in _SECRET_SHAPES:
        substituted, count = shape.subn(replacement, result)
        if count:
            result = substituted
    return result


def redact_text(
    text: str,
    *,
    key_patterns: Sequence[re.Pattern[str]] | None = None,
) -> str:
    """Full outgoing-text redaction: content shapes plus key-labelled values.

    Runs ``redact_secrets()`` and then masks values labelled by any sensitive
    key pattern (``<key> [:=] <value>``). This is THE function curator/export/
    MCP call before emitting any text (CLAUDE.md security req #1).

    Args:
        text: Arbitrary outgoing text.
        key_patterns: Compiled sensitive-key patterns; defaults to the
            module-level defaults.

    Returns:
        Redacted text; idempotent (re-redacting output is a no-op).
    """
    result = redact_secrets(text)
    labelled = _DEFAULT_LABELLED if key_patterns is None else _compile_labelled(key_patterns)
    for pattern in labelled:
        substituted, count = pattern.subn(rf"\g<prefix>{REDACTED}", result)
        if count:
            result = substituted
    return result


def validate_leaf_name(name: str) -> str:
    """Validate a leaf name (export file stem, key-as-filename) — NOT a path.

    Validates, never transforms — a sanitizer that rewrites names invites
    collision bugs. The error message never echoes the rejected name.

    Args:
        name: Candidate leaf name.

    Returns:
        ``name`` unchanged on success.

    Raises:
        SecurityError: If ``name`` is empty, longer than 128 chars, contains
            any character outside ``[a-zA-Z0-9_-]`` (so no ``/``, ``\\``,
            ``..``, ``:``, ADS suffixes, spaces, or dots — extensions are
            appended by trusted code AFTER validation), or case-insensitively
            equals a Windows reserved device name.
    """
    if not _LEAF_RE.fullmatch(name):
        raise SecurityError("invalid leaf name: must be 1-128 characters of [a-zA-Z0-9_-]")
    if name.lower() in _WINDOWS_RESERVED:
        raise SecurityError("invalid leaf name: Windows reserved device name")
    return name


def contain_path(path: str | Path, allowed_root: str | Path) -> Path:
    """Resolve ``path`` and require it to live under ``allowed_root``.

    Both arguments are ``expanduser()``-ed (``~`` support) and ``resolve()``-d
    (realpath: symlinks, junctions, ``..``, relative → absolute; non-strict so
    not-yet-created export targets validate). Containment is decided with
    ``os.path.commonpath`` over ``os.path.normcase``-d strings — Windows-aware
    (case-insensitive, drive letters, backslash/slash mixes, UNC roots).
    An exact match (path == root) is contained.

    Args:
        path: Candidate path (need not exist).
        allowed_root: Directory the path must stay inside. Blank roots are
            refused — never default to the filesystem root.

    Returns:
        The resolved absolute path on success.

    Raises:
        SecurityError: If the resolved path escapes the allowed root, if the
            two are on different drives/mounts, or if ``allowed_root`` is
            empty or blank.
    """
    if not str(allowed_root).strip():
        raise SecurityError("allowed_root must be a non-empty directory path")
    resolved = Path(path).expanduser().resolve()
    root = Path(allowed_root).expanduser().resolve()
    root_cased = os.path.normcase(str(root))
    try:
        common = os.path.commonpath([root_cased, os.path.normcase(str(resolved))])
    except ValueError as exc:
        # Different drives or UNC-vs-local mix: no common path at all.
        raise SecurityError("path resolves outside the allowed root") from exc
    if common != root_cased:
        raise SecurityError("path resolves outside the allowed root")
    return resolved


def credential_from_env(env_var: str, *, adapter_name: str) -> str:
    """Read a credential from the environment (CLAUDE.md security req #3).

    Args:
        env_var: Name of the environment variable to read.
        adapter_name: Adapter requesting the credential (for the error message).

    Returns:
        The credential value, exactly as set in the environment.

    Raises:
        ConfigError: If the variable is unset or blank (absence is a config
            problem, NOT a security violation). The message names the variable
            and adapter but never echoes any value.
    """
    value = os.environ.get(env_var, "")
    if not value.strip():
        raise ConfigError(
            f"adapter '{adapter_name}' requires the environment variable "
            f"'{env_var}' to be set to a non-empty value"
        )
    return value


def reject_inline_credential(value: object, *, adapter_name: str) -> None:
    """Raise SecurityError if ``value`` is a non-empty string.

    Adapters call this on any ``api_key=``-style constructor argument so
    inline credentials fail loudly (ADR-010 #4). The error message never
    includes the credential value.

    Args:
        value: The constructor argument to check.
        adapter_name: Adapter name (for the error message).

    Raises:
        SecurityError: If a non-empty string credential was passed inline.
    """
    if isinstance(value, str) and value:
        raise SecurityError(
            f"adapter '{adapter_name}' does not accept inline credentials; "
            "provide them via an environment variable instead"
        )
