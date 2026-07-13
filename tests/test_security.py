"""Tests for tulving.security — written BEFORE implementation.

100% line and branch coverage is mandatory (CLAUDE.md security req #7).
Includes positive AND negative redaction corpora (D10) and Windows path cases.
"""

import os
import sys
from pathlib import Path

import pytest

from tulving.exceptions import ConfigError, SecurityError
from tulving.security import (
    DEFAULT_SENSITIVE_KEY_PATTERNS,
    REDACTED,
    _content_looks_secret,
    _has_opaque_token,
    compile_explicit_patterns,
    compile_key_patterns,
    contain_path,
    credential_from_env,
    is_sensitive_key,
    redact_secrets,
    redact_text,
    reject_inline_credential,
    should_mask_content,
    validate_leaf_name,
)


class TestPathContainmentFailures:
    """Traversal corpus — every one raises SecurityError from contain_path."""

    def test_relative_dotdot_traversal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "root"
        root.mkdir()
        monkeypatch.chdir(root)
        with pytest.raises(SecurityError):
            contain_path("../../etc/passwd", root)

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="backslash is a path separator only on Windows; on POSIX it is a literal "
        "filename char, so this cannot traverse (covered by the forward-slash tests)",
    )
    def test_backslash_dotdot_traversal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "root"
        root.mkdir()
        monkeypatch.chdir(root)
        with pytest.raises(SecurityError):
            contain_path("..\\..\\Windows\\System32", root)

    def test_absolute_path_outside_root(self) -> None:
        with pytest.raises(SecurityError):
            contain_path("C:\\other\\place", "C:\\allowed")

    def test_cross_drive_escape(self) -> None:
        """Different drives make commonpath raise ValueError -> SecurityError."""
        with pytest.raises(SecurityError):
            contain_path("D:\\anything", "C:\\allowed")

    def test_unc_path_against_local_root(self) -> None:
        """UNC share against a local drive root is an escape.

        Uses \\\\localhost\\<nonexistent share> instead of a fake server name:
        a nonexistent hostname costs a >1s network probe in resolve().
        """
        with pytest.raises(SecurityError):
            contain_path("\\\\localhost\\tulving-no-such-share\\x", "C:\\allowed")

    def test_prefix_collision_escape(self, tmp_path: Path) -> None:
        """Naive startswith would pass C:\\allowed_evil against C:\\allowed."""
        root = tmp_path / "allowed"
        root.mkdir()
        evil = tmp_path / "allowed_evil"
        evil.mkdir()
        with pytest.raises(SecurityError):
            contain_path(evil / "f", root)

    def test_prefix_collision_escape_drive_literals(self) -> None:
        with pytest.raises(SecurityError):
            contain_path("C:\\allowed_evil\\f", "C:\\allowed")

    def test_blank_allowed_root_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(SecurityError):
            contain_path(tmp_path / "f", "")

    def test_whitespace_allowed_root_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(SecurityError):
            contain_path(tmp_path / "f", "   ")

    @pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows-only")
    def test_junction_inside_root_pointing_outside(self, tmp_path: Path) -> None:
        """A junction under the root escaping it must be caught via resolve()."""
        import _winapi

        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "loot.txt").write_text("x")
        junction = root / "link"
        _winapi.CreateJunction(str(outside), str(junction))
        with pytest.raises(SecurityError):
            contain_path(junction / "loot.txt", root)

    def test_symlink_inside_root_pointing_outside(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        link = root / "sym"
        try:
            os.symlink(outside, link, target_is_directory=True)
        except OSError:
            pytest.skip("symlink privilege unavailable")
        with pytest.raises(SecurityError):
            contain_path(link / "f.txt", root)


class TestLeafNameFailures:
    """Whitelist violations and Windows reserved device names -> SecurityError."""

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "a/b",
            "a\\b",
            "..",
            ".",
            "name.json",
            "file:stream",
            "CON",
            "nul",
            "COM7",
            "lpt1",
            "a" * 129,
            "na me",
            "naïve",
            "emoji\U0001f4a5",
        ],
    )
    def test_rejected_leaf_names(self, name: str) -> None:
        with pytest.raises(SecurityError):
            validate_leaf_name(name)

    def test_rejection_message_never_echoes_name(self) -> None:
        with pytest.raises(SecurityError) as excinfo:
            validate_leaf_name("evil/../name")
        assert "evil" not in str(excinfo.value)


class TestCredentialFailures:
    def test_unset_env_var_raises_config_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TULVING_TEST_UNSET", raising=False)
        with pytest.raises(ConfigError) as excinfo:
            credential_from_env("TULVING_TEST_UNSET", adapter_name="x")
        assert "TULVING_TEST_UNSET" in str(excinfo.value)

    def test_empty_env_var_raises_config_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TULVING_TEST_EMPTY", "")
        with pytest.raises(ConfigError):
            credential_from_env("TULVING_TEST_EMPTY", adapter_name="x")

    def test_whitespace_env_var_raises_config_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TULVING_TEST_BLANK", "   ")
        with pytest.raises(ConfigError):
            credential_from_env("TULVING_TEST_BLANK", adapter_name="x")

    def test_error_names_var_and_adapter_but_no_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TULVING_TEST_BLANK2", "  hunter2  ")
        monkeypatch.delenv("TULVING_TEST_GONE", raising=False)
        with pytest.raises(ConfigError) as excinfo:
            credential_from_env("TULVING_TEST_GONE", adapter_name="openai-embedder")
        message = str(excinfo.value)
        assert "TULVING_TEST_GONE" in message
        assert "openai-embedder" in message
        assert "hunter2" not in message

    def test_inline_credential_rejected(self) -> None:
        with pytest.raises(SecurityError) as excinfo:
            reject_inline_credential("sk-abc123def456ghi789jkl", adapter_name="openai")
        message = str(excinfo.value)
        assert "sk-abc123def456ghi789jkl" not in message
        assert "openai" in message

    def test_inline_credential_none_and_empty_pass(self) -> None:
        reject_inline_credential(None, adapter_name="openai")
        reject_inline_credential("", adapter_name="openai")

    def test_inline_credential_non_string_passes(self) -> None:
        reject_inline_credential(12345, adapter_name="openai")


class TestKeyRedactionPositive:
    """Key-name positives (D10 word-boundary matching)."""

    @pytest.mark.parametrize(
        "key",
        [
            "password",
            "PASSWORD",
            "api_key",
            "API-KEY",
            "db.password",
            "auth_token",
            "key",
            "my_key",
            "private-notes",
            "github_credential",
        ],
    )
    def test_sensitive_keys_match(self, key: str) -> None:
        assert is_sensitive_key(key) is True


class TestKeyRedactionNegative:
    """Key-name negatives (D10, non-negotiable corpus)."""

    @pytest.mark.parametrize(
        "key",
        [
            "monkey",
            "keyboard",
            "tokenize",
            "authentic",
            "keystone",
            "passwords",
            "secretary",
            "donkey",
            "hotkey",
            "turkey",
        ],
    )
    def test_benign_keys_do_not_match(self, key: str) -> None:
        assert is_sensitive_key(key) is False


class TestContentRedactionPositive:
    """Token-shape positives: substring replaced, surrounding prose intact."""

    def test_aws_access_key_mid_sentence(self) -> None:
        text = "the key AKIAIOSFODNN7EXAMPLE was found in logs"
        result = redact_secrets(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in result
        assert REDACTED in result
        assert result.startswith("the key ")
        assert result.endswith(" was found in logs")

    def test_github_token(self) -> None:
        token = "ghp_" + "A1b2C3d4" * 5  # 40 chars after prefix
        result = redact_secrets(f"push failed with {token} embedded")
        assert token not in result
        assert REDACTED in result

    def test_sk_style_api_key(self) -> None:
        secret = "sk-" + "x" * 24
        result = redact_secrets(f"config had {secret} in it")
        assert secret not in result
        assert REDACTED in result

    def test_slack_token(self) -> None:
        result = redact_secrets("slack said xoxb-1234567890-abcdefGHIJ was revoked")
        assert "xoxb-1234567890-abcdefGHIJ" not in result
        assert REDACTED in result

    def test_jwt_three_part(self) -> None:
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpM"
        result = redact_secrets(f"Authorization used {jwt} today")
        assert jwt not in result
        assert REDACTED in result

    def test_pem_private_key_block(self) -> None:
        pem = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA7bq7\nx8FkV2m1\n"
            "-----END RSA PRIVATE KEY-----"
        )
        result = redact_secrets(f"found this:\n{pem}\nin the repo")
        assert "MIIEowIBAAKCAQEA7bq7" not in result
        assert REDACTED in result
        assert result.endswith("in the repo")

    def test_truncated_pem_block_still_caught(self) -> None:
        truncated = "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQ"
        result = redact_secrets(f"partial dump: {truncated}")
        assert "MIIEvQIBADANBgkqhkiG9w0BAQ" not in result
        assert REDACTED in result

    def test_bearer_header(self) -> None:
        result = redact_secrets("Authorization: Bearer abcDEF1234567890xyz")
        assert "abcDEF1234567890xyz" not in result
        assert REDACTED in result

    def test_kv_assignment_label_preserved(self) -> None:
        result = redact_secrets("password = hunter2secret")
        assert "hunter2secret" not in result
        assert "password" in result
        assert REDACTED in result

    def test_kv_assignment_quoted_value(self) -> None:
        result = redact_secrets('api_key: "sk-live-XYZ"')
        assert "sk-live-XYZ" not in result
        assert "api_key" in result
        assert REDACTED in result


class TestContentRedactionNegative:
    """Content-shape negatives: returned unchanged (D10, no false positives)."""

    @pytest.mark.parametrize(
        "text",
        [
            "my password is safe elsewhere",
            "the token bucket algorithm limits throughput",
            "commit e758ad8f3b2c1d4e5f6a7b8c9d0e1f2a3b4c5d6e was tagged",
            "short value sk-abc is below minimum length",
            "set the Bearer header later",
            "the monkey sat on the keyboard",
        ],
    )
    def test_benign_content_unchanged(self, text: str) -> None:
        assert redact_secrets(text) == text

    @pytest.mark.parametrize(
        ("shape", "text"),
        [
            ("aws_access_key", "lowercase akiaiosfodnn7example is not an AWS key"),
            ("aws_access_key", "AKIA123 is too short to be an AWS key"),
            ("github_token", "ghp_abc123 is too short to be a GitHub token"),
            ("slack_token", "xoxb-12345 is too short to be a Slack token"),
            ("jwt", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0 has only two parts"),
            (
                "private_key_pem",
                "-----BEGIN PUBLIC KEY-----\nMIIBIjANBg\n-----END PUBLIC KEY-----",
            ),
            ("kv_assignment", "token=abc"),  # value below 4-char minimum
        ],
    )
    def test_per_shape_negatives(self, shape: str, text: str) -> None:
        """Each token shape individually rejects its near-miss (QA addition)."""
        assert redact_secrets(text) == text

    def test_no_match_returns_same_object(self) -> None:
        text = "nothing sensitive here at all"
        assert redact_secrets(text) is text

    def test_redact_text_benign_content_unchanged(self) -> None:
        text = "the token bucket algorithm limits throughput"
        assert redact_text(text) == text


class TestRedactionBoundary:
    """Idempotency, purity, empty input."""

    def test_redact_text_idempotent(self) -> None:
        text = "password = hunter2secret and AKIAIOSFODNN7EXAMPLE"
        once = redact_text(text)
        assert redact_text(once) == once

    def test_already_redacted_input_unchanged(self) -> None:
        text = f"password = {REDACTED}"
        assert redact_text(text) == text

    def test_empty_string(self) -> None:
        assert redact_secrets("") == ""
        assert redact_text("") == ""

    def test_kv_assignment_minimum_value_length_boundary(self) -> None:
        """Exactly 4 chars is the shortest value the kv shape masks (QA addition)."""
        redacted = redact_secrets("secret=abcd")
        assert "abcd" not in redacted
        assert redacted == f"secret={REDACTED}"
        assert redact_secrets("secret=abc") == "secret=abc"

    def test_redact_text_masks_key_labelled_values(self) -> None:
        result = redact_text("my_key = abcd1234efgh")
        assert "abcd1234efgh" not in result
        assert "my_key" in result
        assert REDACTED in result

    def test_redact_text_with_custom_key_patterns(self) -> None:
        patterns = compile_key_patterns(["x-custom"])
        result = redact_text("x-custom: supersecretvalue", key_patterns=patterns)
        assert "supersecretvalue" not in result
        assert REDACTED in result


class TestCompileKeyPatterns:
    def test_extras_are_literal_escaped(self) -> None:
        patterns = compile_key_patterns(["a.b"])
        assert is_sensitive_key("aXb", patterns) is False
        assert is_sensitive_key("a.b", patterns) is True

    def test_extras_augment_defaults(self) -> None:
        patterns = compile_key_patterns(["x-custom"])
        assert is_sensitive_key("x-custom", patterns) is True
        assert is_sensitive_key("password", patterns) is True

    def test_default_pattern_inventory_is_stable(self) -> None:
        assert "password" in DEFAULT_SENSITIVE_KEY_PATTERNS
        assert "key" in DEFAULT_SENSITIVE_KEY_PATTERNS


class TestPathContainmentHappy:
    def test_root_itself_is_contained(self, tmp_path: Path) -> None:
        result = contain_path(tmp_path, tmp_path)
        assert result == tmp_path.resolve()

    def test_direct_child(self, tmp_path: Path) -> None:
        result = contain_path(tmp_path / "child.txt", tmp_path)
        assert result == (tmp_path / "child.txt").resolve()

    def test_deep_descendant(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c" / "d.json"
        assert contain_path(deep, tmp_path) == deep.resolve()

    def test_relative_path_resolving_inside(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = contain_path("sub/file.txt", tmp_path)
        assert result == (tmp_path / "sub" / "file.txt").resolve()

    def test_tilde_under_home_rooted_allowed_root(self) -> None:
        result = contain_path("~/tulving-test-file", "~")
        assert result == (Path.home() / "tulving-test-file").resolve()

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="mixed \\ and / separators normalize to path parts only on Windows; "
        "on POSIX backslash is a literal filename char",
    )
    def test_mixed_separators(self, tmp_path: Path) -> None:
        mixed = f"{tmp_path}/allowed\\sub/file"
        result = contain_path(mixed, tmp_path)
        assert result == (tmp_path / "allowed" / "sub" / "file").resolve()

    @pytest.mark.skipif(sys.platform != "win32", reason="case-insensitive paths are Windows-only")
    def test_windows_case_insensitive_containment(self, tmp_path: Path) -> None:
        """normcase makes containment case-insensitive on Windows (QA addition)."""
        child = str(tmp_path / "Sub" / "File.txt").upper()
        result = contain_path(child, str(tmp_path).lower())
        assert result == Path(child).resolve()

    def test_not_yet_existing_target(self, tmp_path: Path) -> None:
        target = tmp_path / "exports" / "not_written_yet.json"
        assert not target.exists()
        result = contain_path(target, tmp_path)
        assert result.is_absolute()

    def test_returned_value_is_absolute_and_resolved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = contain_path("x/../y.txt", tmp_path)
        assert result.is_absolute()
        assert ".." not in result.parts
        assert result == (tmp_path / "y.txt").resolve()


class TestLeafNameHappy:
    @pytest.mark.parametrize(
        "name",
        ["a", "snake_case_key", "kebab-name-2", "UPPER123", "a" * 128],
    )
    def test_accepted_and_returned_identical(self, name: str) -> None:
        assert validate_leaf_name(name) is name


class TestCredentialHappy:
    def test_credential_read_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TULVING_TEST_SET", "value-from-env")
        assert credential_from_env("TULVING_TEST_SET", adapter_name="x") == "value-from-env"


# ---------------------------------------------------------------------------
# should_mask_content / compile_explicit_patterns / opaque-token softening
# (v0.2 sensitive-key redaction softening, D-v02-7)
# ---------------------------------------------------------------------------


class TestShouldMaskContent:
    def test_non_sensitive_key_never_masks(self) -> None:
        assert should_mask_content("note", "AKIA1234567890ABCDEF") is False

    def test_default_key_with_named_shape_masks(self) -> None:
        assert should_mask_content("api_key", "here sk-" + "x" * 24) is True

    def test_default_key_with_prose_does_not_mask(self) -> None:
        """The reported bug: fact:auth-ttl prose no longer whole-masked."""
        assert should_mask_content("auth", "auth token TTL is 15 min") is False

    def test_default_key_with_opaque_token_masks(self) -> None:
        assert should_mask_content("secret", "Ab3" + "kZ9qWvT1" * 4) is True

    def test_user_declared_key_masks_unconditionally(self) -> None:
        patterns = compile_explicit_patterns(["projectcode"])
        assert (
            should_mask_content("projectcode", "just some prose here", explicit_patterns=patterns)
            is True
        )

    def test_user_declared_key_overrides_default_overlap(self) -> None:
        """D-v02-7 Q3 (mandatory): explicit declaration wins even when the key
        ALSO matches a built-in default ("token"/"auth"); prose still masks."""
        patterns = compile_explicit_patterns(["auth-prod"])
        assert (
            should_mask_content(
                "auth-prod-token",
                "rotate quarterly, no value here",
                explicit_patterns=patterns,
            )
            is True
        )

    def test_default_key_empty_content_not_masked(self) -> None:
        assert should_mask_content("secret", "") is False

    def test_explicit_patterns_none_falls_through_to_default(self) -> None:
        assert should_mask_content("api_key", "prose", explicit_patterns=None) is False

    # -- SEV-001 (mandatory): a real secret broken by ONE embedded whitespace
    # char must still whole-mask under a sensitive key (repro: store("deploy
    # secret is aB3...\nN8p...", key="secret:deploy") -> curate() leaked it).

    def test_newline_broken_secret_masks(self) -> None:
        content = "deploy secret is aB3dE6fG9hJ2kL5m\nN8pQ1rS4tU7vW0xY3zA6"
        assert should_mask_content("secret:deploy", content) is True

    def test_tab_broken_secret_masks(self) -> None:
        content = "deploy secret is aB3dE6fG9hJ2kL5m\tN8pQ1rS4tU7vW0xY3zA6"
        assert should_mask_content("secret:deploy", content) is True

    def test_double_space_broken_secret_masks(self) -> None:
        content = "deploy secret is aB3dE6fG9hJ2kL5m  N8pQ1rS4tU7vW0xY3zA6"
        assert should_mask_content("secret:deploy", content) is True

    def test_longer_prose_still_does_not_mask(self) -> None:
        """False-positive guard: the adjacent-pair join must not turn ordinary
        multi-word prose into a fake opaque token."""
        assert should_mask_content("auth", "auth token TTL is 15 minutes and rotates") is False


class TestContentLooksSecret:
    def test_named_shape_true(self) -> None:
        assert _content_looks_secret("here is AKIAIOSFODNN7EXAMPLE") is True

    def test_kv_assignment_true(self) -> None:
        assert _content_looks_secret("password = hunter2secret") is True

    def test_opaque_token_true(self) -> None:
        assert _content_looks_secret("aB3dE6fG9hJ2kL5mN8pQ1rS4") is True

    def test_prose_false(self) -> None:
        assert _content_looks_secret("auth token TTL is 15 min") is False

    def test_git_sha_false(self) -> None:
        """40-char lowercase-hex SHA has only 2 classes (D10 honored)."""
        assert _content_looks_secret("e758ad8f3b2c1d4e5f6a7b8c9d0e1f2a3b4c5d6e") is False


class TestHasOpaqueToken:
    def test_empty_false(self) -> None:
        assert _has_opaque_token("") is False

    def test_short_token_false(self) -> None:
        assert _has_opaque_token("Ab3xyz") is False

    def test_long_lowercase_only_false(self) -> None:
        assert _has_opaque_token("a" * 30) is False

    def test_long_lower_digit_false(self) -> None:
        """2-class (hex-like) token: named accepted-risk gap (D-v02-7 Q2)."""
        assert _has_opaque_token("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3") is False

    def test_long_three_class_true(self) -> None:
        assert _has_opaque_token("aB3dE6fG9hJ2kL5mN8pQ1rS4tU7vW0") is True

    def test_upper_digit_only_false(self) -> None:
        assert _has_opaque_token("A1B2C3D4E5F6A1B2C3D4E5F6A1B2C3") is False

    def test_boundary_exactly_min_len(self) -> None:
        token24 = "aB3dE6fG9hJ2kL5mN8pQ1rS4"
        assert len(token24) == 24
        assert _has_opaque_token(token24) is True
        assert _has_opaque_token(token24[:-1]) is False

    # -- SEV-001 (mandatory): adjacent-pair join catches a secret broken by
    # exactly one embedded whitespace char (space/tab/newline/double-space
    # all collapse identically under content.split()).

    def test_single_space_broken_secret_true(self) -> None:
        assert _has_opaque_token("aB3dE6fG9hJ2kL5m N8pQ1rS4tU7vW0xY3zA6") is True

    def test_double_space_broken_secret_true(self) -> None:
        assert _has_opaque_token("aB3dE6fG9hJ2kL5m  N8pQ1rS4tU7vW0xY3zA6") is True

    def test_tab_broken_secret_true(self) -> None:
        assert _has_opaque_token("aB3dE6fG9hJ2kL5m\tN8pQ1rS4tU7vW0xY3zA6") is True

    def test_newline_broken_secret_true(self) -> None:
        assert _has_opaque_token("aB3dE6fG9hJ2kL5m\nN8pQ1rS4tU7vW0xY3zA6") is True

    def test_each_half_too_short_alone(self) -> None:
        """Confirms the pair-join branch is what fires, not the per-token one:
        neither half alone reaches _OPAQUE_MIN_LEN."""
        assert _has_opaque_token("aB3dE6fG9hJ2kL5m") is False
        assert _has_opaque_token("N8pQ1rS4tU7vW0xY3zA6") is False

    def test_adjacent_prose_pair_does_not_trigger(self) -> None:
        """False-positive guard: real prose's longest adjacent-pair join stays
        far below _OPAQUE_MIN_LEN, so ordinary sentences never qualify."""
        assert _has_opaque_token("auth token TTL is 15 minutes and rotates") is False

    def test_multi_break_split_not_caught(self) -> None:
        """Documents the residual scope: only ADJACENT-PAIR joins are checked
        (not triples+) — a secret split across >=2 embedded breaks (3+
        fragments, each fragment and each adjacent pair below the threshold)
        is a named accepted risk, not covered by this fix."""
        assert _has_opaque_token("aB3dE6fG kL5mN8pQ tU7vW0xY") is False


class TestCompileExplicitPatterns:
    def test_none_returns_none(self) -> None:
        assert compile_explicit_patterns(None) is None

    def test_empty_returns_none(self) -> None:
        assert compile_explicit_patterns([]) is None

    def test_compiles_literal_boundary(self) -> None:
        patterns = compile_explicit_patterns(["a.b"])
        assert patterns is not None
        assert is_sensitive_key("a.b", patterns) is True
        assert is_sensitive_key("aXb", patterns) is False
        assert is_sensitive_key("password", patterns) is False
