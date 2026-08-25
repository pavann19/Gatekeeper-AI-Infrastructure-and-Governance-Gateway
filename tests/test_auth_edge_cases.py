"""
Deepened edge-case coverage for core/auth.py -- the module's own docstring
calls this "the one real trust boundary" of the gateway. These tests are
deliberately new ground relative to tests/test_auth.py: hash determinism and
collision behavior, duplicate-registration semantics in KeyStore, case and
whitespace sensitivity of raw keys, garbage/unicode input handling, an
exhaustive sweep of malformed Authorization header shapes, and a pinned
security property that no object ever leaks a raw key via repr/str.
"""
import json

import pytest

from core import auth as auth_mod
from core.auth import (
    KeyStore,
    Principal,
    generate_key,
    hash_key,
    resolve_principal,
)


@pytest.fixture
def key_store(tmp_path, monkeypatch):
    """Installs an isolated key store and returns a helper that issues keys."""
    path = tmp_path / "api_keys.json"
    store = {}

    def issue(capability="ELEVATED", tenant="acme", key_id="test-key"):
        plaintext = generate_key()
        store[hash_key(plaintext)] = {
            "capability": capability, "tenant": tenant, "key_id": key_id,
        }
        path.write_text(json.dumps(store), encoding="utf-8")
        monkeypatch.setattr(auth_mod, "_store", KeyStore(str(path)))
        return plaintext

    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(auth_mod, "_store", KeyStore(str(path)))
    return issue


# ===========================================================================
# hash_key: determinism and collision resistance
# ===========================================================================

def test_hash_key_is_deterministic_across_repeated_calls():
    key = generate_key()
    digests = {hash_key(key) for _ in range(50)}
    assert len(digests) == 1


def test_hash_key_is_deterministic_across_separate_processes_worth_of_state():
    """Same plaintext bytes always produce the same digest, independent of
    any in-memory KeyStore state -- hash_key is a pure function."""
    key = "gk_fixed-example-key-for-determinism-check"
    assert hash_key(key) == hash_key(key)
    import hashlib
    assert hash_key(key) == hashlib.sha256(key.encode("utf-8")).hexdigest()


def test_hash_key_has_no_collisions_across_a_large_sample():
    keys = [generate_key() for _ in range(2000)]
    digests = {hash_key(k) for k in keys}
    assert len(digests) == len(keys)


def test_hash_key_of_similar_keys_differs_completely():
    """Adjacent/near-identical plaintexts must not produce similar digests
    (avalanche property) -- a weak hash could leak structure."""
    base = "gk_" + "a" * 40
    variant = "gk_" + "a" * 39 + "b"
    d1, d2 = hash_key(base), hash_key(variant)
    assert d1 != d2
    # crude avalanche check: most hex characters should differ
    differing = sum(1 for a, b in zip(d1, d2) if a != b)
    assert differing > len(d1) // 2


def test_hash_key_output_is_fixed_length_hex():
    for key in ("", "a", "gk_" + "x" * 500, "unicode-key-é中文"):
        digest = hash_key(key)
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)


# ===========================================================================
# KeyStore: duplicate registration
# ===========================================================================

def test_duplicate_key_hash_last_entry_in_json_wins(tmp_path):
    """If the same key hash appears twice in the JSON object (e.g. a
    provisioning script wrote it twice with different grants), Python's json
    module collapses duplicate object keys to the last one -- KeyStore must
    not silently merge or union the grants."""
    path = tmp_path / "api_keys.json"
    key = generate_key()
    digest = hash_key(key)
    # Hand-construct JSON text with a duplicate key manually since dict
    # literals can't hold two identical keys.
    raw = (
        '{"%s": {"capability": "GENERAL", "tenant": "first", "key_id": "first"}, '
        '"%s": {"capability": "INTERNAL", "tenant": "second", "key_id": "second"}}'
        % (digest, digest)
    )
    path.write_text(raw, encoding="utf-8")

    store = KeyStore(str(path))
    assert len(store) == 1
    grant = store.lookup(key)
    assert grant["key_id"] == "second"
    assert grant["capability"] == "INTERNAL"


def test_two_distinct_plaintext_keys_registered_to_same_key_id_both_work(tmp_path):
    """Registering two different keys under the same human-readable key_id is
    legal (e.g. a rotation window) -- both must independently authenticate."""
    path = tmp_path / "api_keys.json"
    key_a, key_b = generate_key(), generate_key()
    path.write_text(json.dumps({
        hash_key(key_a): {"capability": "ELEVATED", "tenant": "t", "key_id": "shared-id"},
        hash_key(key_b): {"capability": "ELEVATED", "tenant": "t", "key_id": "shared-id"},
    }), encoding="utf-8")

    store = KeyStore(str(path))
    assert len(store) == 2
    assert store.lookup(key_a)["key_id"] == "shared-id"
    assert store.lookup(key_b)["key_id"] == "shared-id"


def test_reloading_store_with_removed_key_revokes_it(tmp_path, monkeypatch):
    """Re-provisioning the file with a key hash removed must not leave the
    old grant reachable through a stale in-memory copy after force reload."""
    path = tmp_path / "api_keys.json"
    key = generate_key()
    path.write_text(json.dumps({
        hash_key(key): {"capability": "INTERNAL", "tenant": "t", "key_id": "k"}
    }), encoding="utf-8")
    store = KeyStore(str(path))
    assert store.lookup(key) is not None

    path.write_text("{}", encoding="utf-8")
    store.load(force=True)
    assert store.lookup(key) is None


# ===========================================================================
# Case sensitivity and whitespace of raw API keys
# ===========================================================================

def test_api_key_lookup_is_case_sensitive(key_store):
    key = key_store(capability="INTERNAL")
    assert resolve_principal(api_key=key).authenticated is True
    assert resolve_principal(api_key=key.upper()).authenticated is False
    assert resolve_principal(api_key=key.swapcase()).authenticated is False


def test_key_store_hash_lookup_lowercases_stored_hash_only_not_the_key(tmp_path):
    """KeyStore.load() lowercases the *hash* key from the JSON file (so a
    hex hash written in uppercase still matches), but this must not be
    confused with the plaintext key itself being case-insensitive."""
    path = tmp_path / "api_keys.json"
    key = generate_key()
    uppercase_hash = hash_key(key).upper()
    path.write_text(json.dumps({
        uppercase_hash: {"capability": "ELEVATED", "tenant": "t", "key_id": "k"}
    }), encoding="utf-8")

    store = KeyStore(str(path))
    # The stored hash was uppercase hex; lookup must still succeed because
    # candidate hashes are always lowercase hex from hashlib.
    assert store.lookup(key) is not None
    assert store.lookup(key)["capability"] == "ELEVATED"


def test_api_key_passed_directly_is_not_stripped_of_whitespace(key_store):
    """resolve_principal(api_key=...) is a direct programmatic entry point
    (not parsed from a header), so it performs no trimming. A caller who
    passes a whitespace-padded key gets rejected, not silently normalized."""
    key = key_store(capability="INTERNAL")
    assert resolve_principal(api_key=f" {key}").authenticated is False
    assert resolve_principal(api_key=f"{key} ").authenticated is False
    assert resolve_principal(api_key=f"\t{key}\n").authenticated is False


def test_bearer_header_token_whitespace_is_stripped_by_extraction(key_store):
    """Unlike the direct api_key path, _extract_bearer explicitly calls
    .strip() on the token portion, so accidental extra whitespace around the
    token inside a Bearer header IS tolerated. This is intentional, real
    behavior -- pinned here so a future refactor can't silently change it
    either direction without a failing test."""
    key = key_store(capability="ELEVATED")
    assert resolve_principal(authorization=f"Bearer {key} ").authenticated is True
    assert resolve_principal(authorization=f"Bearer  {key}").authenticated is True


# ===========================================================================
# Garbage, oversized, and unicode tokens
# ===========================================================================

def test_extremely_long_garbage_bearer_token_is_rejected_not_crashed():
    huge = "x" * 5_000_000
    p = resolve_principal(authorization=f"Bearer {huge}")
    assert p.authenticated is False
    assert p.capability == "GENERAL"


def test_extremely_long_garbage_api_key_is_rejected_not_crashed():
    huge = "y" * 5_000_000
    p = resolve_principal(api_key=huge)
    assert p.authenticated is False
    assert p.capability == "GENERAL"


@pytest.mark.parametrize("token", [
    "gk_éèêë",           # accented latin
    "gk_中文密钥",           # CJK
    "gk_\U0001F511\U0001F510",               # emoji (key emoji, lock emoji)
    "gk_" + chr(0) + "embedded-null",   # embedded null character
    "gk_" + chr(0x202E) + chr(0x202D),  # bidi control chars
])
def test_non_ascii_unicode_tokens_are_rejected_not_crashed(token):
    p = resolve_principal(api_key=token)
    assert p.authenticated is False
    assert p.capability == "GENERAL"


def test_unicode_key_can_still_authenticate_if_registered(tmp_path):
    """hash_key encodes as UTF-8, so a unicode plaintext key is not
    inherently unsupported -- if it was genuinely provisioned, it must work
    exactly like an ASCII key."""
    path = tmp_path / "api_keys.json"
    unicode_key = "gk_中文密钥-\U0001F511"
    path.write_text(json.dumps({
        hash_key(unicode_key): {"capability": "ELEVATED", "tenant": "t", "key_id": "unicode-key"}
    }), encoding="utf-8")
    store = KeyStore(str(path))
    assert store.lookup(unicode_key) is not None
    assert store.lookup(unicode_key)["key_id"] == "unicode-key"


# ===========================================================================
# resolve_principal: exhaustive malformed Authorization header shapes
# ===========================================================================

@pytest.mark.parametrize("header,description", [
    ("Bearer", "scheme with nothing after it at all"),
    ("Bearer ", "scheme with a single trailing space and empty value"),
    ("Bearer   ", "scheme with multiple trailing spaces and empty value"),
    ("bearer", "lowercase scheme alone, no value"),
    ("BEARER", "uppercase scheme alone, no value"),
    ("BeArEr sometoken", "mixed-case scheme with a value"),
    ("Bearer" + " " + "", "scheme plus space plus empty string concatenation"),
    ("BearerNoSpace", "scheme glued to value with no separator"),
    ("Bearersometoken", "scheme glued directly to a token-like value"),
    (" Bearer sometoken", "leading whitespace before the whole header"),
    ("Bearer sometoken ", "trailing whitespace after the whole header"),
    ("Bearer\tsometoken", "tab as the separator instead of space"),
    ("Bearer\nsometoken", "newline as the separator instead of space"),
    ("Bearer  multi  word  token", "multiple internal words after scheme"),
    ("Token sometoken", "an entirely different, unsupported scheme"),
    ("Basic dXNlcjpwYXNz", "Basic auth scheme, not Bearer"),
    ("", "empty string header"),
    ("   ", "whitespace-only header"),
    (None, "header entirely absent"),
])
def test_every_malformed_authorization_header_shape_falls_back_to_anonymous(header, description):
    p = resolve_principal(authorization=header)
    assert p.authenticated is False, f"Should not authenticate for: {description!r} ({header!r})"
    assert p.capability == "GENERAL", f"Should default to GENERAL for: {description!r} ({header!r})"
    assert p.tenant == "default"
    assert p.key_id == "anonymous"


def test_bearer_glued_to_a_real_valid_key_does_not_authenticate(key_store):
    """'Bearer<key>' with no separating space must not accidentally parse as
    a valid Bearer token -- the scheme must be its own whitespace-delimited
    word."""
    key = key_store(capability="INTERNAL")
    p = resolve_principal(authorization=f"Bearer{key}")
    assert p.authenticated is False
    assert p.capability == "GENERAL"


def test_multiple_spaces_between_scheme_and_valid_key_still_authenticates(key_store):
    """split(None, 1) collapses runs of whitespace before the token, so a
    real valid key preceded by several spaces must still resolve -- pinning
    the exact split() semantics being relied on."""
    key = key_store(capability="ELEVATED")
    p = resolve_principal(authorization=f"Bearer     {key}")
    assert p.authenticated is True
    assert p.capability == "ELEVATED"


# ===========================================================================
# Security property: raw keys never leak through repr/str
# ===========================================================================

def test_principal_repr_and_str_never_contain_raw_key_material(key_store):
    """Principal never stores the raw key at all (only capability/tenant/
    key_id), so this is somewhat structural -- but it's exactly the kind of
    invariant that a careless future field addition (e.g. caching the raw
    token on the Principal for convenience) could quietly violate. Pinned
    here as a regression guard."""
    key = key_store(capability="INTERNAL", key_id="repr-test-key")
    principal = resolve_principal(api_key=key)

    assert key not in repr(principal)
    assert key not in str(principal)
    assert hash_key(key) not in repr(principal)
    assert hash_key(key) not in str(principal)


def test_keystore_repr_and_str_never_contain_raw_or_hashed_keys(key_store, tmp_path):
    key = key_store(capability="ELEVATED", key_id="another-repr-test")
    store = auth_mod._store
    store.load()

    assert key not in repr(store)
    assert key not in str(store)
    assert hash_key(key) not in repr(store)
    assert hash_key(key) not in str(store)


def test_dataclass_default_repr_of_principal_is_safe_even_with_secrets_dict_field():
    """Principal is a plain frozen dataclass with no raw-key field, so its
    auto-generated __repr__ only ever surfaces capability/tenant/key_id/
    authenticated/reason -- never anything secret. Assert the field set
    explicitly so an added field is caught by this test."""
    p = Principal(capability="INTERNAL", tenant="acme", key_id="k1", authenticated=True, reason="verified api key")
    r = repr(p)
    assert "capability='INTERNAL'" in r or "capability=\"INTERNAL\"" in r
    assert "tenant='acme'" in r or "tenant=\"acme\"" in r
    # No field on Principal is named anything like a raw key/token/secret.
    field_names = {f for f in p.__dataclass_fields__}
    assert field_names == {"capability", "tenant", "key_id", "authenticated", "reason"}
