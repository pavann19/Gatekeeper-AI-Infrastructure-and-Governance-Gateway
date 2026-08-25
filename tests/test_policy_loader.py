"""
Tests for core/policy_loader.py -- the module-level JSON policy file loader
(domain anchors + symbolic rules), loaded once at import time.

Scope: this module's own responsibility is parsing/loading/validation-at-load-time
of the two policy JSON files (policies/domain_anchors.json and
policies/symbolic_rules.json) into module-level state, with fail-closed
semantics (None) on missing/invalid symbolic rules and fail-open (empty list)
for domain anchors. This is distinct from core/policy.py's PolicyStore
(tenant-scoped capability/risk policy) and core/policy_versioning.py's
version history -- neither of those is re-tested here.

Because the real module loads its files once at import time via module-level
globals (_domain_anchors, _jailbreak_patterns, etc.), these tests exercise the
internal, reusable functions directly (_load_json_file, _init_policies)
against tmp_path files, patching the module's file-path constants and
resetting/restoring global state around each test so tests don't leak into
each other or into the process-wide singleton state used by other modules.
"""
import json

import pytest

from core import policy_loader


@pytest.fixture(autouse=True)
def restore_module_state():
    """Snapshot and restore core.policy_loader's module-level globals so
    tests that call _init_policies() (which mutates globals) don't leak
    state into other tests or other test files relying on real policy data."""
    snapshot = {
        "_domain_anchors": policy_loader._domain_anchors,
        "_suspicious_phrases": policy_loader._suspicious_phrases,
        "_jailbreak_patterns": policy_loader._jailbreak_patterns,
        "_instruction_override_patterns": policy_loader._instruction_override_patterns,
        "_hard_ban_keywords": policy_loader._hard_ban_keywords,
    }
    yield
    for name, value in snapshot.items():
        setattr(policy_loader, name, value)


# --- _load_json_file: the low-level file loader -----------------------------

def test_load_json_file_returns_parsed_content_for_valid_json(tmp_path):
    path = tmp_path / "valid.json"
    path.write_text(json.dumps({"domains": ["Science", "Math"]}), encoding="utf-8")

    result = policy_loader._load_json_file(str(path))

    assert result == {"domains": ["Science", "Math"]}


def test_load_json_file_returns_none_for_missing_file(tmp_path, caplog):
    missing_path = tmp_path / "does_not_exist.json"

    with caplog.at_level("WARNING"):
        result = policy_loader._load_json_file(str(missing_path))

    assert result is None
    assert any(
        str(missing_path) in record.getMessage() and "not found" in record.getMessage()
        for record in caplog.records
    )


def test_load_json_file_returns_none_for_malformed_json_syntax(tmp_path, caplog):
    path = tmp_path / "broken.json"
    path.write_text("{ this is not valid json ]", encoding="utf-8")

    with caplog.at_level("WARNING"):
        result = policy_loader._load_json_file(str(path))

    assert result is None
    assert any(
        "Failed to load" in record.getMessage() and str(path) in record.getMessage()
        for record in caplog.records
    )


def test_load_json_file_returns_none_for_empty_file(tmp_path, caplog):
    """An empty file is not valid JSON and should be treated like any other
    parse failure -- caught and logged, not raised."""
    path = tmp_path / "empty.json"
    path.write_text("", encoding="utf-8")

    with caplog.at_level("WARNING"):
        result = policy_loader._load_json_file(str(path))

    assert result is None
    assert any("Failed to load" in r.getMessage() for r in caplog.records)


def test_load_json_file_handles_json_array_at_top_level(tmp_path):
    """json.load succeeds on any valid JSON document, not just objects --
    verify the loader doesn't assume a dict shape at this layer."""
    path = tmp_path / "array.json"
    path.write_text(json.dumps(["a", "b", "c"]), encoding="utf-8")

    result = policy_loader._load_json_file(str(path))

    assert result == ["a", "b", "c"]


def test_load_json_file_does_not_raise_on_permission_style_errors(tmp_path, caplog):
    """A generic exception during open()/parsing (simulated here by pointing
    at a directory instead of a file) must be caught, not propagated."""
    dir_path = tmp_path / "a_directory"
    dir_path.mkdir()

    with caplog.at_level("WARNING"):
        result = policy_loader._load_json_file(str(dir_path))

    assert result is None


# --- _init_policies: full load sequence, both files -------------------------

def test_init_policies_loads_real_shaped_domain_anchors_and_symbolic_rules(tmp_path, monkeypatch):
    domain_file = tmp_path / "domain_anchors.json"
    domain_file.write_text(
        json.dumps({"domains": ["Artificial Intelligence", "Cybersecurity"]}),
        encoding="utf-8",
    )
    rules_file = tmp_path / "symbolic_rules.json"
    rules_file.write_text(
        json.dumps(
            {
                "suspicious_phrases": ["ignore safety"],
                "jailbreak_patterns": ["dan mode", "jailbreak"],
                "instruction_override_patterns": ["ignore (all )?previous"],
                "hard_ban_keywords": ["anthrax"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(policy_loader, "DOMAIN_ANCHORS_FILE", str(domain_file))
    monkeypatch.setattr(policy_loader, "SYMBOLIC_RULES_FILE", str(rules_file))

    policy_loader._init_policies()

    assert policy_loader.get_domain_anchors() == ["Artificial Intelligence", "Cybersecurity"]
    assert policy_loader.get_suspicious_phrases() == ["ignore safety"]
    assert policy_loader.get_jailbreak_patterns() == ["dan mode", "jailbreak"]
    assert policy_loader.get_instruction_override_patterns() == ["ignore (all )?previous"]
    assert policy_loader.get_hard_ban_keywords() == ["anthrax"]


def test_init_policies_domain_anchors_missing_key_defaults_to_empty_list(tmp_path, monkeypatch):
    """If domain_anchors.json parses but lacks the 'domains' key, the module
    should fail open to an empty list rather than raising a KeyError."""
    domain_file = tmp_path / "domain_anchors.json"
    domain_file.write_text(json.dumps({"unexpected_key": []}), encoding="utf-8")
    rules_file = tmp_path / "symbolic_rules.json"
    rules_file.write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setattr(policy_loader, "DOMAIN_ANCHORS_FILE", str(domain_file))
    monkeypatch.setattr(policy_loader, "SYMBOLIC_RULES_FILE", str(rules_file))

    policy_loader._init_policies()

    assert policy_loader.get_domain_anchors() == []


def test_init_policies_missing_domain_anchors_file_logs_warning_and_fails_open(tmp_path, monkeypatch, caplog):
    domain_file = tmp_path / "does_not_exist.json"
    rules_file = tmp_path / "symbolic_rules.json"
    rules_file.write_text(
        json.dumps(
            {
                "suspicious_phrases": [],
                "jailbreak_patterns": [],
                "instruction_override_patterns": [],
                "hard_ban_keywords": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(policy_loader, "DOMAIN_ANCHORS_FILE", str(domain_file))
    monkeypatch.setattr(policy_loader, "SYMBOLIC_RULES_FILE", str(rules_file))

    with caplog.at_level("WARNING"):
        policy_loader._init_policies()

    assert policy_loader.get_domain_anchors() == []
    assert any(
        "Domain guardrail disabled" in r.getMessage() for r in caplog.records
    )


def test_init_policies_missing_symbolic_rules_file_fails_closed_with_none(tmp_path, monkeypatch, caplog):
    """Per the module docstring/globals: symbolic rules failing to load must
    leave jailbreak/instruction-override/hard-ban lists as None (fail-closed),
    not empty lists -- these are semantically different (None => detection
    disabled/erroring, [] => loaded but empty)."""
    domain_file = tmp_path / "domain_anchors.json"
    domain_file.write_text(json.dumps({"domains": ["X"]}), encoding="utf-8")
    rules_file = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(policy_loader, "DOMAIN_ANCHORS_FILE", str(domain_file))
    monkeypatch.setattr(policy_loader, "SYMBOLIC_RULES_FILE", str(rules_file))

    with caplog.at_level("WARNING"):
        policy_loader._init_policies()

    assert policy_loader.get_jailbreak_patterns() is None
    assert policy_loader.get_instruction_override_patterns() is None
    assert policy_loader.get_hard_ban_keywords() is None
    # suspicious_phrases resets to [] (not None) per the module's own logic
    assert policy_loader.get_suspicious_phrases() == []
    assert any(
        "Symbolic rules not loaded" in r.getMessage() and r.levelname == "ERROR"
        for r in caplog.records
    )


def test_init_policies_malformed_symbolic_rules_json_fails_closed(tmp_path, monkeypatch, caplog):
    domain_file = tmp_path / "domain_anchors.json"
    domain_file.write_text(json.dumps({"domains": []}), encoding="utf-8")
    rules_file = tmp_path / "symbolic_rules.json"
    rules_file.write_text("{ broken json,,", encoding="utf-8")
    monkeypatch.setattr(policy_loader, "DOMAIN_ANCHORS_FILE", str(domain_file))
    monkeypatch.setattr(policy_loader, "SYMBOLIC_RULES_FILE", str(rules_file))

    with caplog.at_level("ERROR"):
        policy_loader._init_policies()

    assert policy_loader.get_jailbreak_patterns() is None
    assert policy_loader.get_instruction_override_patterns() is None
    assert policy_loader.get_hard_ban_keywords() is None
    assert any("Symbolic rules not loaded" in r.getMessage() for r in caplog.records)


def test_init_policies_symbolic_rules_missing_subkeys_default_to_empty_lists(tmp_path, monkeypatch):
    """When symbolic_rules.json parses successfully but omits some of its
    expected keys, those specific fields should default via .get(..., [])
    rather than raising -- distinct from the whole-file-missing case which
    fails closed to None."""
    domain_file = tmp_path / "domain_anchors.json"
    domain_file.write_text(json.dumps({"domains": []}), encoding="utf-8")
    rules_file = tmp_path / "symbolic_rules.json"
    rules_file.write_text(json.dumps({"suspicious_phrases": ["bypass rules"]}), encoding="utf-8")
    monkeypatch.setattr(policy_loader, "DOMAIN_ANCHORS_FILE", str(domain_file))
    monkeypatch.setattr(policy_loader, "SYMBOLIC_RULES_FILE", str(rules_file))

    policy_loader._init_policies()

    assert policy_loader.get_suspicious_phrases() == ["bypass rules"]
    assert policy_loader.get_jailbreak_patterns() == []
    assert policy_loader.get_instruction_override_patterns() == []
    assert policy_loader.get_hard_ban_keywords() == []


def test_get_accessors_reflect_current_module_state_without_copying(tmp_path, monkeypatch):
    """Accessor functions should just return the live module state (no
    defensive copy is documented/expected), matching real usage as read-only
    module-level singletons."""
    domain_file = tmp_path / "domain_anchors.json"
    domain_file.write_text(json.dumps({"domains": ["Only Domain"]}), encoding="utf-8")
    rules_file = tmp_path / "symbolic_rules.json"
    rules_file.write_text(
        json.dumps(
            {
                "suspicious_phrases": ["p1"],
                "jailbreak_patterns": ["j1"],
                "instruction_override_patterns": ["i1"],
                "hard_ban_keywords": ["h1"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(policy_loader, "DOMAIN_ANCHORS_FILE", str(domain_file))
    monkeypatch.setattr(policy_loader, "SYMBOLIC_RULES_FILE", str(rules_file))

    policy_loader._init_policies()

    assert policy_loader.get_domain_anchors() is policy_loader._domain_anchors
    assert policy_loader.get_suspicious_phrases() is policy_loader._suspicious_phrases
