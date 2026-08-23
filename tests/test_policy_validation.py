"""
Tests for core.policy.validate_policy_file (Phase 3, Policy-as-Code:
"Validation step"). Unlike PolicyStore.load(), which silently drops one
malformed tenant to keep serving its siblings, this reports EVERY problem
at once for an operator checking a file before deploying it.
"""
import json

from core.policy import validate_policy_file


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_valid_file_has_no_errors(tmp_path):
    path = write(tmp_path / "policy.json", {
        "default_action": "BLOCK",
        "tenants": {"default": {"policies": {"GENERAL": {"HIGH": "BLOCK", "LOW": "ALLOW"}}}},
    })
    assert validate_policy_file(path) == []


def test_missing_file():
    errors = validate_policy_file("/nonexistent/path/policy.json")
    assert len(errors) == 1
    assert "not found" in errors[0].lower()


def test_not_valid_json_or_yaml(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text("{ this is not valid json or yaml: [[[", encoding="utf-8")
    errors = validate_policy_file(str(path))
    assert len(errors) == 1


def test_missing_tenants_key(tmp_path):
    path = write(tmp_path / "policy.json", {"default_action": "BLOCK"})
    errors = validate_policy_file(path)
    assert any("tenants" in e for e in errors)


def test_missing_default_tenant(tmp_path):
    path = write(tmp_path / "policy.json", {
        "default_action": "BLOCK",
        "tenants": {"acme": {"policies": {"GENERAL": {"HIGH": "BLOCK"}}}},
    })
    errors = validate_policy_file(path)
    assert any("default" in e.lower() for e in errors)


def test_invalid_default_action(tmp_path):
    path = write(tmp_path / "policy.json", {
        "default_action": "MAYBE",
        "tenants": {"default": {"policies": {"GENERAL": {"HIGH": "BLOCK"}}}},
    })
    errors = validate_policy_file(path)
    assert any("default_action" in e for e in errors)


def test_invalid_action_value_reported_with_full_path(tmp_path):
    path = write(tmp_path / "policy.json", {
        "default_action": "BLOCK",
        "tenants": {"default": {"policies": {"GENERAL": {"HIGH": "MAYBE_BLOCK"}}}},
    })
    errors = validate_policy_file(path)
    assert any("tenants.default.policies.GENERAL.HIGH" in e for e in errors)


def test_multiple_problems_are_all_reported_at_once(tmp_path):
    """The entire reason this function exists rather than reusing the
    loader's fail-fast-per-entry behaviour: report everything, not one
    problem discovered per fix-and-rerun cycle."""
    path = write(tmp_path / "policy.json", {
        "default_action": "NOT_AN_ACTION",
        "tenants": {
            "acme": {"policies": {"GENERAL": {"HIGH": "ALSO_NOT_AN_ACTION"}}},
        },
    })
    errors = validate_policy_file(path)
    # default_action + missing "default" tenant + acme's bad action = 3 distinct problems
    assert len(errors) >= 3


def test_tenant_entry_not_an_object(tmp_path):
    path = write(tmp_path / "policy.json", {
        "default_action": "BLOCK",
        "tenants": {"default": "not an object"},
    })
    errors = validate_policy_file(path)
    assert any("tenants.default" in e for e in errors)


def test_empty_policies_object(tmp_path):
    path = write(tmp_path / "policy.json", {
        "default_action": "BLOCK",
        "tenants": {"default": {"policies": {}}},
    })
    errors = validate_policy_file(path)
    assert any("policies" in e for e in errors)
