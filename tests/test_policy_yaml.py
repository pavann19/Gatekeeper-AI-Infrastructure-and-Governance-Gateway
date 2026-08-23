"""
Tests for YAML policy file support (Phase 3, Policy-as-Code).
core/policy.py::_parse_policy_file dispatches on extension.
"""
import json

import pytest

from core.policy import PolicyStore, policy_decision

VALID_YAML = """
default_action: BLOCK
tenants:
  default:
    policies:
      GENERAL:
        HIGH: BLOCK
        MEDIUM: RESTRICT
        LOW: ALLOW
"""

VALID_JSON = json.dumps({
    "default_action": "BLOCK",
    "tenants": {"default": {"policies": {"GENERAL": {"HIGH": "BLOCK", "MEDIUM": "RESTRICT", "LOW": "ALLOW"}}}},
})


@pytest.mark.parametrize("filename,content", [
    ("policy.yaml", VALID_YAML),
    ("policy.yml", VALID_YAML),
])
def test_yaml_extensions_load_correctly(tmp_path, filename, content):
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")
    store = PolicyStore(path=str(path))
    store.load()
    assert len(store) == 1
    action, _ = policy_decision("GENERAL", "HIGH", store=store)
    assert action == "BLOCK"


def test_json_extension_still_uses_json_parser(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(VALID_JSON, encoding="utf-8")
    store = PolicyStore(path=str(path))
    store.load()
    assert len(store) == 1


def test_yaml_and_json_produce_identical_decisions_for_equivalent_content(tmp_path):
    """The whole point of adding a second format: it must not change
    what the policy actually enforces, only how it's authored."""
    yaml_path = tmp_path / "policy.yaml"
    json_path = tmp_path / "policy.json"
    yaml_path.write_text(VALID_YAML, encoding="utf-8")
    json_path.write_text(VALID_JSON, encoding="utf-8")

    yaml_store = PolicyStore(path=str(yaml_path))
    json_store = PolicyStore(path=str(json_path))

    for capability, risk in [("GENERAL", "HIGH"), ("GENERAL", "MEDIUM"), ("GENERAL", "LOW")]:
        yaml_result = policy_decision(capability, risk, store=yaml_store)
        json_result = policy_decision(capability, risk, store=json_store)
        assert yaml_result == json_result


def test_yaml_safe_load_used_not_full_load(tmp_path):
    """yaml.safe_load must be used, not yaml.load -- a policy file that
    decides every request's fate must not also be a code-execution vector
    for whoever can write to it (e.g. via !!python/object tags)."""
    malicious = """
default_action: BLOCK
tenants:
  default:
    policies: !!python/object/apply:os.system ["echo pwned"]
"""
    path = tmp_path / "policy.yaml"
    path.write_text(malicious, encoding="utf-8")
    store = PolicyStore(path=str(path))
    store.load()
    # safe_load refuses the tag and raises during parse; the store must
    # fail closed (unusable), not execute anything.
    assert len(store) == 0


def test_shipped_yaml_example_is_valid():
    """policy_rules.yaml, the real shipped example file, must load cleanly."""
    import os
    if not os.path.exists("policy_rules.yaml"):
        pytest.skip("policy_rules.yaml not present in this checkout")
    store = PolicyStore(path="policy_rules.yaml")
    store.load()
    assert len(store) >= 1
    action, _ = policy_decision("GENERAL", "HIGH", store=store)
    assert action in ("BLOCK", "RESTRICT", "ALLOW")
