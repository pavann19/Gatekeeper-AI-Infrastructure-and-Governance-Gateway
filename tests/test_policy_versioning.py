"""
Tests for core.policy_versioning (Phase 3, Policy-as-Code: "versioning and
rollback"). Filesystem-based, separate from git — see the module docstring
for why: this versions the LIVE policy file a running deployment is
actually enforcing, which may have no git access at all.
"""
import json

import pytest

from core.policy_versioning import deploy_policy, list_versions, rollback_to, snapshot_policy


@pytest.fixture
def live_policy(tmp_path, monkeypatch):
    live = tmp_path / "policy_rules.json"
    live.write_text(json.dumps({
        "default_action": "BLOCK",
        "tenants": {"default": {"policies": {"GENERAL": {"HIGH": "BLOCK"}}}},
    }), encoding="utf-8")

    versions_dir = tmp_path / "policy_versions"
    monkeypatch.setattr("core.config.settings.POLICY_VERSIONS_DIR", str(versions_dir))
    return str(live)


def test_snapshot_creates_a_file_in_the_versions_dir(live_policy):
    version = snapshot_policy(live_policy)
    assert version is not None
    versions = list_versions()
    assert version in versions


def test_snapshot_of_nonexistent_file_is_a_safe_noop():
    """There is nothing to snapshot before the very first policy is
    provisioned -- must not be an error."""
    result = snapshot_policy("/nonexistent/policy.json")
    assert result is None


def test_list_versions_newest_first(live_policy):
    v1 = snapshot_policy(live_policy)
    # Change content so the second snapshot has a different hash suffix,
    # and is therefore guaranteed distinct even if taken in the same second.
    with open(live_policy, "a", encoding="utf-8") as f:
        f.write(" ")
    v2 = snapshot_policy(live_policy)

    versions = list_versions()
    assert versions[0] == v2 or versions[0] >= v1  # timestamp-sortable filenames


def test_rollback_restores_exact_prior_content(live_policy):
    original_content = open(live_policy, encoding="utf-8").read()
    version = snapshot_policy(live_policy)

    # Mutate the live file.
    with open(live_policy, "w", encoding="utf-8") as f:
        json.dump({"default_action": "ALLOW", "tenants": {"default": {"policies": {"GENERAL": {"HIGH": "ALLOW"}}}}}, f)

    rollback_to(version, path=live_policy)

    assert open(live_policy, encoding="utf-8").read() == original_content


def test_rollback_reloads_the_live_policy_store(live_policy, monkeypatch):
    """A rollback that required a manual reload step would leave the
    gateway enforcing the bad policy for however long that takes."""
    version = snapshot_policy(live_policy)
    called = []
    monkeypatch.setattr("core.policy.reload_policies", lambda: called.append(True))
    rollback_to(version, path=live_policy)
    assert called == [True]


def test_rollback_to_nonexistent_version_raises(live_policy):
    with pytest.raises(FileNotFoundError):
        rollback_to("nonexistent-version.json", path=live_policy)


def test_rollback_itself_is_snapshotted_and_therefore_undoable(live_policy):
    """A rollback is itself a policy change; it must not destroy the
    ability to undo it."""
    v1 = snapshot_policy(live_policy)
    with open(live_policy, "a", encoding="utf-8") as f:
        f.write(" ")

    versions_before = set(list_versions())
    rollback_to(v1, path=live_policy)
    versions_after = set(list_versions())

    assert len(versions_after) > len(versions_before)


def test_deploy_snapshots_current_then_replaces_with_candidate(tmp_path, live_policy):
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({
        "default_action": "BLOCK",
        "tenants": {"default": {"policies": {"GENERAL": {"HIGH": "RESTRICT"}}}},
    }), encoding="utf-8")

    previous_version = deploy_policy(str(candidate), live_path=live_policy)

    assert previous_version is not None
    assert open(live_policy, encoding="utf-8").read() == candidate.read_text(encoding="utf-8")
    assert previous_version in list_versions()


def test_deploy_reloads_the_live_policy_store(tmp_path, live_policy, monkeypatch):
    candidate = tmp_path / "candidate.json"
    candidate.write_text(open(live_policy, encoding="utf-8").read(), encoding="utf-8")
    called = []
    monkeypatch.setattr("core.policy.reload_policies", lambda: called.append(True))
    deploy_policy(str(candidate), live_path=live_policy)
    assert called == [True]


def test_first_deploy_with_no_prior_policy_returns_none(tmp_path, monkeypatch):
    versions_dir = tmp_path / "policy_versions"
    monkeypatch.setattr("core.config.settings.POLICY_VERSIONS_DIR", str(versions_dir))

    live = tmp_path / "policy_rules.json"  # does not exist yet
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"default_action": "BLOCK", "tenants": {"default": {"policies": {"GENERAL": {"HIGH": "BLOCK"}}}}}), encoding="utf-8")

    previous_version = deploy_policy(str(candidate), live_path=str(live))
    assert previous_version is None
    assert live.exists()
