"""
Additional edge-case coverage for core.policy_versioning, on top of
tests/test_policy_versioning.py: version ordering correctness, naming/
timestamp uniqueness under rapid repeats, exact byte-for-byte rollback
content, a full deploy->deploy->rollback->rollback round trip, and
auto-creation of the versions directory when it doesn't exist yet.

core/policy_versioning.py has no explicit disk-full/permission-error
handling (no try/except around the filesystem calls), so those scenarios
are intentionally not fabricated here -- there is no real behavior to pin.
"""
import json
import os
import time

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


# --- version history ordering ------------------------------------------------

def test_list_versions_is_strictly_newest_first_by_timestamp(live_policy, monkeypatch):
    """list_versions is documented as 'newest first' -- pin the actual
    order using distinct, controlled timestamps rather than relying on
    real-clock granularity."""
    import core.policy_versioning as pv

    fake_times = iter([
        "20260101T000000Z",
        "20260102T000000Z",
        "20260103T000000Z",
    ])

    class FakeDatetime:
        @classmethod
        def now(cls, tz=None):
            return cls()

        def strftime(self, fmt):
            return next(fake_times)

    monkeypatch.setattr(pv, "datetime", FakeDatetime)

    with open(live_policy, "a", encoding="utf-8") as f:
        f.write(" a")
    v1 = snapshot_policy(live_policy)

    with open(live_policy, "a", encoding="utf-8") as f:
        f.write(" b")
    v2 = snapshot_policy(live_policy)

    with open(live_policy, "a", encoding="utf-8") as f:
        f.write(" c")
    v3 = snapshot_policy(live_policy)

    versions = list_versions()
    assert versions == [v3, v2, v1], (
        f"expected strictly newest-first order [{v3}, {v2}, {v1}], got {versions}"
    )
    assert v1.startswith("20260101T000000Z")
    assert v2.startswith("20260102T000000Z")
    assert v3.startswith("20260103T000000Z")


def test_list_versions_orders_purely_by_filename_not_mtime(live_policy, monkeypatch):
    """The module docstring says the directory listing IS the version
    history and filenames sort chronologically 'by construction' -- prove
    ordering comes from the timestamp in the name, not filesystem mtime,
    by writing files out of mtime order but with in-order names."""
    d = live_policy.rsplit(os.sep, 1)[0] + os.sep + "policy_versions"
    os.makedirs(d, exist_ok=True)

    # Create the "older-named" file SECOND, so its mtime is newer than the
    # "newer-named" file's mtime -- if sorting used mtime, order would flip.
    newer_name = "20260201T000000Z__aaaaaaaaaaaa.json"
    older_name = "20260101T000000Z__bbbbbbbbbbbb.json"

    with open(os.path.join(d, newer_name), "w", encoding="utf-8") as f:
        f.write("newer")
    time.sleep(0.01)
    with open(os.path.join(d, older_name), "w", encoding="utf-8") as f:
        f.write("older")

    versions = list_versions()
    assert versions == [newer_name, older_name]


# --- naming / timestamp uniqueness ------------------------------------------

def test_rapid_successive_snapshots_never_collide_in_versions_list(live_policy):
    """Deploying/snapshotting twice in a tight loop must never produce a
    silently-overwritten version: each distinct content must remain
    independently listed and restorable, even if two snapshots land in the
    same wall-clock second."""
    produced = []
    for i in range(10):
        with open(live_policy, "a", encoding="utf-8") as f:
            f.write(f" {i}")
        version = snapshot_policy(live_policy)
        assert version is not None
        produced.append(version)

    # No two distinct-content snapshots collided into the same filename.
    assert len(set(produced)) == len(produced), f"collision among: {produced}"

    versions = list_versions()
    for v in produced:
        assert v in versions


def test_snapshotting_identical_content_twice_in_immediate_succession_is_distinguishable_or_idempotent(live_policy):
    """Same content hashed twice in the same second would produce the same
    '<timestamp>__<hash>' name if the timestamp also matches -- verify this
    doesn't crash and that the version is still usable for rollback
    (whether it's treated as one entry or overwritten harmlessly, the
    content on disk must stay correct)."""
    v1 = snapshot_policy(live_policy)
    v2 = snapshot_policy(live_policy)  # identical content, no mutation in between

    assert v1 is not None and v2 is not None
    # Whatever the naming collision behavior, the version(s) present must
    # still round-trip the exact original content.
    for v in {v1, v2}:
        assert v in list_versions()
        restored_path = os.path.join(
            live_policy.rsplit(os.sep, 1)[0], "policy_versions", v
        )
        with open(restored_path, encoding="utf-8") as f:
            snapshot_content = f.read()
        with open(live_policy, encoding="utf-8") as f:
            live_content = f.read()
        assert snapshot_content == live_content


# --- byte-for-byte rollback content ------------------------------------------

def test_rollback_restores_byte_for_byte_including_whitespace_and_no_trailing_newline(live_policy):
    """Not just 'some content' -- exact bytes, including trailing
    whitespace/newline quirks that a naive re-serialization would lose."""
    exact_bytes = b'{"default_action": "BLOCK", "weird":   "  spaced  ", "tab":"\t"}'
    with open(live_policy, "wb") as f:
        f.write(exact_bytes)

    version = snapshot_policy(live_policy)

    with open(live_policy, "wb") as f:
        f.write(b'{"default_action": "ALLOW"}')

    rollback_to(version, path=live_policy)

    with open(live_policy, "rb") as f:
        restored = f.read()
    assert restored == exact_bytes


# --- full deploy -> deploy -> rollback -> rollback round trip ---------------

def test_deploy_deploy_rollback_rollback_round_trip_ends_at_original_state(tmp_path, live_policy):
    original_content = open(live_policy, encoding="utf-8").read()

    candidate_a = tmp_path / "candidate_a.json"
    candidate_a.write_text(json.dumps({"default_action": "ALLOW", "marker": "A"}), encoding="utf-8")
    candidate_b = tmp_path / "candidate_b.json"
    candidate_b.write_text(json.dumps({"default_action": "RESTRICT", "marker": "B"}), encoding="utf-8")

    # Deploy A: snapshot(original) -> live becomes A
    snap_before_a = deploy_policy(str(candidate_a), live_path=live_policy)
    assert open(live_policy, encoding="utf-8").read() == candidate_a.read_text(encoding="utf-8")

    # Deploy B: snapshot(A) -> live becomes B
    snap_before_b = deploy_policy(str(candidate_b), live_path=live_policy)
    assert open(live_policy, encoding="utf-8").read() == candidate_b.read_text(encoding="utf-8")

    # Rollback 1: undo deploy of B -> live becomes A again
    rollback_to(snap_before_b, path=live_policy)
    assert open(live_policy, encoding="utf-8").read() == candidate_a.read_text(encoding="utf-8")

    # Rollback 2: undo deploy of A -> live becomes original again
    rollback_to(snap_before_a, path=live_policy)
    assert open(live_policy, encoding="utf-8").read() == original_content


# --- versions directory auto-creation ---------------------------------------

def test_deploy_auto_creates_versions_dir_when_missing(tmp_path, monkeypatch):
    versions_dir = tmp_path / "does" / "not" / "exist" / "yet"
    monkeypatch.setattr("core.config.settings.POLICY_VERSIONS_DIR", str(versions_dir))
    assert not versions_dir.exists()

    live = tmp_path / "policy_rules.json"
    live.write_text(json.dumps({"default_action": "BLOCK"}), encoding="utf-8")

    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"default_action": "ALLOW"}), encoding="utf-8")

    previous_version = deploy_policy(str(candidate), live_path=str(live))

    assert versions_dir.exists() and versions_dir.is_dir()
    assert previous_version is not None
    assert os.path.exists(os.path.join(str(versions_dir), previous_version))


def test_list_versions_auto_creates_versions_dir_when_missing(tmp_path, monkeypatch):
    versions_dir = tmp_path / "brand" / "new" / "dir"
    monkeypatch.setattr("core.config.settings.POLICY_VERSIONS_DIR", str(versions_dir))
    assert not versions_dir.exists()

    result = list_versions()

    assert result == []
    assert versions_dir.exists() and versions_dir.is_dir()
