# core/policy_versioning.py
"""
Filesystem-based snapshot and rollback for the LIVE policy file (Phase 3,
Policy-as-Code roadmap item: "versioning and rollback").

WHY THIS EXISTS SEPARATELY FROM GIT
--------------------------------------
`policy_rules.json`/`.yaml` is already a tracked file, and for the file
sitting in this repository, git already is version history -- building a
parallel version-control system for one config file would be reinventing
something git already does correctly. What git does NOT cover is the file
as it exists on a RUNNING deployment: `core/policy.py::reload_policies()`
already supports hot-reloading a new policy into a live process without a
restart, and an operator doing that against a mounted volume (see
config/README.md) may have no git access to that server at all, or may be
deploying a policy that never goes through a commit (generated, or pushed
by a separate tenant-management tool). This module versions THAT file --
the one actually being enforced right now -- independent of whether it is
also tracked in git somewhere.

DESIGN
------
Every snapshot is a full copy of the policy file at that moment, named
`<timestamp>__<sha256[:12]>.<original-extension>` in POLICY_VERSIONS_DIR.
Content-addressed by the hash suffix so identical content taken at two
different times is still distinguishable by timestamp, and a snapshot's
integrity can be checked by re-hashing it. No metadata database, no SQLite
-- the directory listing IS the version history, which means an operator
can inspect, back up, or prune it with plain filesystem tools if this
module is ever unavailable.
"""
import hashlib
import os
import shutil
from datetime import datetime, timezone

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


def _versions_dir() -> str:
    d = settings.POLICY_VERSIONS_DIR
    os.makedirs(d, exist_ok=True)
    return d


def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def snapshot_policy(path: str = None) -> str:
    """
    Copies the current policy file into POLICY_VERSIONS_DIR. Returns the
    version filename (not a full path — that's an implementation detail of
    _versions_dir the caller shouldn't need).

    Silently no-ops (returns None) if `path` doesn't exist yet — there is
    nothing to snapshot before the very first policy file is provisioned,
    and that must not be an error.
    """
    path = path or settings.POLICY_RULES_FILE
    if not os.path.exists(path):
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    content_hash = _file_hash(path)
    ext = os.path.splitext(path)[1] or ".json"
    version_name = f"{timestamp}__{content_hash}{ext}"

    dest = os.path.join(_versions_dir(), version_name)
    shutil.copy2(path, dest)
    logger.info(f"Policy snapshot saved: {version_name}")
    return version_name


def list_versions() -> list[str]:
    """Newest first — snapshot filenames sort chronologically by construction."""
    d = _versions_dir()
    return sorted(
        (f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))),
        reverse=True,
    )


def rollback_to(version_name: str, path: str = None) -> None:
    """
    Restores a snapshot over the live policy file and reloads it
    immediately — a rollback that required a separate manual reload step
    would leave the gateway enforcing the bad policy for however long that
    second step takes.

    Raises FileNotFoundError if `version_name` doesn't exist, rather than
    silently no-op-ing: a rollback request that didn't roll back anything
    must be loud, not swallowed.
    """
    path = path or settings.POLICY_RULES_FILE
    src = os.path.join(_versions_dir(), version_name)
    if not os.path.exists(src):
        raise FileNotFoundError(f"No such policy version: {version_name}")

    # Snapshot what's about to be overwritten too -- a rollback is itself a
    # policy change, and should be undoable the same way any other one is.
    snapshot_policy(path)

    shutil.copy2(src, path)
    logger.warning(f"Policy rolled back to version {version_name}")

    from core.policy import reload_policies
    reload_policies()


def deploy_policy(new_path: str, live_path: str = None) -> str:
    """
    The safe way to change the live policy: snapshot the current one, then
    replace it with `new_path`'s content, then reload. Returns the snapshot
    version name that was taken immediately before the swap (None if there
    was nothing to snapshot yet), so the caller always has the exact
    rollback target for what it just deployed.

    Does NOT validate `new_path` — callers (the CLI, an admin endpoint)
    should call core.policy.validate_policy_file first and refuse to reach
    this function at all on a validation failure, the same discipline
    scripts/simulate_policy.py already applies before simulating.
    """
    live_path = live_path or settings.POLICY_RULES_FILE
    previous_version = snapshot_policy(live_path)

    shutil.copy2(new_path, live_path)
    logger.info(f"Policy deployed from {new_path} to {live_path}")

    from core.policy import reload_policies
    reload_policies()

    return previous_version
