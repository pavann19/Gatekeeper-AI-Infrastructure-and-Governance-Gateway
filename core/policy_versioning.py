# core/policy_versioning.py
"""Validated policy deployment, immutable snapshots and rollback.

Capture exact bytes, validate the prepared candidate and replace atomically
under a bounded local filesystem lock. Snapshot identity rejects traversal
and tampering. Serving-process reload coordination belongs to the operator."""
import hashlib
import os
import tempfile
from datetime import datetime, timezone

from core.config import settings
from core.state_lock import state_lock
from core.logger import get_logger

logger = get_logger(__name__)


def _atomic_write(data: bytes, destination: str, validate=False):
    directory = os.path.dirname(destination) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=directory, suffix=os.path.splitext(destination)[1])
    try:
        with os.fdopen(fd, "wb") as dst:
            dst.write(data)
            dst.flush()
            os.fsync(dst.fileno())
        if validate:
            from core.policy import validate_policy_file
            errors = validate_policy_file(temporary)
            if errors:
                raise ValueError("Invalid policy: " + "; ".join(errors))
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_copy(source: str, destination: str, validate=False):
    with open(source, 'rb') as src:
        data = src.read()
    _atomic_write(data, destination, validate)


def _versions_dir() -> str:
    d = settings.POLICY_VERSIONS_DIR
    os.makedirs(d, exist_ok=True)
    return d


def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def snapshot_policy(path: str | None = None) -> str | None:
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
    with open(path, 'rb') as source:
        data = source.read()
    content_hash = hashlib.sha256(data).hexdigest()[:12]
    ext = os.path.splitext(path)[1] or ".json"
    version_name = f"{timestamp}__{content_hash}{ext}"

    dest = os.path.join(_versions_dir(), version_name)
    # Existing versions are immutable, including rapid identical snapshots.
    with state_lock(dest):
        if not os.path.exists(dest):
            _atomic_write(data, dest)
        elif _file_hash(dest) != content_hash:
            raise ValueError("Policy version integrity failure")
    logger.info(f"Policy snapshot saved: {version_name}")
    return version_name


def list_versions() -> list[str]:
    """Newest first — snapshot filenames sort chronologically by construction."""
    d = _versions_dir()
    return sorted(
        (f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and not f.endswith('.lock')),
        reverse=True,
    )


def rollback_to(version_name: str, path: str | None = None) -> None:
    """
        Restores a snapshot over the live policy file and reloads it
        immediately — a rollback that required a separate manual reload step
        would leave the gateway enforcing the bad policy for however long that
        second step takes.

        Raises FileNotFoundError if `version_name` doesn't exist, rather than
        silently no-op-ing: a rollback request that didn't roll back anything
        must be loud, not swallowed. The SAME exception is raised if
        `version_name` is anything other than a bare filename (an absolute
        path, or containing a path separator or `..`) -- this function was
        written when its only caller was a local CLI operator who already had
        full filesystem access, so `os.path.join(_versions_dir(), version_name)`
        happily resolving an absolute path (which DISCARDS the versions dir
        entirely -- `os.path.join("/a", "/etc/passwd") == "/etc/passwd"`) or a
        `../` traversal was never exploitable. It became exploitable the
        moment `POST /api/v1/policy/rollback` (implementation) exposed this function
        to any INTERNAL-capability API key over the network: without this
        check, that endpoint is an arbitrary-file-read primitive (roll back to
        any readable file, then GET /api/v1/policy to read it back). Rejected
        the same way a missing version is, not with a distinct error, so a
        caller can't use the response to distinguish "traversal blocked" from
        "that literal filename doesn't exist".
        """
    if (os.path.basename(version_name) != version_name or '/' in version_name
            or '\\' in version_name or '..' in version_name or ':' in version_name):
        raise FileNotFoundError(f"No such policy version: {version_name}")

    path = path or settings.POLICY_RULES_FILE
    src = os.path.join(_versions_dir(), version_name)
    if not os.path.exists(src):
        raise FileNotFoundError(f"No such policy version: {version_name}")
    if os.path.islink(src):
        raise FileNotFoundError('Policy snapshot must be a regular file')
    expected = version_name.rsplit('__', 1)[-1].rsplit('.', 1)[0]
    with open(src, 'rb') as source:
        data = source.read()
    if hashlib.sha256(data).hexdigest()[:12] != expected:
        raise ValueError('Policy snapshot digest mismatch')

    # Snapshot what's about to be overwritten too -- a rollback is itself a
    # policy change, and should be undoable the same way any other one is.
    with state_lock(path):
        snapshot_policy(path)
        _atomic_write(data, path, validate=True)
        from core.policy import reload_policies
        reload_policies()


def deploy_policy(new_path: str, live_path: str | None = None) -> str | None:
    """
    The safe way to change the live policy: snapshot the current one, then
    replace it with `new_path`'s content, then reload. Returns the snapshot
    version name that was taken immediately before the swap (None if there
    was nothing to snapshot yet), so the caller always has the exact
    rollback target for what it just deployed.

    Validates the exact prepared bytes using the destination file format.
    Invalid candidates leave the live file and loaded mapping unchanged.
    """
    live_path = live_path or settings.POLICY_RULES_FILE
    with state_lock(live_path):
        previous_version = snapshot_policy(live_path)
        _atomic_copy(new_path, live_path, validate=True)
        from core.policy import reload_policies
        reload_policies()

    return previous_version
