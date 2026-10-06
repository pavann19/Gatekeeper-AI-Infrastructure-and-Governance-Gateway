"""Durable review state, serialized across threads and local processes."""
import json
from copy import deepcopy
import os
import uuid
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from core.config import settings
from core.state_lock import state_lock
from core.logger import get_logger

logger = get_logger(__name__)

VALID_STATUSES = ("PENDING", "APPROVED", "REJECTED", "CONSUMED")
_RESOLUTION_TO_DECISION = {"APPROVED": "ALLOW", "REJECTED": "BLOCK"}


@dataclass
class ReviewRecord:
    review_id: str
    status: str
    reason: str
    capability: str
    risk: str
    tenant: str
    prompt_hash: str
    request_id: str
    created_at: str
    resolved_at: str | None = None
    reviewer: str | None = None
    final_decision: str | None = None
    expires_at: str | None = None
    context: dict = field(default_factory=dict)
    consumed_at: str | None = None


class ReviewQueue:
    def __init__(self, path=None):
        self.path = path or settings.REVIEW_QUEUE_FILE
        self._records: dict[str, dict] = {}
        self._loaded = False
        self._corrupt = False

    def load(self, force=False):
        if self._loaded and not force:
            return self
        self._loaded = True
        self._corrupt = False
        if not os.path.exists(self.path):
            self._records = {}
            return self
        try:
            if os.path.getsize(self.path) > settings.REVIEW_MAX_BYTES:
                raise ValueError('Review store exceeds configured byte limit')
            with open(self.path, "r", encoding="utf-8") as f:
                records = json.load(f)
                if not isinstance(records, dict) or any(
                    not isinstance(r, dict) or r.get("review_id") != key
                    or r.get("status") not in VALID_STATUSES
                    for key, r in records.items()
                ):
                    raise ValueError("Malformed review state")
                self._records = records
        except Exception:
            logger.error("Review state unreadable; mutations disabled until recovery")
            self._records = {}
            self._corrupt = True
        return self

    @contextmanager
    def _transaction(self, write=False):
        with state_lock(self.path):
            self.load(force=True)
            if write and self._corrupt:
                raise RuntimeError("Review state is corrupt; restore it before writing")
            yield
            if write:
                self._save()

    def _save(self):
        payload = json.dumps(self._records, indent=2)
        if len(payload.encode('utf-8')) > settings.REVIEW_MAX_BYTES:
            raise RuntimeError('Review store capacity exhausted')
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".review-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def enqueue(self, reason, capability, risk, tenant, prompt_hash, request_id, context=None) -> ReviewRecord:
        record = ReviewRecord(
            review_id=uuid.uuid4().hex,
            status="PENDING",
            reason=reason,
            capability=capability,
            risk=risk,
            tenant=tenant,
            prompt_hash=prompt_hash,
            request_id=request_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            expires_at=datetime.fromtimestamp(datetime.now(timezone.utc).timestamp() + 3600, timezone.utc).isoformat(),
            context=deepcopy(context or {}),
        )
        with self._transaction(write=True):
            if len(self._records) >= settings.REVIEW_MAX_RECORDS:
                raise RuntimeError('Review store capacity exhausted')
            self._records[record.review_id] = asdict(record)
        logger.info(f"Review queued: {record.review_id} (tenant={tenant}, risk={risk})")
        return record

    def get(self, review_id: str) -> dict | None:
        with self._transaction():
            record = self._records.get(review_id)
            return deepcopy(record) if record is not None else None

    def list_pending(self) -> list:
        with self._transaction():
            return [deepcopy(r) for r in self._records.values() if r["status"] == "PENDING"]

    def resolve(self, review_id: str, outcome: str, reviewer: str) -> dict:
        """
        outcome: "APPROVED" or "REJECTED". Raises KeyError if the review
        doesn't exist. Matching resolutions are idempotent; conflicting
        outcomes cannot overwrite the original reviewer or resolution.
        """
        with self._transaction(write=True):
            record = self._records.get(review_id)
            if record is None:
                raise KeyError(f"No such review: {review_id}")
            if outcome not in ("APPROVED", "REJECTED"):
                raise ValueError("outcome must be APPROVED or REJECTED")
            if record["status"] == outcome or (record['status'] == 'CONSUMED' and outcome == 'APPROVED'):
                return deepcopy(record)
            if record["status"] != "PENDING":
                raise ValueError(
                    f"Review {review_id} is already {record['status']} "
                    f"(by {record.get('reviewer')!r}) — cannot resolve twice."
                )
            expiry = record.get("expires_at")
            if outcome == 'APPROVED' and (not expiry or datetime.fromisoformat(expiry) <= datetime.now(timezone.utc)):
                raise ValueError("Review expired or has no trustworthy expiry")
            record["status"] = outcome
            record["reviewer"] = reviewer
            record["resolved_at"] = datetime.now(timezone.utc).isoformat()
            record["final_decision"] = _RESOLUTION_TO_DECISION[outcome]
            logger.info(f"Review resolved: {review_id} -> {outcome} by {reviewer!r}")
            return deepcopy(record)

    def consume(self, review_id: str, context: dict) -> dict:
        """Atomically reserve an approval before execution; never retry an effect.

        A crash after reservation requires operator investigation. Exactly-once
        effects across arbitrary external handlers are not claimed.
        """
        required = {'request_id', 'tenant', 'principal', 'capability', 'action',
                    'policy_digest', 'security_version', 'decision_context'}
        if set(context) != required or any(not value for value in context.values()):
            raise ValueError("Incomplete approval context")
        with self._transaction(write=True):
            record = self._records.get(review_id)
            if record is None:
                raise KeyError(review_id)
            expiry = record.get('expires_at')
            if (record['status'] != 'APPROVED' or record.get('consumed_at')
                    or not expiry or datetime.fromisoformat(expiry) <= datetime.now(timezone.utc)
                    or record.get('context') != context):
                raise ValueError("Approval is unavailable or context changed")
            record['consumed_at'] = datetime.now(timezone.utc).isoformat()
            record['status'] = 'CONSUMED'
            return deepcopy(record)


_queue = ReviewQueue()


def enqueue_review(reason, capability, risk, tenant, prompt_hash, request_id, context=None) -> ReviewRecord:
    return _queue.enqueue(reason, capability, risk, tenant, prompt_hash, request_id, context)


def get_review(review_id: str) -> dict | None:
    return _queue.get(review_id)


def list_pending_reviews() -> list:
    return _queue.list_pending()


def resolve_review(review_id: str, outcome: str, reviewer: str) -> dict:
    return _queue.resolve(review_id, outcome, reviewer)
