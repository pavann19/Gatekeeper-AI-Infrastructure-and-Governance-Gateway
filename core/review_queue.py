# core/review_queue.py
"""
Human review queue (Phase 4 roadmap item). Backs the `REVIEW` decision
outcome: a request a tenant's policy maps to REVIEW is neither auto-allowed
nor auto-blocked — it is queued, and a human resolves it to a final
ALLOW/BLOCK.

DESIGN, MIRRORING core/auth.py's KeyStore / core/tenancy.py's TenantStore
--------------------------------------------------------------------------
A single JSON object file (review_queue.json: review_id -> record),
read-modify-written on each mutation — NOT the audit log's append-only
JSONL convention. The audit log is append-only because it never needs to
change once written (a compliance record). A review record's whole point
is that it changes exactly once, from PENDING to APPROVED/REJECTED — an
append-only log would need every reader to replay history to find a
record's current state, for a use case (a handful of pending reviews at
any moment) where that replay buys nothing a mutable file doesn't already
give more simply.

WHAT IS DELIBERATELY NOT STORED
----------------------------------
No raw prompt text — only its SHA-256 hash, matching core/logger.py's
audit-record convention exactly (log_event stores `prompt_hash`, never the
prompt). A reviewer resolving a request needs to know WHAT triggered review
(capability, risk, tenant, reason) to make a judgement call, but this
module does not assume a human reviewer necessarily needs the raw prompt
text visible in this queue's own storage — a real deployment wanting that
would surface it via the request_id, joining against wherever the caller
itself logs the actual prompt (which Gatekeeper, as a sidecar, was never
positioned to be the source of truth for).
"""
import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

VALID_STATUSES = ("PENDING", "APPROVED", "REJECTED")
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
    resolved_at: str = None
    reviewer: str = None
    final_decision: str = None


class ReviewQueue:
    def __init__(self, path=None):
        self.path = path or settings.REVIEW_QUEUE_FILE
        self._records: dict[str, dict] = {}
        self._loaded = False

    def load(self, force=False):
        if self._loaded and not force:
            return self
        self._loaded = True
        if not os.path.exists(self.path):
            self._records = {}
            return self
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self._records = json.load(f)
        except Exception as e:
            # Fail closed to an EMPTY queue, not a crash — a corrupt review
            # queue file must not take down request handling. Reviews
            # already resolved are unaffected (the resolution already fed
            # back to whichever caller checked status before corruption);
            # anything still PENDING is lost, which is the same class of
            # data-loss risk core/auth.py's KeyStore accepts for the same
            # reason: a queue this codebase can't parse is not a queue this
            # codebase can safely keep enforcing decisions from either.
            logger.error(f"Review queue at {self.path} unreadable ({e}); starting empty.")
            self._records = {}
        return self

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, indent=2)
        os.replace(tmp_path, self.path)  # atomic on both POSIX and Windows

    def enqueue(self, reason, capability, risk, tenant, prompt_hash, request_id) -> ReviewRecord:
        self.load()
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
        )
        self._records[record.review_id] = asdict(record)
        self._save()
        logger.info(f"Review queued: {record.review_id} (tenant={tenant}, risk={risk})")
        return record

    def get(self, review_id: str) -> dict:
        self.load()
        return self._records.get(review_id)

    def list_pending(self) -> list:
        self.load()
        return [r for r in self._records.values() if r["status"] == "PENDING"]

    def resolve(self, review_id: str, outcome: str, reviewer: str) -> dict:
        """
        outcome: "APPROVED" or "REJECTED". Raises KeyError if the review
        doesn't exist, ValueError if it was already resolved — a caller
        resolving twice (a double-click, a race between two reviewers) must
        get a loud error, not silently overwrite who resolved it first and
        why.
        """
        self.load()
        record = self._records.get(review_id)
        if record is None:
            raise KeyError(f"No such review: {review_id}")
        if record["status"] != "PENDING":
            raise ValueError(
                f"Review {review_id} is already {record['status']} "
                f"(by {record.get('reviewer')!r}) — cannot resolve twice."
            )
        if outcome not in ("APPROVED", "REJECTED"):
            raise ValueError(f"outcome must be one of APPROVED/REJECTED, got {outcome!r}")

        record["status"] = outcome
        record["reviewer"] = reviewer
        record["resolved_at"] = datetime.now(timezone.utc).isoformat()
        record["final_decision"] = _RESOLUTION_TO_DECISION[outcome]
        self._save()
        logger.info(f"Review resolved: {review_id} -> {outcome} by {reviewer!r}")
        return record


_queue = ReviewQueue()


def enqueue_review(reason, capability, risk, tenant, prompt_hash, request_id) -> ReviewRecord:
    return _queue.enqueue(reason, capability, risk, tenant, prompt_hash, request_id)


def get_review(review_id: str) -> dict:
    return _queue.get(review_id)


def list_pending_reviews() -> list:
    return _queue.list_pending()


def resolve_review(review_id: str, outcome: str, reviewer: str) -> dict:
    return _queue.resolve(review_id, outcome, reviewer)
