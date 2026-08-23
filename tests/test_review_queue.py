"""
Tests for core.review_queue (Phase 4: Human Review). Backs the REVIEW
decision outcome — a request queued for a human to resolve rather than
auto-allowed or auto-blocked.
"""
import pytest

from core.review_queue import ReviewQueue


@pytest.fixture
def queue(tmp_path):
    return ReviewQueue(path=str(tmp_path / "review_queue.json"))


def test_enqueue_creates_a_pending_record(queue):
    record = queue.enqueue(
        reason="Policy applied for GENERAL (Risk: MEDIUM)", capability="GENERAL",
        risk="MEDIUM", tenant="default", prompt_hash="abc123", request_id="req-1",
    )
    assert record.status == "PENDING"
    assert record.reviewer is None
    assert record.final_decision is None
    assert record.review_id


def test_no_raw_prompt_text_is_stored(queue):
    """Only the hash -- matching core/logger.py's audit-record privacy
    convention exactly."""
    record = queue.enqueue(
        reason="r", capability="GENERAL", risk="HIGH", tenant="default",
        prompt_hash="deadbeef", request_id="req-1",
    )
    record_dict = queue.get(record.review_id)
    assert "prompt" not in record_dict
    assert "prompt_text" not in record_dict
    assert record_dict["prompt_hash"] == "deadbeef"


def test_get_returns_none_for_unknown_id(queue):
    assert queue.get("nonexistent") is None


def test_list_pending_excludes_resolved(queue):
    r1 = queue.enqueue(reason="r", capability="GENERAL", risk="MEDIUM",
                       tenant="default", prompt_hash="h1", request_id="req-1")
    r2 = queue.enqueue(reason="r", capability="GENERAL", risk="MEDIUM",
                       tenant="default", prompt_hash="h2", request_id="req-2")
    queue.resolve(r1.review_id, "APPROVED", reviewer="reviewer-1")

    pending = queue.list_pending()
    pending_ids = {p["review_id"] for p in pending}
    assert r1.review_id not in pending_ids
    assert r2.review_id in pending_ids


def test_resolve_approved_sets_final_decision_allow(queue):
    record = queue.enqueue(reason="r", capability="GENERAL", risk="MEDIUM",
                           tenant="default", prompt_hash="h", request_id="req-1")
    resolved = queue.resolve(record.review_id, "APPROVED", reviewer="alice")
    assert resolved["status"] == "APPROVED"
    assert resolved["final_decision"] == "ALLOW"
    assert resolved["reviewer"] == "alice"
    assert resolved["resolved_at"] is not None


def test_resolve_rejected_sets_final_decision_block(queue):
    record = queue.enqueue(reason="r", capability="GENERAL", risk="HIGH",
                           tenant="default", prompt_hash="h", request_id="req-1")
    resolved = queue.resolve(record.review_id, "REJECTED", reviewer="bob")
    assert resolved["status"] == "REJECTED"
    assert resolved["final_decision"] == "BLOCK"


def test_resolve_unknown_review_raises_keyerror(queue):
    with pytest.raises(KeyError):
        queue.resolve("nonexistent", "APPROVED", reviewer="alice")


def test_resolve_twice_raises_valueerror(queue):
    """A double-click or a race between two reviewers must be loud, not
    silently overwrite who resolved it first and why."""
    record = queue.enqueue(reason="r", capability="GENERAL", risk="MEDIUM",
                           tenant="default", prompt_hash="h", request_id="req-1")
    queue.resolve(record.review_id, "APPROVED", reviewer="alice")
    with pytest.raises(ValueError):
        queue.resolve(record.review_id, "REJECTED", reviewer="bob")


def test_resolve_invalid_outcome_raises_valueerror(queue):
    record = queue.enqueue(reason="r", capability="GENERAL", risk="MEDIUM",
                           tenant="default", prompt_hash="h", request_id="req-1")
    with pytest.raises(ValueError):
        queue.resolve(record.review_id, "MAYBE", reviewer="alice")


def test_state_persists_across_instances(tmp_path):
    """Not just an in-memory dict -- a second ReviewQueue instance pointed
    at the same path must see what the first one wrote."""
    path = str(tmp_path / "review_queue.json")
    q1 = ReviewQueue(path=path)
    record = q1.enqueue(reason="r", capability="GENERAL", risk="MEDIUM",
                        tenant="default", prompt_hash="h", request_id="req-1")

    q2 = ReviewQueue(path=path)
    assert q2.get(record.review_id) is not None
    assert q2.get(record.review_id)["status"] == "PENDING"


def test_corrupt_queue_file_fails_to_empty_not_a_crash(tmp_path):
    path = tmp_path / "review_queue.json"
    path.write_text("{ not valid json [[[", encoding="utf-8")
    queue = ReviewQueue(path=str(path))
    assert queue.list_pending() == []
