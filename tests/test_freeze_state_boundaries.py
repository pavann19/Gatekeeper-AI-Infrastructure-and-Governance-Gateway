import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from core.policy import PolicyStore, policy_decision
from core.policy_versioning import deploy_policy
from core.review_queue import ReviewQueue


def enqueue(queue, i=0):
    return queue.enqueue('reason', 'GENERAL', 'HIGH', 'tenant', f'h{i}', f'req{i}')


def test_independent_instances_lose_no_concurrent_writes(tmp_path):
    path = str(tmp_path / 'reviews.json')
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(lambda i: enqueue(ReviewQueue(path), i), range(40)))
    assert len(ReviewQueue(path).list_pending()) == 40
    assert len({r.review_id for r in records}) == 40


def test_concurrent_resolution_is_idempotent_and_preserves_first_reviewer(tmp_path):
    path = str(tmp_path / 'reviews.json')
    record = enqueue(ReviewQueue(path))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: ReviewQueue(path).resolve(record.review_id, 'APPROVED', str(i)), range(20)))
    assert all(r == results[0] for r in results)
    with pytest.raises(ValueError):
        ReviewQueue(path).resolve(record.review_id, 'REJECTED', 'other')


def test_duplicate_rejection_is_idempotent(tmp_path):
    queue = ReviewQueue(str(tmp_path / 'reviews.json'))
    record = enqueue(queue)
    first = queue.resolve(record.review_id, 'REJECTED', 'a')
    assert queue.resolve(record.review_id, 'REJECTED', 'b') == first


def test_expired_approval_is_denied(tmp_path):
    path = tmp_path / 'reviews.json'
    queue = ReviewQueue(str(path))
    record = enqueue(queue)
    state = json.loads(path.read_text())
    state[record.review_id]['expires_at'] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match='expired'):
        queue.resolve(record.review_id, 'APPROVED', 'a')
    assert queue.get(record.review_id)['status'] == 'PENDING'


def test_corrupt_state_is_preserved_and_cannot_be_overwritten(tmp_path):
    path = tmp_path / 'reviews.json'
    path.write_text('{ broken')
    with pytest.raises(RuntimeError, match='corrupt'):
        enqueue(ReviewQueue(str(path)))
    assert path.read_text() == '{ broken'


def test_reader_cannot_mutate_authoritative_state(tmp_path):
    queue = ReviewQueue(str(tmp_path / 'reviews.json'))
    record = enqueue(queue)
    queue.get(record.review_id)['status'] = 'APPROVED'
    assert queue.get(record.review_id)['status'] == 'PENDING'


def test_invalid_candidate_does_not_replace_live_policy(tmp_path, monkeypatch):
    monkeypatch.setattr('core.config.settings.POLICY_VERSIONS_DIR', str(tmp_path / 'versions'))
    live = tmp_path / 'live.json'
    candidate = tmp_path / 'candidate.json'
    live.write_text(json.dumps({'tenants': {'default': {'policies': {'GENERAL': {'HIGH': 'BLOCK'}}}}}))
    original = live.read_bytes()
    candidate.write_text('{"default_action":"ALLOW"}')
    with pytest.raises(ValueError, match='Invalid policy'):
        deploy_policy(str(candidate), str(live))
    assert live.read_bytes() == original


def test_concurrent_policy_reload_never_exposes_partial_snapshot(tmp_path):
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps({'tenants': {'default': {'policies': {'GENERAL': {'HIGH': 'REVIEW'}}}}}))
    store = PolicyStore(str(path))
    def worker(i):
        if i % 2:
            store.load(force=True)
        return policy_decision('GENERAL', 'HIGH', store=store)[0]
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert set(pool.map(worker, range(100))) == {'REVIEW'}
    with pytest.raises(TypeError):
        store.get('default').policies['GENERAL']['HIGH'] = 'ALLOW'


def test_empty_candidate_store_never_falls_back_to_live_allow_policy(tmp_path):
    store = PolicyStore(str(tmp_path / 'missing.json'))
    assert policy_decision('INTERNAL', 'LOW', store=store)[0] == 'BLOCK'


def test_review_overload_preserves_existing_records(tmp_path, monkeypatch):
    monkeypatch.setattr('core.config.settings.REVIEW_MAX_RECORDS', 1)
    queue = ReviewQueue(str(tmp_path / 'reviews.json'))
    first = enqueue(queue)
    with pytest.raises(RuntimeError, match='capacity'):
        enqueue(queue, 2)
    assert [r['review_id'] for r in queue.list_pending()] == [first.review_id]
