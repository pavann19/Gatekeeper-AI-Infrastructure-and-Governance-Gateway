from concurrent.futures import ThreadPoolExecutor

import pytest

from core.review_queue import ReviewQueue

CONTEXT = dict(request_id='r', tenant='t', principal='p', capability='GENERAL',
               action='tool', policy_digest='digest', security_version='version', decision_context='original')


def approved(tmp_path):
    queue = ReviewQueue(str(tmp_path / 'review.json'))
    record = queue.enqueue('r', 'GENERAL', 'HIGH', 't', 'h', 'r', CONTEXT)
    queue.resolve(record.review_id, 'APPROVED', 'reviewer')
    return queue, record.review_id


@pytest.mark.parametrize('key', list(CONTEXT))
def test_every_binding_component_is_required(tmp_path, key):
    queue, identity = approved(tmp_path)
    with pytest.raises(ValueError):
        queue.consume(identity, {**CONTEXT, key: 'different'})
    assert queue.get(identity)['consumed_at'] is None
    assert queue.consume(identity, CONTEXT)['consumed_at']
    with pytest.raises(ValueError):
        queue.consume(identity, CONTEXT)


def test_concurrent_consumption_reserves_exactly_once(tmp_path):
    queue, identity = approved(tmp_path)
    def consume(_):
        try:
            queue.consume(identity, CONTEXT)
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(consume, range(20))) == 1
    assert ReviewQueue(queue.path).get(identity)['consumed_at']
