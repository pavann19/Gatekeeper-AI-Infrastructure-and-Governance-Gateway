from dataclasses import asdict

import pytest

from core.approval import approval_context
from core.auth import Principal
from core.review_queue import ReviewQueue
from core.tools import ToolRegistry, ToolSpec, decide_tool_call, execute_tool


@pytest.fixture
def approved_tool(tmp_path, monkeypatch):
    queue = ReviewQueue(str(tmp_path / 'reviews.json'))
    monkeypatch.setattr('core.review_queue._queue', queue)
    monkeypatch.setattr('core.logger.log_tool_event', lambda *a, **k: None)
    registry = ToolRegistry()
    spec = ToolSpec('delete', 'Delete approved object',
                    {'type': 'object', 'properties': {'id': {'type': 'integer'}}, 'required': ['id']},
                    risk_level='HIGH')
    calls = []
    registry.register(spec, handler=lambda id: calls.append(id))
    principal = Principal(capability='GENERAL', tenant='tenant', key_id='owner', authenticated=True)
    result = decide_tool_call('GENERAL', spec, {'id': 1})
    context = approval_context(principal, 'original', 'delete', {'arguments': {'id': 1}, 'spec': asdict(spec)}, result['reason'])
    record = queue.enqueue(result['reason'], 'GENERAL', 'HIGH', 'tenant', 'hash', 'original', context)
    queue.resolve(record.review_id, 'APPROVED', 'reviewer')
    return registry, calls, record.review_id


def test_approval_continues_actual_execution_once(approved_tool):
    registry, calls, identity = approved_tool
    kwargs = dict(registry=registry, tenant='tenant', request_id='original', principal_id='owner', approval_id=identity)
    assert execute_tool('GENERAL', 'delete', {'id': 1}, **kwargs)['decision'] == 'ALLOW'
    assert execute_tool('GENERAL', 'delete', {'id': 1}, **kwargs)['decision'] == 'BLOCK'
    assert calls == [1]


@pytest.mark.parametrize('override', [dict(tenant='other'), dict(request_id='new'), dict(principal_id='foreign')])
def test_changed_context_never_invokes_handler(approved_tool, override):
    registry, calls, identity = approved_tool
    kwargs = dict(registry=registry, tenant='tenant', request_id='original', principal_id='owner', approval_id=identity)
    kwargs.update(override)
    assert execute_tool('GENERAL', 'delete', {'id': 1}, **kwargs)['decision'] == 'BLOCK'
    assert calls == []


def test_changed_arguments_never_invoke_handler(approved_tool):
    registry, calls, identity = approved_tool
    result = execute_tool('GENERAL', 'delete', {'id': 2}, registry=registry, tenant='tenant',
                          request_id='original', principal_id='owner', approval_id=identity)
    assert result['decision'] == 'BLOCK'
    assert calls == []


def test_old_approval_cannot_be_attached_to_new_allow_path(approved_tool):
    registry, calls, identity = approved_tool
    registry.register(ToolSpec('read', 'Read object', risk_level='LOW'), handler=lambda: calls.append(2))
    result = execute_tool('GENERAL', 'read', {}, registry=registry, tenant='tenant',
                          request_id='original', principal_id='owner', approval_id=identity)
    assert result['decision'] == 'BLOCK'
    assert calls == []
