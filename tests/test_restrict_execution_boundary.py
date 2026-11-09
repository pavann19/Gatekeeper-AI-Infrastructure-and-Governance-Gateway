import json

from core.mcp_compat import handle_mcp_tool_call
from core.policy import PolicyStore, request_policy
from core.tools import ToolRegistry, ToolSpec, execute_tool


def test_restrict_denies_tool_effects_through_both_shared_and_mcp_paths(tmp_path, monkeypatch):
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps({'tenants': {'default': {'policies': {'GENERAL': {'MEDIUM': 'RESTRICT'}}}}}))
    token = request_policy.set(PolicyStore(str(path)).load())
    monkeypatch.setattr('core.logger.log_tool_event', lambda *a, **k: None)
    calls = []
    registry = ToolRegistry()
    registry.register(ToolSpec('write', 'Write object', risk_level='MEDIUM'), handler=lambda: calls.append(1))
    try:
        direct = execute_tool('GENERAL', 'write', {}, registry=registry, tenant='default')
        mcp = handle_mcp_tool_call('GENERAL', 'write', {}, registry=registry, tenant='default')
        assert direct['decision'] == 'BLOCK'
        assert 'RESTRICT' in direct['reason']
        assert mcp['isError'] is True
        assert calls == []
    finally:
        request_policy.reset(token)
