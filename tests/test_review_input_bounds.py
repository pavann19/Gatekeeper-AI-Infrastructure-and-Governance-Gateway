import asyncio
import json

from core.config import settings
from core.metrics import safe_source, unknown_source_total
from core.policy import validate_policy_file
from core.request_ingress import RequestIngress


def test_unknown_metric_sources_have_one_bounded_label():
    for index in range(500):
        assert safe_source('synthetic-untrusted-' + str(index)) == 'other'
    assert safe_source({'untrusted': 'object'}) == 'other'
    labels = {sample.labels['source'] for metric in unknown_source_total.collect()
              for sample in metric.samples if sample.name.endswith('_total')}
    assert labels == {'unknown'}


def test_oversized_content_length_is_rejected_without_reading(monkeypatch):
    monkeypatch.setattr(settings, 'REQUEST_MAX_BODY_BYTES', 4)
    called = []

    async def receive():
        raise AssertionError('Oversized declared body must not be read')

    async def downstream(scope, receive, send):
        raise AssertionError('Oversized body must not reach parsing')

    async def send(message):
        called.append(message)

    asyncio.run(RequestIngress(downstream)({'type': 'http', 'method': 'POST',
                                          'headers': [(b'content-length', b'5')]}, receive, send))
    assert called[0]['status'] == 413


def test_policy_typos_and_unknown_fields_are_rejected(tmp_path):
    path = tmp_path / 'candidate.json'
    path.write_text(json.dumps({'default_action': 'ALLOW', 'unknown': True,
                               'tenants': {'default': {'unknown': 'field', 'policies': {
                                   'GENERALL': {'HIGHH': 'ALLOW'}}}}}))
    errors = validate_policy_file(str(path))
    assert any('top-level' in error for error in errors)
    assert any('unknown field' in error for error in errors)
    assert any('unknown capability' in error for error in errors)
    assert any('unknown risk' in error for error in errors)
