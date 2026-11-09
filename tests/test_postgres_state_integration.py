"""Real SQL contracts, only against the runner's owned loopback synthetic DB."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import os
from unittest.mock import Mock
from pathlib import Path
from uuid import uuid4

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope='module')
def store():
    psycopg2 = pytest.importorskip('psycopg2')
    from psycopg2.extensions import parse_dsn
    from core.postgres_state import PostgresStore
    dsn = os.getenv('GATEKEEPER_TEST_PG_URL')
    if not dsn:
        pytest.skip('Owned PostgreSQL fixture not configured')
    parsed = parse_dsn(dsn)
    assert parsed.get('host') == '127.0.0.1' and parsed.get('dbname', '').startswith('gk_synthetic_')
    assert psycopg2
    result = PostgresStore(dsn)
    result.migrate()
    assert result.ready()
    return result


def test_shared_audit_writers_have_a_valid_chain_and_immutable_rows(store):
    from core.postgres_state import GENESIS, PostgresStore, PostgresStateUnavailable, canonical
    tenant = 'synthetic-' + uuid4().hex

    def write(index):
        independent = PostgresStore(os.environ['GATEKEEPER_TEST_PG_URL'])
        independent.write_audit({'event_id': uuid4().hex, 'tenant': tenant,
                                 'request_id': 'synthetic-' + str(index), 'event_type': 'synthetic_probe',
                                 'timestamp': datetime.now(timezone.utc).isoformat(), 'policy_digest': 'a' * 64})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(40)))
    with store.transaction() as cursor:
        cursor.execute('SELECT prev_hash,event_hash,payload FROM gk_audit_events WHERE tenant=%s ORDER BY sequence', (tenant,))
        rows = cursor.fetchall()
    assert len(rows) == 40
    previous = GENESIS
    for prev_hash, event_hash, payload in rows:
        assert prev_hash == previous
        assert event_hash == hashlib.sha256((previous + '\n' + canonical(payload)).encode()).hexdigest()
        previous = event_hash
    with pytest.raises(PostgresStateUnavailable):
        with store.transaction() as cursor:
            cursor.execute('DELETE FROM gk_audit_events WHERE tenant=%s', (tenant,))
    with pytest.raises(PostgresStateUnavailable):
        with store.transaction() as cursor:
            cursor.execute('TRUNCATE gk_audit_events')
    assert len(store.recent(tenant=tenant)['events']) == 40


def test_high_tool_approval_executes_at_most_once_across_shared_connections(store, monkeypatch):
    from core.approval import approval_context
    from core.auth import Principal
    from core.postgres_state import PostgresReviewQueue
    from core.tools import ToolRegistry, ToolSpec, decide_tool_call, execute_tool
    import core.review_queue
    queue = PostgresReviewQueue(store)
    monkeypatch.setattr(core.review_queue, '_queue', queue)
    registry = ToolRegistry()
    spec = ToolSpec('synthetic.delete', 'Synthetic effect', risk_level='HIGH')
    calls = []
    registry.register(spec, handler=lambda: calls.append('effect'))
    principal = Principal(capability='GENERAL', tenant='default', key_id='maker', authenticated=True)
    decision = decide_tool_call('GENERAL', spec, {})
    context = approval_context(principal, 'original', spec.name, {'arguments': {}, 'spec': asdict(spec)}, decision['reason'])
    record = queue.enqueue(decision['reason'], 'GENERAL', 'HIGH', 'default', 'digest', 'original', context)
    with pytest.raises(ValueError):
        queue.resolve(record.review_id, 'APPROVED', 'maker')
    queue.resolve(record.review_id, 'APPROVED', 'checker')

    def execute(_):
        return execute_tool('GENERAL', spec.name, {}, registry=registry, tenant='default', request_id='original',
                            principal_id='maker', approval_id=record.review_id)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(execute, range(16)))
    assert sum(result['decision'] == 'ALLOW' for result in results) == 1
    assert calls == ['effect']
    assert queue.get(record.review_id)['status'] == 'CONSUMED'


def test_review_resolution_and_audit_rollback_together_on_database_failure(store):
    from core.postgres_state import PostgresReviewQueue, PostgresStateUnavailable
    queue = PostgresReviewQueue(store)
    record = queue.enqueue('synthetic', 'GENERAL', 'HIGH', 'atomic-review', 'digest', 'request', {'principal': 'maker'})
    with store.transaction() as cursor:
        cursor.execute("CREATE FUNCTION gk_test_reject_review() RETURNS trigger LANGUAGE plpgsql AS $$ "
                       "BEGIN IF NEW.event_type='review_resolution' THEN RAISE EXCEPTION 'synthetic failure'; "
                       "END IF; RETURN NEW; END; $$")
        cursor.execute('CREATE TRIGGER gk_test_reject_review BEFORE INSERT ON gk_audit_events '
                       'FOR EACH ROW EXECUTE FUNCTION gk_test_reject_review()')
    try:
        with pytest.raises(PostgresStateUnavailable):
            queue.resolve(record.review_id, 'APPROVED', 'checker')
        assert queue.get(record.review_id)['status'] == 'PENDING'
        assert store.recent(tenant='atomic-review')['events'] == []
    finally:
        with store.transaction() as cursor:
            cursor.execute('DROP TRIGGER gk_test_reject_review ON gk_audit_events')
            cursor.execute('DROP FUNCTION gk_test_reject_review()')
    assert queue.resolve(record.review_id, 'APPROVED', 'checker')['status'] == 'APPROVED'
    assert len(store.recent(tenant='atomic-review')['events']) == 1


def test_postgres_audit_failure_is_503_without_local_fallback(store, monkeypatch):
    from api import main
    from fastapi.testclient import TestClient
    from core.config import settings
    inference = Mock(return_value=('LOW', {'source': 'synthetic'}))
    monkeypatch.setattr(main, 'assess_risk', inference)
    monkeypatch.setattr(main, '_redact_or_fail', lambda text, tenant: (text, {'items': []}))
    with store.transaction() as cursor:
        cursor.execute("CREATE FUNCTION gk_test_reject_input() RETURNS trigger LANGUAGE plpgsql AS $$ "
                       "BEGIN IF NEW.event_type='input_assessment' THEN RAISE EXCEPTION 'synthetic failure'; "
                       "END IF; RETURN NEW; END; $$")
        cursor.execute('CREATE TRIGGER gk_test_reject_input BEFORE INSERT ON gk_audit_events '
                       'FOR EACH ROW EXECUTE FUNCTION gk_test_reject_input()')
    try:
        response = TestClient(main.app).post('/api/v1/assess', json={'prompt': 'synthetic hello'},
                                             headers={'X-Request-ID': 'synthetic-failed-audit'})
        assert response.status_code == 503
        assert 'audit' in response.json()['detail'].lower()
        inference.assert_called_once()
        assert response.headers['Retry-After'] == '5'
        assert store.recent(request_id='synthetic-failed-audit')['events'] == []
        assert not Path(settings.AUDIT_LOG_PATH).exists()
        assert not Path(settings.AUDIT_DB_PATH).exists()
    finally:
        with store.transaction() as cursor:
            cursor.execute('DROP TRIGGER gk_test_reject_input ON gk_audit_events')
            cursor.execute('DROP FUNCTION gk_test_reject_input()')


def test_schema_checksum_is_sealed(store, tmp_path, monkeypatch):
    import core.postgres_state as state
    path = tmp_path / 'altered.sql'
    path.write_text(state.MIGRATION.read_text() + '\n-- synthetic change\n')
    monkeypatch.setattr(state, 'MIGRATION', path)
    with pytest.raises(ValueError, match='checksum'):
        store.migrate()


def test_production_requires_a_restricted_runtime_role(store, monkeypatch):
    from psycopg2 import sql
    from psycopg2.extensions import make_dsn
    from core.config import settings
    from core.postgres_state import PostgresStore
    role = 'gk_runtime_' + uuid4().hex[:10]
    with store.transaction() as cursor:
        cursor.execute(sql.SQL('CREATE ROLE {} LOGIN PASSWORD {}').format(
            sql.Identifier(role), sql.Literal('synthetic-runtime-only')))
        cursor.execute(sql.SQL('GRANT USAGE ON SCHEMA public TO {}').format(sql.Identifier(role)))
        for tables, privileges in [(['gk_audit_events', 'gk_schema_version'], 'SELECT'),
                                   (['gk_audit_events'], 'INSERT'),
                                   (['gk_audit_heads', 'gk_reviews'], 'SELECT,INSERT,UPDATE')]:
            for table in tables:
                cursor.execute(sql.SQL('GRANT ' + privileges + ' ON {} TO {}').format(
                    sql.Identifier(table), sql.Identifier(role)))
        cursor.execute(sql.SQL('GRANT USAGE ON SEQUENCE gk_audit_events_sequence_seq TO {}').format(sql.Identifier(role)))
    runtime = PostgresStore(make_dsn(os.environ['GATEKEEPER_TEST_PG_URL'], user=role, password='synthetic-runtime-only'))
    monkeypatch.setattr(settings, 'ENVIRONMENT', 'production')
    assert store.ready() is False  # Fixture migration owner must not be used by production.
    assert runtime.ready() is True
    runtime.write_audit({'event_type': 'synthetic_probe', 'tenant': 'restricted-runtime', 'request_id': 'restricted'})
    assert len(runtime.recent(tenant='restricted-runtime')['events']) == 1
    with store.transaction() as cursor:
        cursor.execute(sql.SQL('ALTER ROLE {} CREATEROLE').format(sql.Identifier(role)))
    assert runtime.ready() is False
    with store.transaction() as cursor:
        cursor.execute(sql.SQL('ALTER ROLE {} NOCREATEROLE').format(sql.Identifier(role)))
    assert runtime.ready() is True


def test_shared_review_expiry_and_capacity_are_enforced(store, monkeypatch):
    from core.config import settings
    from core.postgres_state import PostgresReviewQueue, PostgresStateUnavailable
    queue = PostgresReviewQueue(store)
    record = queue.enqueue('synthetic', 'GENERAL', 'HIGH', 'expiry-test', 'digest', 'request', {'principal': 'maker'})
    with store.transaction() as cursor:
        cursor.execute("UPDATE gk_reviews SET expires_at=clock_timestamp()-interval '1 second' WHERE review_id=%s",
                       (record.review_id,))
    with pytest.raises(ValueError, match='expired'):
        queue.resolve(record.review_id, 'APPROVED', 'checker')
    assert queue.list_pending(tenant='expiry-test') == []
    assert queue.expire_pending() >= 1
    assert queue.get(record.review_id)['status'] == 'EXPIRED'
    monkeypatch.setattr(settings, 'REVIEW_MAX_RECORDS', 0)
    with pytest.raises(PostgresStateUnavailable):
        queue.enqueue('full', 'GENERAL', 'MEDIUM', 'capacity', 'digest', 'request')
    assert queue.get(record.review_id)['status'] == 'EXPIRED'


def test_review_timestamps_are_utc_even_with_a_different_database_timezone(store):
    from psycopg2 import sql
    from psycopg2.extensions import parse_dsn
    from core.postgres_state import PostgresReviewQueue
    database = parse_dsn(os.environ['GATEKEEPER_TEST_PG_URL'])['dbname']
    with store.transaction() as cursor:
        cursor.execute(sql.SQL("ALTER DATABASE {} SET timezone='Asia/Kolkata'").format(sql.Identifier(database)))
    try:
        record = PostgresReviewQueue(store).enqueue('synthetic', 'GENERAL', 'MEDIUM', 'utc-test', 'digest', 'request')
        assert record.created_at.endswith('+00:00')
        assert record.expires_at.endswith('+00:00')
    finally:
        with store.transaction() as cursor:
            cursor.execute(sql.SQL('ALTER DATABASE {} RESET timezone').format(sql.Identifier(database)))
