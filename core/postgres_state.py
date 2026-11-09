"""Transactional shared audit/review state. Schema installation is explicit.

The runtime never falls back to JSON/SQLite when this authority is configured.
Use a separate migration owner and a restricted application role in deployment.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from core.config import settings

MIGRATION = Path(__file__).resolve().parents[1] / 'migrations/postgres/001_state.sql'
GENESIS = '0' * 64


class PostgresStateUnavailable(RuntimeError):
    """Required state authority did not acknowledge the transaction."""


def canonical(record):
    return json.dumps(record, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


class PostgresStore:
    def __init__(self, dsn):
        self._dsn = dsn

    @contextmanager
    def transaction(self):
        connection = None
        try:
            import psycopg2
            connection = psycopg2.connect(self._dsn, connect_timeout=3,
                                          options='-c statement_timeout=5000 -c lock_timeout=2000')
            with connection:
                with connection.cursor() as cursor:
                    yield cursor
        except (ValueError, KeyError):
            raise
        except Exception:
            # Driver errors can carry connection details; never expose them.
            raise PostgresStateUnavailable('Required PostgreSQL state unavailable') from None
        finally:
            if connection is not None:
                connection.close()

    def migrate(self):
        """Install/check a sealed migration in an empty or known Gatekeeper DB."""
        text = MIGRATION.read_text(encoding='utf-8')
        checksum = hashlib.sha256(text.encode()).hexdigest()
        with self.transaction() as cursor:
            cursor.execute('SELECT pg_advisory_xact_lock(1196117331)')
            cursor.execute("SELECT tablename FROM pg_tables WHERE schemaname='public'")
            tables = {row[0] for row in cursor.fetchall()}
            allowed = {'gk_schema_version', 'gk_audit_heads', 'gk_audit_events', 'gk_reviews'}
            if tables - allowed:
                raise ValueError('Refusing migration of an unrelated database')
            cursor.execute('CREATE TABLE IF NOT EXISTS gk_schema_version '
                           '(version INTEGER PRIMARY KEY, checksum TEXT NOT NULL)')
            cursor.execute('SELECT checksum FROM gk_schema_version WHERE version=1')
            old = cursor.fetchone()
            if old:
                if old[0] != checksum:
                    raise ValueError('Applied migration checksum differs; create a new migration')
                return
            cursor.execute(text)
            cursor.execute('INSERT INTO gk_schema_version(version,checksum) VALUES(1,%s)', (checksum,))

    def ready(self):
        try:
            expected = hashlib.sha256(MIGRATION.read_text(encoding='utf-8').encode()).hexdigest()
            with self.transaction() as cursor:
                cursor.execute('SELECT checksum FROM gk_schema_version WHERE version=1')
                row = cursor.fetchone()
                if not row or row[0] != expected:
                    return False
                if settings.ENVIRONMENT == 'production':
                    cursor.execute("SELECT rolsuper OR rolcreaterole OR rolcreatedb OR rolreplication OR rolbypassrls "
                                   "FROM pg_roles WHERE rolname=current_user")
                    if cursor.fetchone()[0]:
                        return False
                    cursor.execute("SELECT has_table_privilege('gk_audit_events','UPDATE,DELETE,TRUNCATE'), "
                                   "has_table_privilege('gk_schema_version','INSERT,UPDATE,DELETE,TRUNCATE'), "
                                   "has_schema_privilege('public','CREATE'), "
                                   "has_table_privilege('gk_audit_events','SELECT'), "
                                   "has_table_privilege('gk_audit_events','INSERT'), "
                                   "has_table_privilege('gk_reviews','DELETE,TRUNCATE') OR "
                                   "has_table_privilege('gk_audit_heads','DELETE,TRUNCATE'), "
                                   "has_table_privilege('gk_reviews','SELECT') AND "
                                   "has_table_privilege('gk_reviews','INSERT') AND "
                                   "has_table_privilege('gk_reviews','UPDATE') AND "
                                   "has_table_privilege('gk_audit_heads','SELECT') AND "
                                   "has_table_privilege('gk_audit_heads','INSERT') AND "
                                   "has_table_privilege('gk_audit_heads','UPDATE') AND "
                                   "has_sequence_privilege('gk_audit_events_sequence_seq','USAGE')")
                    mutation, schema_write, create_objects, read_audit, write_audit, deletion, state_access = cursor.fetchone()
                    return not (mutation or schema_write or create_objects or deletion) and read_audit and write_audit and state_access
                return True
        except (PostgresStateUnavailable, OSError):
            return False

    def append_audit(self, cursor, record):
        """The chain head lock serializes concurrent writers for one tenant."""
        record = deepcopy(record)
        record.setdefault('event_id', uuid4().hex)
        record.setdefault('timestamp', datetime.now(timezone.utc).isoformat())
        record.setdefault('tenant', 'unset')
        tenant = record['tenant']
        cursor.execute('INSERT INTO gk_audit_heads(tenant,event_hash) VALUES(%s,%s) ON CONFLICT DO NOTHING',
                       (tenant, GENESIS))
        cursor.execute('SELECT event_hash FROM gk_audit_heads WHERE tenant=%s FOR UPDATE', (tenant,))
        previous = cursor.fetchone()[0]
        payload = canonical(record)
        digest = hashlib.sha256((previous + '\n' + payload).encode()).hexdigest()
        cursor.execute('INSERT INTO gk_audit_events '
                       '(event_id,tenant,request_id,event_type,occurred_at,policy_digest,prev_hash,event_hash,payload) '
                       'VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)',
                       (record['event_id'], tenant, record.get('request_id'), record['event_type'],
                        record['timestamp'], record.get('policy_digest'), previous, digest, payload))
        cursor.execute('UPDATE gk_audit_heads SET event_hash=%s WHERE tenant=%s', (digest, tenant))

    def write_audit(self, record):
        with self.transaction() as cursor:
            self.append_audit(cursor, record)

    def recent(self, limit=50, tenant=None, event_types=None, request_id=None):
        limit = max(1, min(int(limit), 200))
        conditions, parameters = [], []
        if tenant is not None:
            conditions.append('tenant=%s')
            parameters.append(tenant)
        if event_types:
            conditions.append('event_type=ANY(%s)')
            parameters.append(list(event_types))
        if request_id is not None:
            conditions.append('request_id=%s')
            parameters.append(request_id)
        sql = 'SELECT payload FROM gk_audit_events'
        if conditions:
            sql += ' WHERE ' + ' AND '.join(conditions)
        sql += ' ORDER BY sequence ' + ('ASC' if request_id is not None else 'DESC') + ' LIMIT %s'
        parameters.append(limit + 1)
        with self.transaction() as cursor:
            cursor.execute(sql, parameters)
            rows = [row[0] for row in cursor.fetchall()]
        return {'events': rows[:limit], 'scan_truncated': len(rows) > limit}


@lru_cache(maxsize=4)
def _store_for(dsn):
    return PostgresStore(dsn)


def get_postgres_store():
    if settings.STATE_BACKEND != 'postgres' or not settings.DATABASE_URL:
        raise PostgresStateUnavailable('PostgreSQL state is not configured')
    return _store_for(settings.DATABASE_URL)


class PostgresReviewQueue:
    """Approval state and its audit event commit in the same DB transaction."""
    def __init__(self, store=None):
        self.store = store or get_postgres_store()

    def enqueue(self, reason, capability, risk, tenant, prompt_hash, request_id, context=None):
        from core.review_queue import ReviewRecord, ReviewCapacityError
        with self.store.transaction() as cursor:
            cursor.execute('SELECT pg_advisory_xact_lock(1196117333)')
            cursor.execute('SELECT count(*) FROM gk_reviews')
            if cursor.fetchone()[0] >= settings.REVIEW_MAX_RECORDS:
                raise ReviewCapacityError('Review store capacity exhausted')
            cursor.execute('SELECT clock_timestamp()')
            now = cursor.fetchone()[0].astimezone(timezone.utc)
            record = ReviewRecord(uuid4().hex, 'PENDING', reason, capability, risk, tenant, prompt_hash,
                                  request_id, now.isoformat(), expires_at=(now + timedelta(hours=1)).isoformat(),
                                  context=deepcopy(context or {}))
            from dataclasses import asdict
            payload = asdict(record)
            cursor.execute('INSERT INTO gk_reviews(review_id,tenant,status,expires_at,payload) '
                           'VALUES(%s,%s,%s,%s,%s::jsonb)',
                           (record.review_id, tenant, 'PENDING', record.expires_at, canonical(payload)))
        return record

    def get(self, review_id):
        if not _uuid(review_id):
            return None
        with self.store.transaction() as cursor:
            cursor.execute('SELECT payload FROM gk_reviews WHERE review_id=%s', (review_id,))
            row = cursor.fetchone()
            return row[0] if row else None

    def list_pending(self, limit=100, offset=0, tenant=None):
        sql = "SELECT payload FROM gk_reviews WHERE status='PENDING' AND expires_at>clock_timestamp()"
        parameters = []
        if tenant is not None:
            sql += ' AND tenant=%s'
            parameters.append(tenant)
        sql += ' ORDER BY expires_at,review_id LIMIT %s OFFSET %s'
        parameters.extend((max(1, min(limit, 200)), max(0, offset)))
        with self.store.transaction() as cursor:
            cursor.execute(sql, parameters)
            return [row[0] for row in cursor.fetchall()]

    def resolve(self, review_id, outcome, reviewer):
        if not _uuid(review_id):
            raise KeyError(review_id)
        from core.policy import policy_identity
        with self.store.transaction() as cursor:
            cursor.execute('SELECT payload,expires_at>clock_timestamp() FROM gk_reviews '
                           'WHERE review_id=%s FOR UPDATE', (review_id,))
            row = cursor.fetchone()
            if not row:
                raise KeyError(review_id)
            record, unexpired = row
            if outcome not in ('APPROVED', 'REJECTED') or not reviewer:
                raise ValueError('Valid outcome and reviewer identity required')
            if outcome == 'APPROVED' and record['risk'] == 'HIGH':
                maker = record.get('context', {}).get('principal')
                if not maker or maker == reviewer:
                    raise ValueError('HIGH reviews require an identified independent reviewer')
            if record['status'] == outcome or (record['status'] == 'CONSUMED' and outcome == 'APPROVED'):
                return record
            if record['status'] != 'PENDING' or (outcome == 'APPROVED' and not unexpired):
                raise ValueError('Review expired or is already resolved')
            record.update(status=outcome, reviewer=reviewer, resolved_at=datetime.now(timezone.utc).isoformat(),
                          final_decision='ALLOW' if outcome == 'APPROVED' else 'BLOCK')
            self.store.append_audit(cursor, {
                'event_id': uuid4().hex, 'event_type': 'review_resolution', 'tenant': record['tenant'],
                'request_id': record['request_id'], 'review_id': review_id, 'reviewer': reviewer,
                'outcome': outcome, 'phase': 'resolution_committed', 'policy_digest': policy_identity(),
                'timestamp': record['resolved_at'], 'prompt_hash': record['prompt_hash'],
            })
            cursor.execute('UPDATE gk_reviews SET status=%s,payload=%s::jsonb WHERE review_id=%s',
                           (outcome, canonical(record), review_id))
        return record

    def consume(self, review_id, context):
        required = {'request_id', 'tenant', 'principal', 'capability', 'action',
                    'policy_digest', 'security_version', 'decision_context'}
        if not _uuid(review_id) or set(context) != required or any(not value for value in context.values()):
            raise ValueError('Incomplete approval context')
        with self.store.transaction() as cursor:
            cursor.execute('SELECT payload,expires_at>clock_timestamp() FROM gk_reviews '
                           'WHERE review_id=%s FOR UPDATE', (review_id,))
            row = cursor.fetchone()
            if not row:
                raise KeyError(review_id)
            record, unexpired = row
            if record['status'] != 'APPROVED' or record.get('consumed_at') or not unexpired or record['context'] != context:
                raise ValueError('Approval unavailable or context changed')
            record.update(status='CONSUMED', consumed_at=datetime.now(timezone.utc).isoformat())
            self.store.append_audit(cursor, {
                'event_id': uuid4().hex, 'event_type': 'review_consumed', 'tenant': record['tenant'],
                'request_id': record['request_id'], 'review_id': review_id,
                'principal': context['principal'], 'policy_digest': context['policy_digest'],
                'security_version': context['security_version'], 'timestamp': record['consumed_at'],
            })
            cursor.execute("UPDATE gk_reviews SET status='CONSUMED',payload=%s::jsonb WHERE review_id=%s",
                           (canonical(record), review_id))
            return record

    def expire_pending(self):
        with self.store.transaction() as cursor:
            cursor.execute("UPDATE gk_reviews SET status='EXPIRED', payload=payload || "
                           "'{\"status\":\"EXPIRED\",\"final_decision\":\"BLOCK\"}'::jsonb "
                           "WHERE status='PENDING' AND expires_at<=clock_timestamp()")
            return cursor.rowcount


def _uuid(value):
    from uuid import UUID
    try:
        UUID(value)
        return True
    except (ValueError, TypeError, AttributeError):
        return False
