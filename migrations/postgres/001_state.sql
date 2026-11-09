CREATE TABLE IF NOT EXISTS gk_audit_heads (
    tenant TEXT PRIMARY KEY,
    event_hash CHAR(64) NOT NULL
);
CREATE TABLE IF NOT EXISTS gk_audit_events (
    sequence BIGSERIAL UNIQUE NOT NULL,
    event_id UUID PRIMARY KEY,
    tenant TEXT NOT NULL,
    request_id TEXT,
    event_type TEXT NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL,
    policy_digest TEXT,
    prev_hash CHAR(64) NOT NULL,
    event_hash CHAR(64) NOT NULL,
    payload JSONB NOT NULL CHECK (octet_length(payload::text) <= 1048576)
);
CREATE INDEX IF NOT EXISTS gk_audit_tenant_sequence ON gk_audit_events(tenant, sequence DESC);
CREATE INDEX IF NOT EXISTS gk_audit_request ON gk_audit_events(request_id, sequence);
CREATE TABLE IF NOT EXISTS gk_reviews (
    review_id UUID PRIMARY KEY,
    tenant TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('PENDING','APPROVED','REJECTED','CONSUMED','EXPIRED')),
    expires_at TIMESTAMPTZ NOT NULL,
    payload JSONB NOT NULL CHECK (octet_length(payload::text) <= 65536)
);
CREATE INDEX IF NOT EXISTS gk_reviews_pending ON gk_reviews(tenant, status, expires_at, review_id);
CREATE OR REPLACE FUNCTION gk_reject_audit_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'Audit events are append-only';
END;
$$;
DROP TRIGGER IF EXISTS gk_audit_no_mutation ON gk_audit_events;
CREATE TRIGGER gk_audit_no_mutation BEFORE UPDATE OR DELETE ON gk_audit_events
    FOR EACH ROW EXECUTE FUNCTION gk_reject_audit_mutation();
DROP TRIGGER IF EXISTS gk_audit_no_truncate ON gk_audit_events;
CREATE TRIGGER gk_audit_no_truncate BEFORE TRUNCATE ON gk_audit_events
    FOR EACH STATEMENT EXECUTE FUNCTION gk_reject_audit_mutation();
