# Isolated pilot candidate: preflight and remaining gates

`deploy/compose.pilot.yml` is a standalone, unexecuted configuration candidate.
It exposes no host ports and depends on explicitly named, pre-existing isolated
network/model resources and a separate PostgreSQL service. It does not provision TLS, Redis, a judge,
monitoring or backups. Do not treat it as a completed production deployment.

Before any deployment, an operator must verify a digest-pinned image, a complete
offline model cache including tokenizer/config files, weight hashes, the judge's
actual model identity, provisioned tenants and hashed grants, database and
read-only secret/model access from the image's non-root user, and a TLS ingress that admits traffic
only when `/ready` returns 200. Docker's healthcheck uses `/live`; `/startup`
separately reports warm-up completion. Lazy loading leaves readiness unavailable
until dependencies really load. A timed-out warm-up thread cannot be forcibly
cancelled by Python; restart/recovery must be exercised under explicit budgets.

Production settings require PostgreSQL with `sslmode=verify-full` and reject optional authentication, wildcard/non-HTTPS CORS,
unauthenticated metrics, unverified detector hashes, disabled warm-up and demo
tools. This validator does not certify the deployment, model quality or operator
identity. Transactional audit/review state and a synthetic restore have local
acceptance evidence. Model/runtime locks, TLS/authenticated Redis acceptance,
signed dual-control policy releases, production backup/restore,
multi-replica failover and external security review remain required.

Initialize a dedicated empty database using a separate trusted migration owner:
set `STATE_BACKEND=postgres`, supply that owner's `DATABASE_URL`, and run
`python -m scripts.manage_postgres_state migrate --confirm-empty-or-known-database`.
The migration refuses unrelated tables and changed applied checksums. Never
run this command against existing user data as a fixture. Production runtime
credentials must be a different, restricted role; startup rejects elevated
roles, schema creation and audit/schema deletion or mutation privileges.
Provision passwords and the PostgreSQL CA through the approved secret system;
the profile mounts the CA at `/run/secrets/postgres_ca`. Include that path as
`sslrootcert` in the TLS-verified runtime DSN.

For an already provisioned restricted `gk_runtime` role in the dedicated DB,
the required grants are:

```sql
GRANT USAGE ON SCHEMA public TO gk_runtime;
GRANT SELECT ON gk_schema_version TO gk_runtime;
GRANT SELECT, INSERT ON gk_audit_events TO gk_runtime;
GRANT SELECT, INSERT, UPDATE ON gk_audit_heads, gk_reviews TO gk_runtime;
GRANT USAGE ON SEQUENCE gk_audit_events_sequence_seq TO gk_runtime;
```

It must have no inherited CREATE/schema, CREATEROLE, CREATEDB, replication,
bypass-RLS or superuser privileges. Validate with
`python -m scripts.manage_postgres_state check` under runtime credentials.
This check does not install or modify the schema. Monthly partitions, archival,
WORM export, production restore/failover and release/job schemas remain open.

For existing installations, retain historical evidence in its original location.
Do not package the workspace's caches, keys, audit or review state into an image.
The image build already uses a source allowlist in `.dockerignore`. The new
cache namespace is `gatekeeper:cache:v3:`; exact writes contain decision metadata
only, and fuzzy vectors stay in memory. Legacy local JSON snapshots and Redis
keys can still contain historical prompt content. Their authorized archival,
retention decision and scoped purge must be verified before shared deployment;
this implementation task does not delete original user data or shared keys.

To reproduce local regression evidence, run
`python -B scripts/validation/run_isolated_tests.py --run _evidence/UNIQUE_RUN tests`.
The output directory must be new. The runner copies source, uses empty local
state/model caches, records hashes and preserves failures. Model integration
tests require an explicit separate `-m integration` run with identified offline
weights; default tests do not certify real model behavior.

For the real SQL contracts on Windows, use
`scripts/validation/run_postgres_contracts.ps1 -Run _evidence/UNIQUE_PG_RUN -Image EXISTING_LOCAL_IMAGE_ID`.
It uses a new labelled loopback fixture with tmpfs data, bounded resources and
ownership-checked cleanup. It never pulls an image or mounts existing volumes.
The integration suite tests chain concurrency, one-use effects, atomic
review/audit rollback, audit failure, restricted roles, expiry and capacity.
The harness restores a quiescent synthetic dump into a fresh database and
compares retained audit/review fingerprints; it is not production RTO/RPO.
