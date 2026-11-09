# Security model

Gatekeeper treats prompts, provider output, tool arguments and transport identifiers as untrusted. Operators, provisioned INTERNAL credentials, registered Python handlers, policy/model files and the host filesystem are trusted. Detection provides risk signals, not proof of safety. See [THREAT_MODEL.md](THREAT_MODEL.md) for the threats being addressed.

## Authentication, authorization and tenants

`core/auth.py` resolves capability and tenant from verified API keys. Stores contain SHA-256 hashes; plaintext is disclosed once at issuance. Bearer and `X-API-Key` credentials are supported. Malformed/unknown credentials never grant elevated privilege. Required authentication denies anonymous callers; optional authentication permits anonymous GENERAL requests where the endpoint allows them. Sensitive endpoints enforce their own authentication/capability requirements.

GENERAL, ELEVATED and INTERNAL are capability tiers. INTERNAL has intentional cross-tenant administration and a permissive input policy; output and tool controls still apply. Protect these keys as operator credentials and coordinate revocation/reload across serving processes.

Tenant identity comes from admission. Suspended, unavailable and unprovisioned tenants are denied. Only the literal `default` tenant can use an absent, unconfigured optional store; a configured store must provision every admitted tenant. Activity, audit views and review access enforce tenant ownership with explicit INTERNAL access. Operators control tenant privacy overrides and policies.

## Rate limits, quotas and execution bounds

Without `REDIS_URL`, token buckets and daily token counters are process local. Configured Redis uses atomic server-side operations for shared accounting. Outage does not grant a fresh local budget. Quota admission and usage recording are separate stages, not a reservation covering every concurrent in-flight request.

HTTP bodies, schema fields, assessment/provider/state-I/O submissions and MCP sessions/queues/dispatch have bounds. Async assessment and gateway routes offload privacy and state operations to bounded workers and await audit acknowledgements. MCP admission is also offloaded and bounded. Timed-out running threads retain capacity until completion; Python cannot forcibly stop them. Custom handlers and nested model operations need their own effect and resource controls. Completed provider responses carry `token_accounting_degraded` if subsequent quota recording fails; this avoids a retryable accounting error but is not reserve/settle accounting or provider idempotency.

## Privacy, detection and cache

Regex and NER processing redact personal data for downstream use. Required NER unavailability denies assessment. Local learned classifiers can inspect the original prompt to avoid placeholder-induced changes to detection; this is not permission to send raw prompts to remote judges. Tenant privacy settings can intentionally change redaction coverage.

Current symbolic rules precede cache reuse. Threat risk and topicality are separate. Optional model loss can select a less capable fusion tier; readiness does not attest every possible tier. Ambiguous input paths whose required judge is unavailable fail closed rather than inventing a safe verdict.

Security cache keys bind tenant, raw/clean input and security identity: source, policies, model artifacts, settings and critical runtime versions. Only HIGH outcomes are reusable and remain monotonic within that identity while retained. Exact persistence contains decision metadata only in a new local filename/Redis namespace; fuzzy vectors remain in memory and legacy fuzzy snapshots are not loaded. Historical caches require a separately reviewed retention/purge procedure before deployment. TTL, eviction or authorized flush can remove records. Configured exact-cache outages fail closed. Redis can share circuit-breaker state; breaker outages use their documented local fallback. Fuzzy indexes remain process-local.

## Policy and review

The live `policy_rules.json` maps tenant, capability and risk to actions. Missing usable default policy denies decisions. Requests retain immutable snapshots/digests. Deployment validates a prepared candidate before atomic replacement; rollback rejects traversal and tampered snapshots. Operators must coordinate reload across processes.

RESTRICT stops provider forwarding and denies restricted tools. Assessment clients must enforce returned decisions themselves. REVIEW is not execution permission. INTERNAL credentials resolve reviews; HIGH approvals require a recorded maker and a different reviewer key ID. This is credential separation, not proof of independent human operators. Resolution intent is audited before approval persistence, with completion afterward. Pending views omit expired records and support bounded pagination; explicit expiry maintenance retains their evidence. Corruption disables mutations. In production mode, live HTTP policy deploy/rollback is disabled pending a reviewed signed-release workflow.

Tool approval binds principal, tenant, original request ID, capability, action/schema/arguments, policy digest, security version and decision context. Atomic one-use consume reserves approval before effects. Changed actions or approvals supplied to newly permissive paths are denied. Prompt/provider reviews have no execution continuation endpoint.

Review JSON uses atomic replacement and bounded advisory locks on a local filesystem. Network-filesystem/distributed storage behavior is not certified. A crash after approval reservation can leave uncertain execution; investigate before retrying an effect.

## Tools and SSRF

Registry entries declare capability, risk and argument schema. Shared execution checks admission/capability/schema/policy, requires HIGH review and persists intent before the trusted handler. Python handlers are not sandboxed.

The optional HTTP tool validates every resolved address, rejects unsafe destinations, connects to a selected validated address, preserves original Host/TLS verification, disables redirects/environment proxies and uses a child process for total network timeout. Allowlist configuration and host network access remain operator responsibilities.

## Providers and output

Operators configure providers. The gateway screens the user prompt before forwarding and applies output checks before returning content. Only authenticated INTERNAL applications may supply a gateway system prompt. That trusted configuration is not itself screened as user input and must contain no secrets.

Output checks block detected secrets and verbatim system-prompt leakage when a reference is supplied, redact PII, and require explicit SAFE toxicity acceptance. Privacy metadata returns labels without matched PII values. A suffix automaton bounds verbatim leak detection to linear work; tool output release has a serialized UTF-8 byte limit. Neither feature controls memory allocation inside a trusted custom handler. Unavailable, invalid, failed or timed-out required judging blocks output. Optional grounding is disabled by default and is not general factuality verification. Substring leakage detection does not cover every paraphrase or transformation.

## Audit

`STATE_BACKEND=postgres` selects PostgreSQL as the audit and review authority,
with no JSONL/SQLite fallback. Explicit, checksum-sealed SQL migrations run
outside the request path. Review resolution and one-use consume commit with
their audit event in the same transaction. Audit rows carry a per-tenant
SHA-256 predecessor chain; update/delete/truncate triggers protect ordinary SQL
mutation. These controls do not make a privileged database owner tamper-proof.
External archive/anchors and production retention remain required.

Production mode requires a TLS-verified PostgreSQL URL and startup/readiness
reject migration-owner, superuser and privilege-escalation roles. The runtime
role must have only the necessary audit insert/read, chain-head and review
permissions; it cannot create schema objects or mutate audit/schema history.
The API and standalone MCP both check this state preflight at startup. Review
and audit storage is shared across connections; full replica/host failover is
separate operational acceptance.

JSONL events record assessment/output/tool metadata, UTC timestamps, event IDs and policy identity. SQLite preserves event IDs and policy digests on its read path; schema reconciliation is cached per database-file identity. With `AUDIT_SQLITE_ENABLED`, SQLite write failures are authoritative when `AUDIT_READ_FROM_SQLITE` is enabled; otherwise the logger reports a best-effort mirror failure. Authoritative audit failure always fails closed; the legacy `AUDIT_WRITE_FAILS_CLOSED` flag cannot opt out. Tool intent must persist before effects. Completion/error audit failure is reported but cannot roll back an already completed effect.

Host files are mutable, not a tamper-proof ledger. JSONL append and schema initialization use bounded local-filesystem cross-process locks; this is not distributed-store or HA acceptance. Backup, access control, capacity, retention and multi-writer acceptance are deployment responsibilities. Restrict access to operational identifiers and redacted text in records.

## MCP

HTTP/SSE and stateless HTTP MCP require authenticated admission, suspension and rate checks. SSE sessions bind credentials and principal context. Body size, session lifetime, queue depth and pending dispatch are bounded. Shared tool capability/review/audit controls apply.

Stdio uses a fixed capability and tenant selected by its trusted launcher, without per-caller authentication. Restrict process access and launch with least privilege. MCP is a tool transport, not a prompt-assessment endpoint or protocol-compliance certification.

## Limits and vulnerability reporting

This repository does not prove production operation, arbitrary multi-replica deployment, exhaustive adversarial resistance or complete model-profile readiness. Read [deployment](DEPLOYMENT_RUNBOOK.md) and [evaluation](EVALUATION.md) before interpreting tests or measurements.

Use the repository's private vulnerability reporting channel if GitHub displays one. Otherwise contact a maintainer through an available private channel; no private address or reporting service is promised here. Do not post credentials, raw user data or exploit details in a public issue. Include the affected revision, configuration, impact and a minimal reproduction.

