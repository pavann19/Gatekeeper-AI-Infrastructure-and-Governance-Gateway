# Security model

Gatekeeper treats prompts, provider output, tool arguments and transport identifiers as untrusted. Operators, provisioned INTERNAL credentials, registered Python handlers, policy/model files and the host filesystem are trusted. Detection provides risk signals, not proof of safety. See [THREAT_MODEL.md](THREAT_MODEL.md) for the threats being addressed.

## Authentication, authorization and tenants

`core/auth.py` resolves capability and tenant from verified API keys. Stores contain SHA-256 hashes; plaintext is disclosed once at issuance. Bearer and `X-API-Key` credentials are supported. Malformed/unknown credentials never grant elevated privilege. Required authentication denies anonymous callers; optional authentication permits anonymous GENERAL requests where the endpoint allows them. Sensitive endpoints enforce their own authentication/capability requirements.

GENERAL, ELEVATED and INTERNAL are capability tiers. INTERNAL has intentional cross-tenant administration and a permissive input policy; output and tool controls still apply. Protect these keys as operator credentials and coordinate revocation/reload across serving processes.

Tenant identity comes from admission. Suspended or unavailable tenant state denies admission. Missing explicitly configured tenant stores fail closed; an unconfigured optional store uses defaults. Activity, audit views and review access enforce tenant ownership with explicit INTERNAL access. Operators control tenant privacy overrides and policies.

## Rate limits, quotas and execution bounds

Without `REDIS_URL`, token buckets and daily token counters are process local. Configured Redis uses atomic server-side operations for shared accounting. Outage does not grant a fresh local budget. Quota admission and usage recording are separate stages, not a reservation covering every concurrent in-flight request.

HTTP bodies, schema fields, assessment/provider submissions and MCP sessions/queues/dispatch have bounds. Timed-out running threads retain capacity until completion. The monotonic budget does not cover all synchronous privacy/audit/review work or every nested detector/MCP stage. Custom handlers need their own effect and resource controls.

## Privacy, detection and cache

Regex and NER processing redact personal data for downstream use. Required NER unavailability denies assessment. Local learned classifiers can inspect the original prompt to avoid placeholder-induced changes to detection; this is not permission to send raw prompts to remote judges. Tenant privacy settings can intentionally change redaction coverage.

Current symbolic rules precede cache reuse. Threat risk and topicality are separate. Optional model loss can select a less capable fusion tier; readiness does not attest every possible tier. Ambiguous input paths whose required judge is unavailable fail closed rather than inventing a safe verdict.

Security cache keys bind tenant, raw/clean input and security identity: source, policies, model artifacts, settings and critical runtime versions. Only HIGH outcomes are reusable and remain monotonic within that identity while retained. TTL, eviction or authorized flush can remove records. Configured exact-cache outages fail closed. Redis does not share the local fuzzy index or circuit breakers.

## Policy and review

The live `policy_rules.json` maps tenant, capability and risk to actions. Missing usable default policy denies decisions. Requests retain immutable snapshots/digests. Deployment validates a prepared candidate before atomic replacement; rollback rejects traversal and tampered snapshots. Operators must coordinate reload across processes.

RESTRICT stops provider forwarding and denies restricted tools. Assessment clients must enforce returned decisions themselves. REVIEW is not execution permission. Records expire, reject conflicting resolution and deny writes on corruption. ELEVATED and INTERNAL reviewers are constrained by tenant ownership.

Tool approval binds principal, tenant, original request ID, capability, action/schema/arguments, policy digest, security version and decision context. Atomic one-use consume reserves approval before effects. Changed actions or approvals supplied to newly permissive paths are denied. Prompt/provider reviews have no execution continuation endpoint.

Review JSON uses atomic replacement and bounded advisory locks on a local filesystem. Network-filesystem/distributed storage behavior is not certified. A crash after approval reservation can leave uncertain execution; investigate before retrying an effect.

## Tools and SSRF

Registry entries declare capability, risk and argument schema. Shared execution checks admission/capability/schema/policy, requires HIGH review and persists intent before the trusted handler. Python handlers are not sandboxed.

The optional HTTP tool validates every resolved address, rejects unsafe destinations, connects to a selected validated address, preserves original Host/TLS verification, disables redirects/environment proxies and uses a child process for total network timeout. Allowlist configuration and host network access remain operator responsibilities.

## Providers and output

Operators configure providers. The gateway screens the user prompt before forwarding and applies output checks before returning content. Supplied system prompts are forwarded through a separate path without user-prompt privacy/assessment: accept only trusted application configuration and do not include secrets.

Output checks block detected secrets and verbatim system-prompt leakage when a reference is supplied, redact PII, and require explicit SAFE toxicity acceptance. Unavailable, invalid, failed or timed-out required judging blocks output. Optional grounding is disabled by default and is not general factuality verification. Substring leakage detection does not cover every paraphrase or transformation.

## Audit

JSONL events record sanitized assessment/output/tool fields and policy identity. SQLite supplies queryable persistence. With `AUDIT_SQLITE_ENABLED`, SQLite write failures are authoritative when `AUDIT_READ_FROM_SQLITE` is enabled; otherwise the logger reports a best-effort mirror failure. Verify the actual settings and storage permissions. Tool intent must persist before effects. Completion/error audit failure is reported but cannot roll back an already completed effect.

Host files are mutable, not a tamper-proof ledger. JSONL append locking is process local. Backup, access control, capacity, retention and multi-writer acceptance are deployment responsibilities. Restrict access to operational identifiers and redacted text in records.

## MCP

HTTP/SSE and stateless HTTP MCP require authenticated admission, suspension and rate checks. SSE sessions bind credentials and principal context. Body size, session lifetime, queue depth and pending dispatch are bounded. Shared tool capability/review/audit controls apply.

Stdio uses a fixed capability and tenant selected by its trusted launcher, without per-caller authentication. Restrict process access and launch with least privilege. MCP is a tool transport, not a prompt-assessment endpoint or protocol-compliance certification.

## Limits and vulnerability reporting

This repository does not prove production operation, arbitrary multi-replica deployment, exhaustive adversarial resistance or complete model-profile readiness. Read [deployment](DEPLOYMENT_RUNBOOK.md) and [evaluation](EVALUATION.md) before interpreting tests or measurements.

Use the repository's private vulnerability reporting channel if GitHub displays one. Otherwise contact a maintainer through an available private channel; no private address or reporting service is promised here. Do not post credentials, raw user data or exploit details in a public issue. Include the affected revision, configuration, impact and a minimal reproduction.

