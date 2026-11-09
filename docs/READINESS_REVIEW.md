# Production-readiness remediation checkpoint

This tracks the supplied production-readiness review. It is implementation and
local verification evidence, not production acceptance. The quality, release and sustained-operation acceptance gates remain open.
Privacy and admission fixes precede further quality/evaluation work so that
failure tests exercise the hardened candidate.

## Critical findings

| Finding | Implemented locally | Still required before closing the finding |
|---|---|---|
| C1: persisted prompts/vectors | Exact persistence projects decision metadata only; new v3 Redis namespace and local filename; fuzzy vectors remain in memory; legacy fuzzy snapshots are not loaded. | Authorized legacy-cache retention/purge and credential rotation, persistent-store scan, cache access auditing and operational retention acceptance. Original local/shared state was preserved. |
| C2: blocking async paths | Privacy, auth/tenant/rate/quota state work, review enqueue/context construction and audit acknowledgements are offloaded to bounded pools; MCP admission is bounded/offloaded. SQLite schema DDL runs once per database-file identity. Slow-stage tests verify liveness continues responding. | Durable transactional outbox, batching, full nested-stage deadline/resource verification and measured end-to-end capacity. Offloading is not a throughput claim. |
| C3: durable audit/review | Explicit checksum-sealed PostgreSQL migration; per-tenant audit hash chain; append-only mutation/truncate guards; restricted runtime-role preflight; transactional review resolution/consumption plus audit; capacity/expiry/pagination and UTC timestamps. Eight real PostgreSQL contracts and a quiescent synthetic dump/restore passed. SQLite IDs/digests and bounded local locks remain available for lab use. | Retention/archive/WORM, independent integrity verification, production restore drills with measured RTO/RPO, policy/job schemas and multi-replica/failover acceptance. A database administrator remains trusted; the local topology is not a production substitute. |
| C4: unsafe deployment defaults | Explicit production settings validation; distinct startup/live/ready; bounded warm-up and five-second cached readiness; image healthcheck uses liveness. Standalone pilot profile has no published ports, offline model settings, non-root image user, read-only root, dropped capabilities, restart/resource/log limits and required external resource identities. | Hash-locked build, image build/scan, actual TLS ingress, secret handling, provisioned offline models, startup/shutdown/restore drills and operational acceptance. Profile validation started no services. |
| C5: governance bypass | Only authenticated INTERNAL applications can supply gateway system prompts. HTTP/MCP header resolution is shared, ambiguous credentials are rejected, activity/MCP always require authentication, and unknown tenants are denied. HIGH approvals require a different recorded maker/reviewer key ID and durably audited resolution intent. Production HTTP policy mutations are disabled. | Independent human/operator identity and signed, dual-control policy/model releases with coordinated canary/rollback. Different key IDs do not prove different humans. |

## High-priority findings

| Review item | Status and evidence scope |
|---|---|
| P1-1 auth drift | Implemented shared Bearer/X-API-Key resolution for main HTTP and MCP; conflicting credentials deny authentication. CORS permits X-API-Key. |
| P1-2 unknown tenant fallback | Implemented fail-closed admission for unprovisioned IDs; explicitly provisioned synthetic tenant fixtures replace permissive test assumptions. |
| P1-3 accounting after provider completion | Completed, guarded output returns with `token_accounting_degraded` on quota-recording failure; metric records the outage; regression asserts exactly one provider invocation. Reserve/settle and provider idempotency remain open. |
| P1-4 review capacity/expiry | Implemented bounded backpressure, bounded pagination/filtering and explicit expiry maintenance. A deployed retention/archive/sweeper schedule remains open. |
| P1-5 repeated security-identity hashing | Open. No cache identity was weakened to obtain a performance result; release-aware identity caching and independently verified complete model identities still need coherent release work. |
| P1-6 FAISS/full-file persistence | New vectors append incrementally; replacement/eviction rebuilds preserve positional correctness. Recency is tracked independently from index positions. Fuzzy snapshots are never persisted; metadata-only exact persistence retains its atomic single-writer saver. |
| P1-7 startup/readiness | Implemented bounded warm-up, startup probe, cached readiness and liveness healthcheck. A running thread can outlive timeout; real deployment recovery remains unverified. |
| P1-8 artifacts/dependencies | Offline pilot configuration and a bounded read-only model inventory are implemented. Runtime/image/model locks and reproducible builds remain open. |
| P1-9 test isolation | Fault tests inject detector behavior before warming; real encoder tests are explicitly marked integration; source tests block external DNS/socket connections while permitting loopback; offline flags precede imports; CI has per-test timeout/offline settings. Guard applies to the test process, not an OS network sandbox for arbitrary child processes. |
| P1-10 observability | Added accounting-failure metric and bounded unknown-source labels, including the diagnostic counter. Structured logging, traces, queue/outbox metrics, multiprocess metrics and alert rules remain open. |
| P1-11 coherent releases | Open. Production live HTTP policy writes are disabled as a safety boundary until signed reviewed releases exist. |
| P1-12 PII/leak/output bounds | Matched PII values are absent from metadata; suffix-automaton matching is checked against a substring oracle; tool output release has a serialized UTF-8 byte bound across HTTP/MCP shared execution. Trusted handlers can still allocate memory before returning. |
| P1-13 UI | Redirects are restricted to same-origin UI paths and identity uses textContent. JavaScript execution regression covers external/protocol-relative/script URLs and malicious identity strings. CSP/security headers and high-privilege session design remain open. |

`evidence/` is now ignored, and the corrected evaluator has a source-controlled
copy at `scripts/compute_judge_effect_corrected.py`. Historical evidence and the
original evaluator are retained. Default tests exercise physical JSONL parsing
without importing inference code. The corrected full judge evaluation remains
unrun; no new quality metric or judge-effect result is claimed.

## Reproduction and failed attempts

Every isolated test run has its own copied source, environment, source hashes,
JUnit XML and log under `_evidence/`. Existing receipts are never overwritten.
The original source baseline is retained; its expected drift after authorized
implementation is recorded separately rather than resetting source to match it.

- `readiness-20261108-01`: 54 selected cache/privacy/fault tests passed.
- `readiness-20261108-02`: failed while reconciling old permissive auth/tenant fixtures; snapshot source drift was recorded, so this is not final-source acceptance.
- `readiness-20261108-03`: 14 failures, mainly remaining synthetic admission fixtures and a test import; preserved.
- `readiness-20261108-04`: 1,819 passed, 5 skipped, 6 deselected; no source drift.
- `readiness-20261108-05`: 1,822 passed, 5 skipped, 6 deselected after adding the test-process network guard.
- `readiness-20261108-06`: one model-inventory input-validation failure; all other 1,825 selected tests passed. The inventory helper accepted an invalid repository namespace; fixed with a stricter identity grammar.
- `readiness-ui-20261108-01`: JavaScript security and socket guards passed; the same inventory grammar test failed. Preserved.
- `readiness-encoder-20261108-01`: five integration failures because the runner pointed SentenceTransformer at a different empty cache. The copied original model files were unchanged. The runner was corrected; no weights were downloaded.

Final source and dependency checks are summarized below. Earlier failures remain available for diagnosis.

## Offline model identity evidence

`_evidence/readiness-model-inventory-20261108.json` records a bounded, read-only
8,138,530,149-byte scan of explicitly selected local model files. All eight
declared detector weights and the native 1B judge weight match their SHA-256
pins. Canonical replay artifact hashes and seven runtime package versions match.
The embedding weight and tokenizer/config files have observed fingerprints;
the embedding has no independent expected SHA-256 in the existing profile.
This is file identity evidence, not inference, holdout quality or a serving
judge-identity check. The original caches/models were not migrated or deleted.

## Additional review fixes and final verification

Local verification on 2026-11-08: **1,833 passed, 9 skipped, 14 deselected**
in the copied offline source suite; **8 PostgreSQL contracts**, **5 healthy Redis
contracts plus 2 real-outage contracts**, and **5 real offline encoder checks**
passed separately. Source/model drift checks were empty. Ruff, mypy (48 source
files), Compose configuration and diff whitespace checks passed. The source
suite excludes slow/integration checks; it reported 82 deprecation warnings.
The quiescent PostgreSQL restore lost zero fixture rows, which is not a
production recovery objective. Original inputs, failed attempts and raw logs
remain local and are excluded from the public repository. Hosted CI is a
separate validation and must be read from the pull request checks.

Policy candidate validation rejects unknown root/tenant fields, capabilities and
risk keys. Request ingress enforces a configurable streaming byte bound and
rejects oversized or malformed Content-Length before reading the body. Existing
policy-version fixtures now use valid policies with different literal whitespace
to retain byte-for-byte rollback coverage under the stricter validator.

The commands below reproduce source checks; ignored local receipts retain exact hashes and validation scope.
PostgreSQL run 06 exposed a misplaced capacity-test block in the new UTC test;
the test was corrected and run 07 passed all eight contracts. Source run 10
preserves the two old policy-fixture failures; runs 11 and 12 follow the fixture
correction. No failed receipt was overwritten.

Real Redis run 01 passed five healthy-state contracts and two fail-closed outage
contracts using an owned, paused container. The PostgreSQL and Redis fixture
containers were removed after verification. These are single-instance local
dependency tests, not a multi-replica or production-availability result.

## Remaining operational acceptance

No application image was built or deployed, no public state was changed, no credentials were
rotated, no shared keys were purged and no long soak was launched. Temporary
owned synthetic PostgreSQL/Redis fixtures used existing local images. Sustained
shared-state/failover acceptance, signed releases, independent holdout evaluation,
observability, reproducible supply chain and a separately budgeted real 24-hour
soak remain open. Production readiness must not be inferred from source tests,
file hashes or a successful Compose configuration check.
