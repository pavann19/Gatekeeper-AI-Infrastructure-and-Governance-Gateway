# Budget and provider-outcome hardening

Local validation on 2026-11-22. These results concern backend correctness and
synthetic failure handling, not classifier accuracy, API throughput or
production operation. Source receipts and failed attempts are retained locally
under `_evidence/budget-hardening-20261122/`.

## Fixed behavior

| Problem reproduced | Corrected behavior | Regression |
|---|---|---|
| Fractional tenant quota `0.5` became integer zero, meaning unlimited; malformed overrides inherited defaults. Non-finite rates were accepted. | Invalid explicit budgets deny only the affected tenant. Valid integer quotas, decimal integer strings, and explicit zero remain supported. Global rates, timeouts and burst windows reject non-finite values. | `tests/test_budget_hardening.py` |
| Reducing a burst from 100 to 5 retained 99 old tokens when the clock did not advance. | Both local and Redis checks clamp existing tokens to the new capacity before spending. The identical fixed-clock scenario admits 5 of 100 attempts instead of 99. | `tests/test_budget_hardening.py`, `tests/test_live_redis_acceptance.py` |
| Output assessment/audit failure skipped accounting after a completed provider call. | Record known usage immediately after completion; downstream failures do not erase consumed tokens. Each of three synthetic failure routes records 77 tokens and denies the next request without a second provider call. | `tests/test_budget_hardening.py` |
| Ollama counters were discarded; mixed malformed usage could raise or undercount valid components. | Preserve reported prompt/output counts; normalize non-negative integer usage without coercing booleans or malformed values. The synthetic Ollama response reports 11 + 18 = 29 tokens. Missing usage remains unknown, not estimated billing. | `tests/test_budget_hardening.py` |
| Transport/server/response failures and worker timeouts could dispatch a fallback after the first provider might have executed. | Unknown outcome denies output, suppresses fallback and returns explicit reconciliation guidance without Retry-After. Audit failure cannot replace that warning with an automatic-retry suggestion. | `tests/test_provider_outcome_boundary.py` |
| Source verifier inherited omitted settings such as DATABASE_URL/STATE_BACKEND, used shared model caches and constructed a deny-all empty tenant store. | Strip declared settings without runtime imports, isolate model cache paths, exclude service/model integration from source tests, and leave optional tenancy unconfigured until fixtures provision it. | `tests/test_verifier_isolation.py` |

## Measured Redis backend correctness

`python -m scripts.verify_redis_stress --output <new-output-directory>` creates
one owned digest-pinned Redis container on a random loopback port, bounded to
128 MiB and one CPU. Eight spawned processes each create their own client and
start competing after a barrier. Each performs 1,000 attempts per contract:

| Contract | Observed result |
|---|---:|
| Cached HIGH preserved during competing LOW writes | 8,000 / 8,000 |
| Shared five-token rate budget | Exactly 5 / 8,000 admitted |
| Two-token quota increments | 8,000 writes; 16,000 tokens retained |
| Independent clients/processes | 8 / 8 |
| Capacity reduction from 100 to 5 | Exactly 5 / 100 admitted |
| Separate tenant usage | Independent 7-token counter; original counter unchanged |
| Owned container removed | Yes |

The earlier resume evidence used 100 attempts and two clients with eight
threads. This run increases the experiment size and crosses process boundaries;
it does **not** establish an 80x performance improvement. The original cached
HIGH must still be retained: expiry, eviction or authorized removal can remove
it. All competing overwrites target one synthetic cache key.

`python -m scripts.verify_redis --output <output-directory>` separately passed
six healthy contracts, configured-outage denial and persistent Redis restart
readback. Both container fixtures were removed. No original user Redis data,
provider account, model weights or runtime records were used as fixtures.

## Validation scope and remaining gates

The final covered source run recorded **1,919 passed, 2 failed, 10 skipped and
14 deselected**, with **88.47% line coverage** across core/API source. Both
failures concern existing killable network-worker tests that did not reach
their DNS/stream stage before the deadline. The final run is therefore
**FAIL**, despite meeting the coverage floor. An isolated rerun failed those
two contracts and the stalled-worker elapsed-time assertion. Their cause is
not established; no timeout assertion or acceptance oracle was relaxed.
The same three failures also reproduced against the preserved pre-fix source
in `baseline-network.log` (3 failed), establishing that they predate this patch.

All **88 new regression cases** passed in that full run (64 budget/accounting,
23 provider-outcome, one verifier-isolation). The focused run also passed 133
tests. Local Ruff and the governance-boundary mypy check passed (48 source
files). Source hashes show no core/API/scripts/tests drift between the covered
candidate and the final working source. These are local Python 3.13.3 results;
new hosted Python 3.11 CI has not run. Earlier failed runs remain preserved.

Publication review added one verifier safeguard afterward: release replay now
requires an explicit existing `--canonical-model-cache` directory, while source
tests retain their empty caches. This avoids making frozen replay silently use
the deliberately empty source-test cache. Both verifier regressions passed in
a new isolated run; the original covered run predates this script-only change.
The source-test selection still excludes integration/slow tests, and the
separate canonical stage retains its original frozen acceptance assertions.
The same publication review conservatively classified HTTP 408/409 as unknown
provider outcomes in all three adapters, adding six failure contracts. Those
responses cannot prove that a completion was absent and must not trigger an
automatic fallback. The PR's final hosted checks cover these publication changes.

Token accounting remains post-completion usage gating, without distributed
reserve/settle, billing reconciliation or crash-safe exactly-once provider
execution. Unknown-outcome responses prevent this gateway's immediate fallback,
but cannot stop a caller retrying independently. Detector accuracy, untouched
holdout, complete model/config/tokenizer provenance, application multi-replica
acceptance, production performance and sustained operation remain unverified.
GK-P0 remains in progress; this checkpoint does not start later phases.

## Resume claim supported by this run

Validated Redis security-state consistency across eight independent processes:
preserved HIGH-risk cache entries through 8,000 competing overwrites, admitted
exactly five requests against a shared five-token budget, and retained all
8,000 quota updates in isolated local tests.
