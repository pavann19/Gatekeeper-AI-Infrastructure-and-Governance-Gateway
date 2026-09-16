# Decisions

Short ADR-style log: context, options considered, decision, consequence.
Not exhaustive — the ones that would otherwise need re-explaining.

## 1. Fail-closed, not fail-open

**Context:** what happens when a dependency the decision needs (embedding
model, judge, cache) is unavailable.
**Options:** fail-open (allow the request, log the gap) vs. fail-closed
(restrict/block, log the gap).
**Decision:** fail-closed everywhere in the decision path.
**Consequence:** availability incidents show up as elevated block rates, not
silent policy bypass. Costs legitimate traffic during an outage; that
trade was made deliberately for a security gateway.

## 2. Logistic-regression fusion over a single detector

**Context:** no single open detector covers injection, jailbreak, and
harmful-content well (best single detector: 2.0% recall on harmful-content
at 5% FPR).
**Options:** ship the single best detector (Prompt Guard 2, 0.949 AUC) vs.
a learned fusion of several.
**Decision:** logistic-regression fusion over 5 detectors, trained
out-of-fold.
**Consequence:** pooled-AUC gain over the best single detector isn't
statistically decisive (overlapping bootstrap CIs), but per-class coverage
improves materially. Fusion is the right call for coverage, not for a
headline AUC number — see the README's evaluation section for the honest
framing.

## 3. Per-feature degradation tiers, not an all-or-nothing health check

**Context:** individual detectors can fail to load (RAM pressure, missing
weights) independently of each other.
**Decision:** each detector degrades independently; fusion falls back to
the anchors-only path (`vector_*`/`clean_pass` in the response `source`
field) rather than failing the whole request when one detector is down.
**Consequence:** partial capability loss is visible in the response and in
metrics, instead of an opaque full outage.

## 4. Prompt Guard 2 — rejected, then adopted

**Context:** Prompt Guard 2 (Meta, gated model) was initially skipped in
favor of open, ungated alternatives to avoid a licensing/access dependency.
**Decision:** re-evaluated after the open alternatives underperformed on
AUC; adopted once gated access was confirmed workable for this project's
use.
**Consequence:** best single-detector AUC (0.949) now comes from a gated
model — an explicit dependency worth knowing about before deploying this
in an environment without Meta model access.

## 5. Bounded thread pool over the default executor

**Context:** G1 baseline profiling found 8 detectors × up to
`ASSESS_MAX_CONCURRENCY` concurrent assessments oversubscribing CPU, with
torch's per-process thread pool independently claiming its unpinned
default thread count on top of that.
**Decision:** `ASSESS_MAX_CONCURRENCY=4` bounded pool (already in place)
plus pinning torch's intra/inter-op thread counts to 1 per worker
(`core/torch_threads.py`), imported before any `torch` import in the
process.
**Consequence:** removes a scheduling-only oversubscription source; cannot
change model outputs since it touches only thread-pool sizing. See
[`docs/perf/02-optimisations.md`](perf/02-optimisations.md) for what was
and wasn't re-verified before shipping it.

## 6. FAISS for anchor search

**Context:** threat-signature matching against a growing anchor set was a
linear scan.
**Decision:** `faiss.IndexFlatIP` in-memory index, built at startup from
vectorized policy signatures.
**Consequence:** O(log N)-ish retrieval instead of O(N); adds FAISS as a
dependency and an in-memory index that must be rebuilt on signature
updates (`POST /api/v1/update`).

## 7. SQLite audit mirror alongside JSONL

**Context:** the audit trail needs to be both append-only/tamper-evident
(JSONL) and queryable (for the review queue, compliance reporting).
**Decision:** write both — JSONL as the source of truth, SQLite as a
queryable mirror.
**Consequence:** two write paths to keep in sync; a JSONL-write failure is
treated as the more serious failure mode (fail-closed on that path), while
the SQLite mirror is best-effort.

## 8. Redis-shared circuit breaker

**Context:** circuit-breaker state (is the judge/embedding model healthy)
needs to be shared if the gateway ever runs as more than one process.
**Decision:** back the circuit breaker with Redis rather than in-process
state, even though today's deployment is a single FastAPI process.
**Consequence:** one more runtime dependency (Redis) for a single-process
deployment that doesn't strictly need shared state yet, in exchange for
not needing to revisit this if the process count changes.
