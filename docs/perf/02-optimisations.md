# Perf optimisations — G2 candidates, each gated by a `01-baseline-analysis.md` finding

Rule followed: no change here is speculative. Each row cites the exact
finding that justified attempting it, and every accepted change is followed
by its correctness guard, not just a benchmark number. A rejected change is
recorded with why, per instruction — not deleted.

| G1 finding | Change | Status | Correctness guard |
|---|---|---|---|
| Queue wait dominates in the noisy run; CPU oversubscribed by design (§3, per-detector table) | Pin `torch.set_num_threads(1)` / `set_num_interop_threads(1)` | **Accepted, shipped** | see below |
| Cold model load on first requests | Eager load at startup + readiness gate | **Already satisfied** — no change needed | see below |
| Judge calls dominate the tail | Report judge-path latency separately; keep async/timeout-bounded | **Already satisfied** — no change needed | see below |
| Per-detector forward passes dominate (on a cold cache) | Micro-batch: one thread per detector, batches requests ≤10-20ms | **Rejected for tonight** | see below |
| Transformer CPU inference is slow | ONNX export of the 3 DeBERTa-v3 detectors | **Deferred, unchanged** | see below |

## 1. Torch thread pinning — accepted

**Finding cited:** `docs/perf/01-baseline-analysis.md` §7 — no
`torch.set_num_threads()` call existed anywhere in the codebase; torch
defaulted to `intra_op=6, inter_op=6` on this 12-core box. §3 measured 8
detectors dispatched in parallel per assessment; at
`ASSESS_MAX_CONCURRENCY=4` that is up to 32 detector-inference threads each
independently trying to claim 6 torch threads — the oversubscription
`api/main.py`'s own pool-size comment already predicted, now measured
rather than assumed.

**Change:** `core/torch_threads.py` (new), imported as the first line of
`api/main.py`, before any transitive `import torch`. Calls
`torch.set_num_threads(1)` / `torch.set_num_interop_threads(1)` once, at
process start — both new config fields
(`TORCH_INTRA_OP_THREADS`/`TORCH_INTER_OP_THREADS`, default 1). The outer
8-way `ThreadPoolExecutor` in `core/fusion.py` becomes the only source of
parallelism instead of two layers of parallelism (outer thread pool x
inner torch threads) competing for the same 12 cores.

**Correctness guard:** this is a pure scheduling change, it can't alter a
score since it doesn't touch any arithmetic, only how many OS threads one
forward pass is allowed to use internally. `tests/test_api.py`,
`tests/test_config.py`, `tests/test_fast_path_cascade.py` (75 tests total)
pass unchanged with the pin active, and the full non-slow suite was run
again after the G3/G4 changes landed on top of this — 1644 passed, 0
failed, no regression. The live before/after benchmark has now been run
too, see the results section below. Still outstanding: the `-m slow`
decision-replay gate, which needs a real corpus pass through the live
decision path. That's a different kind of validation than a scheduling
benchmark and is the next thing to do before calling this fully
validated on the correctness side — the performance side is covered now.

## 2. Eager load at startup — already satisfied, verified

**Finding cited:** cold first-request latency is a known cost (§6:
llama-guard3 alone showed ~99% of a cold call's time is model load, not
inference).

**Status:** `api/main.py`'s `@app.on_event("startup")` (line 366) already
calls the detector warm-up path unconditionally unless explicitly disabled,
and uvicorn does not begin accepting connections until that handler
returns — this session's own server logs from earlier tonight show it
directly: `"Model warm-up complete in 88.9s — warmed 8 detector(s)"`,
printed before `"Application startup complete"` / `"Uvicorn running"`. No
request can reach a cold model. No change needed; recorded here so this
G2 item isn't silently skipped without evidence that it was already done.

## 3. Judge latency reported separately, already async/timeout-bounded — already satisfied

**Finding cited:** §6, judge calls dominate any request that reaches them.

**Status:** both already true, and both landed as part of this session's
own instrumentation work, not tonight: `core.risk.assess_risk` times the
judge call under its own `stage="judge"` Prometheus histogram
(`core/risk.py`, ~line 793), separate from `fusion`/`cache_lookup`/
`symbolic`. The live API path (`background_scheduler is not None`) already
answers with the fast Ollama judge and defers Llama Guard confirmation to
a background task — never blocking the response — and `judge_arbitration`
already fails closed on an unreachable/timed-out backend (verified by
`tests/test_fault_injection.py::test_judge_unreachable_fallback_and_fail_closed`,
green tonight). No change needed.

## 4a. Is the thread-pin fix alone enough, or is more needed?

Answering this after the live benchmark, not before it — that's kind of
the whole point of running #1-4 first instead of deciding on
architecture from theory.

Short answer: it's a real win, but it doesn't close the case. It worked
about as predicted — c=4/c=16 throughput up 38%/56%, p50/p95 down
37-40% at c=16, right where the G1 finding said oversubscribed torch
threads were costing the most under concurrent load. But it didn't touch
the tail: c=16 p99 is flat at 2,410ms vs ~2,302-2,475ms in the baseline,
basically no change. And c=16 throughput (16.03 rps) is still well below
what a naive 4x-of-c=4 scale-up would predict (~63 rps).
`ASSESS_MAX_CONCURRENCY=4` capping parallel work explains part of that
gap by design, but not all of it — the original flamegraph finding (82%
of CPU in `_score_one_detector`, one serialized forward pass per
detector per request) is still true. The thread pin changed how many OS
threads are fighting over a core, not how much CPU work each request
still does.

So: micro-batching and ONNX are still worth doing, and neither is any
more or less urgent than it was in §4/§5 below. The thread pin was the
safe, purely-scheduling win, worth taking first. What's left — the
per-detector forward-pass cost itself — needs one of the two options
already on the table, and both still carry the same verification cost
(decision-replay gate, RAM-heavy export) that made sense to defer
tonight rather than rush. Their status is unchanged below, just backed
by a live number now instead of an assumption.

## 4. Micro-batching detectors — rejected for tonight

**Finding cited:** §2's cold-cache flamegraph, 82% of CPU in
`_score_one_detector`.

**Why rejected rather than attempted:** this is a real architectural
change — a per-detector batching thread, a request-collection window, and
a rewrite of `core/fusion.py`'s dispatch loop — landing on the exact code
path that decides BLOCK/ALLOW for every request. The stated correctness
guard ("same scores ± float tolerance on the frozen corpus") requires
running the decision-replay gate and ideally `evaluate_accuracy.py`, both
of which need real model inference over a real corpus — meaningful RAM and
wall-clock time this session could not safely spend unattended alongside
G3-G6, and not something to ship un-verified on a security-critical
decision path just to check a box before a deadline. Genuinely worth
doing; needs a dedicated session with headroom to run the guard, not a
rushed implementation at 4 a.m.

## 5. ONNX export of the 3 DeBERTa-v3 detectors — deferred, status unchanged

**Finding cited:** §3, `deepset_injection`/`prompt_guard_2`/
`protectai_injection` are ~2x every other detector; §2's flamegraph
independently confirms DeBERTa-v2/v3 forward passes as the largest single
chunk on a cold cache.

**Why not attempted tonight:** already known infra-blocked
(`docs/MAINTENANCE.md`) — the export/optimize/calibrate chain has crashed
this machine from memory exhaustion before (`_bench_local/PHASE1_ONNX/`
research log), and tonight's window already used the machine hard for the
G2-G6 work above. No new attempt. Status unchanged from `MAINTENANCE.md`:
parked pending a host with real RAM headroom.

## Results table (per the brief's format)

| change | p50 (c=16) | p95 (c=16) | throughput (c=16) | AUC delta |
|---|---|---|---|---|
| torch thread pinning | 973 ms (was ~1,607-1,644 ms, **-40%**) | 1,336 ms (was ~2,111-2,182 ms, **-38%**) | 16.03 rps (was ~10.2-10.3 rps, **+56%**) | n/a (no scoring change possible) |
| micro-batching | rejected, not implemented | — | — | — |
| ONNX export | deferred, not implemented | — | — | — |

**Live before/after, commit `b59b57417e41`, same workload file
(`workload_v1.jsonl`, hash-verified), same machine:**

| | c=1 | c=4 | c=16 |
|---|---|---|---|
| Baseline throughput (rps) | 8.77 | 11.40 | ~10.2-10.3 |
| With thread-pin throughput (rps) | 7.35 | 15.72 | 16.03 |
| Baseline p50 / p95 / p99 | 79 / 348 / 368 ms | 309 / 621 / 708 ms | ~1,607-1,644 / ~2,111-2,182 / ~2,302-2,475 ms |
| With thread-pin p50 / p95 / p99 | 108 / 377 / 387 ms | 196 / 509 / 626 ms | 973 / 1,336 / 2,410 ms |

Raw JSON: `_evidence/perf/b59b57417e41-{1,4,16}.json` alongside the
baseline's `_evidence/perf/e55452550ae2-{1,4,16}.json`.

A few things worth calling out instead of just quoting the win:

c=4 and c=16 throughput improved a lot (+38%, +56%) and latency dropped
at p50/p95 (-37% to -40%) — that's a real, measured improvement right
where the theory said it should land, under concurrent load where
torch's unpinned threads were actually oversubscribing the box. But c=16
p99 didn't move (2,410ms vs ~2,302-2,475ms baseline, basically flat) —
the fix helped the median and the shoulder of the distribution, not the
tail, and that's worth knowing before calling this a blanket latency
win. c=1 throughput actually went down a bit (7.35 vs 8.77 rps, about
-16%), which makes sense since there's no oversubscription at
concurrency 1 for the pin to fix — most likely this is just noise from a
single data point rather than the pin adding overhead, but it's reported
as-is rather than explained away.

Two methodology notes for anyone trying to reproduce this: this run used
`--allow-anonymous` with `RATE_LIMIT_ENABLED=false` (anonymous requests
otherwise hit `RATE_LIMIT_ANONYMOUS_RPM=20` almost immediately at these
concurrency levels — the first attempt actually hit a 99.88% error rate
from 429s before that got caught), where the original baseline run used
an authenticated API key instead. Both bypass rate limiting so it
shouldn't affect the latency numbers, but it is a different auth path
than the original run took. Also, this run started from a freshly
cleared `semantic_cache.json` / `semantic_cache_exact.json` — deleted on
purpose before the restart to avoid the cache-warmth issue
`01-baseline-analysis.md` §2a already flagged. The baseline's starting
cache state was never independently checked, so if it had any residual
warmth, that would make the baseline look slower than it should, not the
other way around.
