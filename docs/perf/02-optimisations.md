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

**Correctness guard:** this is a pure scheduling change — it cannot alter
a score, since it does not touch any arithmetic, only how many OS threads
one forward pass is allowed to use internally. `tests/test_api.py`,
`tests/test_config.py`, `tests/test_fast_path_cascade.py` (75 tests) pass
unchanged with the pin active. **Update, later the same night:** the full
non-slow suite was also run after the G3/G4 changes landed on top of this
— 1644 passed, 0 failed, no regression. **The `-m slow` decision-replay
gate and a live before/after benchmark were still NOT run** — both need
several minutes of RAM headroom this session's time budget did not have
room for alongside everything else in G2-G6. Ship as a low-risk,
well-justified change on unit-test evidence; **running the decision-replay
gate and one concurrency=16 A/B comparison against this change is the
first thing to do before calling it validated**, not assumed done.

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

| change | p50 | p95 | throughput | AUC delta |
|---|---|---|---|---|
| torch thread pinning | not benchmarked live tonight (unit-verified only) | — | — | n/a (no scoring change possible) |
| micro-batching | rejected, not implemented | — | — | — |
| ONNX export | deferred, not implemented | — | — | — |

Honest gap, stated plainly: this table has one accepted change and no live
before/after number for it. The `01-baseline-analysis.md` methodology gap
(§2a — concurrency levels sharing one warm cache) also means a same-night
before/after comparison would have been comparing two different
cache-warmth regimes, not the same experiment twice — worth fixing the
harness (clear cache between levels, or disjoint workload slices per
level) before that comparison is run at all.
