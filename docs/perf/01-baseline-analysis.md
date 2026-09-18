# Perf baseline 01 — where the time goes in `/api/v1/assess`

**Commit measured:** `e5545255` (dirty — includes the observability
instrumentation this baseline needed to exist; see §5). **Host:** the
project's own 11.34 GB / 12-core Windows dev machine — the only hardware
available at time of writing. Not representative of a production host;
treat every absolute number here as this-machine-specific and every
**relative** comparison (1 vs 4 vs 16, stage vs stage) as the portable
finding.

Raw data: `_evidence/perf/e55452550ae2-{1,4,16}.json`,
`_evidence/perf/e55452550ae2-c16-flamegraph.svg`. Reproduce with
`PYTHONPATH=. python benchmarks/load/assess_bench.py --api-key "$KEY"`.

## 1. Headline numbers

| run | concurrency | measured requests | throughput (rps) | p50 | p95 | p99 | in-flight peak |
|---|---|---:|---:|---:|---:|---:|---:|
| A (first manual session) | 1 | 263 | 8.77 | 79 ms | 348 ms | 368 ms | 1 |
| A | 4 | 342 | 11.40 | 309 ms | 621 ms | 708 ms | 3 |
| A | 16 | 140 | **4.67** | 1,809 ms | **19,814 ms** | **22,735 ms** | 16 |
| B (`run_20260917_002918`) | 16 | 140 | 10.23 | 1,607 ms | 2,182 ms | 2,475 ms | 15 |
| C (`run_20260917_004655`) | 16 | 309 | 10.30 | 1,644 ms | 2,111 ms | 2,302 ms | 16 |

`ASSESS_MAX_CONCURRENCY=4` (default, unchanged across all runs). Every
request across every run and every level returned 200 — 0 errors, 0
timeouts. **Run A's p95/p99 (19.8 s / 22.7 s) is the outlier of the three,
not the norm** — that run shared the machine with a concurrent crash
investigation (see §6) at ~1-2 GB free; runs B and C, at 4+ GB free, land
in the same 2.1-2.5 s p95 range independently. Treat B/C as the
representative concurrency-16 numbers and A as evidence of how much a
contended host can inflate a benchmark, not as the system's true tail.
**All three still show throughput at concurrency 16 far below a
properly-scaled 4x-of-concurrency-4 would predict** (concurrency 4 →
11.40 rps; 4x that is ~46 rps; concurrency 16 delivers ~10.2-10.3 rps in
the representative runs) — the qualitative finding is stable even though
the absolute tail latency isn't.

**Throughput plateaus after concurrency 4 — it does not regress below
concurrency 1.** This is the single most important number in this
document, and the previous version of this paragraph had it backwards:
it cited run A's 4.67 rps at concurrency 16 as if that were the
representative number, when §1's own note two paragraphs up says A is
the contended outlier. The representative runs (B/C, 4+ GB free) show
throughput going 8.77 rps (c=1) → 11.40 rps (c=4) → ~10.2-10.3 rps
(c=16) — a real plateau (and a slight *decline* from the c=4 peak, not
from c=1), consistent with `ASSESS_MAX_CONCURRENCY=4` capping how much
work actually runs in parallel regardless of how many callers are
waiting. More concurrent callers stop buying more throughput past 4, not
"go slower than 1 caller alone." The rest of this document explains why
and locates it precisely.

## 2. Where the time goes (flamegraph, concurrency 16, 40 s / 13,918 samples)

```
100.00%  all
 98.53%  worker-thread pool (concurrent.futures)
 82.45%  core.fusion._score_one_detector          <- ONE stage, dominant
   75.69%   TransformerDetector.score_batch (torch forward pass)
     45.38%  DeBERTa-v2 forward (protectai_injection, deepset_injection, prompt_guard_2)
     25.84%  BERT forward (toxic_bert, german_toxicity_*, jailbreak heads)
 15.13%  api.main._timed_on_worker (the new wrapper — see §5, bounds everything above)
 15.01%  core.embeddings.get_embedding (sentence-transformers encode, inside "cache_lookup")
```

**82.45% of every sampled CPU cycle during the concurrency-16 window is
inside `_score_one_detector` — i.e. running the 8-detector transformer
ensemble.** Nothing else comes close. This is not new information in
principle (the project has known its ensemble is CPU-heavy since Phase 0),
but it had never been isolated with a profiler against a live, loaded
server before this run.

### 2a. Second capture, same code, opposite bottleneck — a real finding, not a contradiction

A second flamegraph (`_evidence/perf/run_20260917_004655/e55452550ae2-c16-flamegraph.svg`,
12,644 samples) taken minutes later on the same commit shows the **inverse**
profile: **86.2% of samples in `get_embedding` → sentence-transformers →
MPNet forward, essentially none in `_score_one_detector`.**

This is not measurement noise; it is the semantic cache doing exactly what
it is supposed to do. `assess_bench.py` runs concurrency 1, then 4, then 16
**sequentially against the same server process**. The concurrency=1 leg
alone made 364 requests (332 measured + 32 warm-up) against a 299-row fixed
workload — more than one full cyclic pass — so by the time concurrency=4
starts, most of the workload is already cache-hit, and by concurrency=16
essentially all of it is. A cache hit still pays for the embedding (it is
the cache key — see §4's `cache_lookup` stage) but skips the entire fusion
ensemble that a cache miss would have run. **Which stage dominates p95 is
not a fixed property of this system — it depends on cache warmth, which
this benchmark's own methodology inadvertently manipulates** by running
levels back-to-back against one shared, disk-persisted cache
(`semantic_cache.json`, TTL 24 h, survives server restarts).

Practical read: production traffic sits somewhere between these two
captures depending on real cache hit rate, which nothing here has measured
directly. **A methodology gap to fix before trusting a single flamegraph
again:** either clear the cache between concurrency levels, or give each
level a disjoint slice of the workload, so "concurrency 16" isn't
implicitly also "concurrency 16, warm cache." Recorded here rather than
silently picking whichever capture told a cleaner story — both are real,
and the discrepancy between them is itself the most interesting single
finding in this document.

## 3. Per-detector breakdown (Prometheus `gatekeeper_detector_duration_seconds`, 14 fusion calls observed)

| detector | mean latency | architecture |
|---|---:|---|
| **deepset_injection** | **2.669 s** | DeBERTa-v3-base |
| **prompt_guard_2** | **2.606 s** | DeBERTa-v3-base (Meta) |
| **protectai_injection** | **2.579 s** | DeBERTa-v3-base |
| german_toxicity_eistakovskii | 1.608 s | BERT |
| german_toxicity_ankekat | 1.414 s | BERT |
| toxic_bert | 1.322 s | BERT |
| multilingual_head | 1.087 s | MiniLM (sentence-transformers) |
| madhurjindal_jailbreak | 0.796 s | DistilBERT-class |

The three DeBERTa-v3-base detectors are ~2× every other detector and
together account for the 45.38% DeBERTa share of the flamegraph. **This is
an independent, live confirmation of the same three models Phase 1
(`_bench_local/PHASE1_ONNX/`) already targeted for ONNX static
quantization** — the profiler and the earlier offline architecture review
agree on exactly the same bottleneck by two different methods.

The 8 detectors run in parallel (`core.fusion._get_pool`,
`ThreadPoolExecutor`), so wall-clock `fusion_ms` (mean 4.42 s, below) is not
their sum — it is closer to the slowest detector's time, inflated by
contention with the other 7 threads and whatever else is running on the
bounded assess-pool's other workers. On a 12-core machine, one assessment
alone dispatches 8 detector threads; `ASSESS_MAX_CONCURRENCY=4` concurrent
assessments can therefore have up to 32 detector-inference threads
contending for 12 cores at once. This is the CPU-oversubscription the
`_assess_pool` docstring already warned about (§3 of this analysis is the
first time it was actually measured).

## 4. Stage breakdown (Prometheus `gatekeeper_stage_duration_seconds`, all 3 runs combined, 760 requests)

| stage | count | mean | share of a full deep-path request |
|---|---:|---:|---|
| **fusion** (8 detectors, parallel dispatch) | 14 | **4.42 s** | dominant |
| **judge** (Ollama arbitration, unreachable this run) | 2 | 4.17 s | rare (2/760 requests reached it) |
| cache_lookup (embedding + cache probe) | 760 | 0.245 s | every request pays this |
| symbolic (normalize + hard-ban regex) | 16 | 0.0013 s | negligible |
| meta_intent / faiss_threat_search | 14 | ~0 ms | negligible |

**`cache_lookup` is not a dictionary lookup — it is dominated by computing
the prompt's embedding** (`core.embeddings.get_embedding`, a real
sentence-transformers forward pass; 15.01% of the flamegraph). It runs on
*every single request*, cache hit or miss, because the embedding is the
cache key. At 0.245 s × 760 requests it is the second-largest aggregate
cost in this dataset after fusion, even though no individual call is slow.

**PII redaction has no stage here.** Confirmed by reading the code before
instrumenting: `core.output_guardrails`'s PII check only runs on
`assess_output` (the response path), never on `/api/v1/assess` (the
request path this benchmark exercises). There was nothing to add a "PII"
histogram for on this endpoint — noted rather than forced in.

**"Normalise" was not split from "symbolic."** `hard_ban_triggered` calls
`normalize_prompt` as its own first line; both are timed together as one
`symbolic` stage rather than adding a second timer around a
single-digit-microsecond call with nothing else to compare it against.

## 5. Queue wait vs. execution (the `_run_bounded` split this baseline required)

Before this work, `gatekeeper_assessments_in_flight` was the only signal,
and it conflates two states with opposite fixes: queued-behind-the-pool and
actually-running. `core.metrics.queue_wait_seconds` /
`assess_execution_seconds` (new, `api.main._timed_on_worker`) separate
them:

| | count | sum | mean |
|---|---:|---:|---:|
| queue_wait_seconds | 760 | **388.1 s** | 0.51 s |
| assess_execution_seconds | 760 | 259.4 s | 0.34 s |

**More aggregate time was spent waiting for a free worker than doing any
work at all**, across the combined 1/4/16 dataset. The distribution is
sharply bimodal, not uniform: 620/760 requests (82%) waited ≤ 0.1 s
(concurrency ≤ `ASSESS_MAX_CONCURRENCY`, no queueing); 12/760 requests
waited 10–20+ s (the concurrency-16 run, where 12 of the 16 concurrent
callers are, by construction, sitting behind 4 workers the entire time).
This is the mechanical explanation for §1's p95/p99 collapse: it is queue
depth, not slower execution, that produces the 19.8 s / 22.7 s tail —
`assess_execution_seconds`' own p99 territory (≤ ~1 s for 97% of requests,
per its histogram buckets) never gets that slow on its own.

## 6. Judge on/off — attempted, and why only one data point exists

The primary baseline (§1–5) ran with `llama-guard3` unreachable (Ollama
stopped), verified by a live probe recorded in each JSON's
`judge_backend_reachable_at_start: false` — not merely assumed.

A judge-*enabled* comparison at load was **not** performed: `llama-guard3`
is a 4.9 GB model, and loading it alongside the already-resident ~1.8 GB
fusion ensemble on this 11.34 GB machine is not a theoretical risk — a
single **isolated**, non-concurrent probe (Ollama running alone, fusion
ensemble down) was attempted to get one clean judge-latency number, and the
machine hard-crashed from memory exhaustion partway through. The one result
that was captured before the crash:

```
model: llama-guard3, single prompt, cold call
total_duration: 16,820.76 ms
eval_duration:     138.33 ms
```

**138 ms of actual inference against 16.7 s of total round-trip** — the
overwhelming majority of a cold judge call is model load, not evaluation.
This independently corroborates the README's pre-existing claim
("Llama Guard... 17–27 s/request on CPU") from a different angle, and the
crash itself is the strongest evidence yet for treating judge-enabled
concurrent load testing as genuinely infra-blocked on this hardware, not
merely inconvenient — consistent with `docs/MAINTENANCE.md`'s existing
"Live benchmark on real hardware" item. No further attempt was made.

## 7. Environment recorded with every run (see the JSON files for the full block)

- **Hardware:** Windows-11, AMD64 12-core, this benchmark's only test bed.
- **`ASSESS_MAX_CONCURRENCY=4`**, `ASSESS_TIMEOUT_SECONDS=30` — both
  defaults, unchanged.
- **Torch threads: `intra_op=6`, `inter_op=6` — torch's own defaults.**
  Verified: **no `torch.set_num_threads()` call exists anywhere in this
  codebase.** This was a "verify, don't assume" item and the answer is
  that nobody has ever deliberately tuned this; 6+6 is whatever torch
  picked given 12 visible cores. Given §3's finding that a single
  assessment alone can dispatch 8 parallel detector threads, each free to
  use up to 6 intra-op threads, explicit thread pinning is a candidate
  fix worth testing before touching model architecture.
- **13 pinned model revisions** — recorded verbatim in each JSON; unchanged
  from `models/detector_manifest.json`.
- `RATE_LIMIT_AUTHENTICATED_RPM` was raised via environment variable for
  this run only (`RATE_LIMIT_AUTHENTICATED_RPM=100000`) so the benchmark
  measured pipeline capacity, not the (separately tested) rate limiter —
  noted here so nobody mistakes this for the production default.

## 8. Harness bug found while writing this document

`benchmarks/load/assess_bench.py`'s `status_counts` field used
`str(status)` as the dict key when writing but the raw `int` when reading
(`status_counts.get(s, 0)`), so it always read back 0 and reported
`{"200": 1}` regardless of the real count. Fixed in the same change that
adds this document. **Does not affect any number in this analysis** —
`measured_2xx`/`measured_total`/`error_rate`/throughput/latency are all
computed through separate, correct code paths and were cross-checked
against the flamegraph and Prometheus data independently. Left as a
concrete example of exactly the kind of small, easy-to-miss measurement
bug this whole exercise exists to catch before trusting a number.

## 9. What dominates p95 — the answer

**Fusion (the 8-detector parallel ensemble), specifically its three
DeBERTa-v3-base members, is the dominant cost of a single request; queue
depth against `ASSESS_MAX_CONCURRENCY=4` is what turns that per-request
cost into a catastrophic p95/p99 under concurrency.** Two independent
problems, two independent levers:

1. **Per-request cost** → Phase 1 (ONNX static quantization of
   `protectai_injection`, `deepset_injection`, `prompt_guard_2` —
   already scoped, blocked on RAM headroom for the export/calibrate step,
   see `docs/MAINTENANCE.md`) is aimed at exactly the right three models,
   now with live confirmation rather than only an offline architecture
   read.
2. **Queueing under concurrency** → `ASSESS_MAX_CONCURRENCY` and torch
   thread pinning are two untried, code-only levers (no model changes,
   no quantization risk) that directly address §5's finding. Worth
   testing *before* Phase 1, since they need no RAM headroom the way
   quantization does and could be validated with this same harness in
   an afternoon.

Neither of these was picked before this document existed. Per the original
brief: G2 actions are scoped from here, not before.
