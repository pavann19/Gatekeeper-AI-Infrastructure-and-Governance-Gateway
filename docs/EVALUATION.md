# Evaluation

No current accuracy, throughput or latency headline is published. Each result must identify what was exercised and which dependencies were available. Tests, oracle agreement and model accuracy answer different questions.

## Profiles

| Profile | Interpretation | Required context |
|---|---|---|
| Canonical offline replay | Agreement with a frozen governance decision oracle | Pinned runtime/models/artifacts; cache bypassed; judges unavailable |
| Detector only | Scoring without live judge arbitration | Exact detector tier, dataset identity, exclusions and split |
| Cached evaluation | Behavior with explicitly populated cache state | Cache security version, hit rate, retention/flush state and warm-up |
| Live model/judge evaluation | Decisions with a reachable configured judge | Model/revision, runtime, availability, timeouts and fallback counts |
| Synthetic benchmark | Generated or hand-written workload exercise | Generator/fixture identity; no production-distribution inference |
| Deployment measurement | Authenticated requests to an actual deployed service | Source/config/server identity, hardware, dependencies, workload and errors |

Do not combine these profiles into one metric. A fail-closed decision caused by unavailable judging measures that configuration's behavior, not detector recall. Topicality is separate from maliciousness. Hand-written test prompts are regression examples, not an independent accuracy dataset.

## Canonical offline replay

`config/canonical_replay.json` declares `frozen-replay-nine-feature-v1`: detector revisions/weight hashes, encoder, runtime, settings, artifact hashes, seed and thread count. `tests/fixtures/decision_replay_corpus.jsonl` holds 300 frozen oracle cases from the declared source revision. This count describes a fixture, not a published pass rate.

Strict preflight rejects missing/mismatched artifacts, wrong runtime, unavailable required detectors and degraded fusion tiers. Decisions bypass cache; both judges are unavailable by construction. Ambiguous paths remain fail closed. Do not regenerate fixtures or adjust thresholds to make a replay pass.

Use a separate environment with the critical versions in `requirements-canonical-replay.txt` and all supporting runtime dependencies. These critical pins differ from `requirements-api.txt` and `requirements-ci.txt`, including their NumPy range; do not install the profiles into one environment indiscriminately. A fresh complete canonical dependency installation is not established by the public contract.

```sh
python -m scripts.canonical_replay --output _evidence/canonical
python -m pytest -o addopts="" tests/ --cov=core --cov=api --cov-fail-under=65
```

These commands require matching pinned model assets and the public hash-pinned frozen corpus, policies and fitted model artifacts. Replay does not consume `data/eval_suite.jsonl`: that ignored local training/corpus-generation dataset is recorded separately under `training_provenance` with its unchanged expected hash. Its upstream revisions and redistribution provenance are incomplete, so it is not public release material. To additionally verify that historical input, run `python -m scripts.canonical_replay --verify-training-provenance --output _evidence/provenance-replay`; missing or changed training data fails that explicit check. The upstream builder is not an immutable historical reconstruction. Missing or changed replay inputs always fail ordinary replay. Canonical replay does not establish live-judge, cached, accuracy or deployment equivalence.

## Detector and artifact tooling

`scripts/build_eval_suite.py` assembles taxonomy-labeled rows from upstream datasets; `scripts/compare_detectors.py` and `scripts/evaluate_suite.py` score supplied suites. Artifact training and nested multilingual validation tools remain for transparency, with their imported helpers. They require dataset/score/embedding inputs, applicable licenses/access and separately identified source revisions. They are not a complete clean-clone reproduction of the shipped artifact.

Declare dataset revisions/hashes, source exclusions, deduplication, labels, language/attack slices and independent train/calibration/test splits before publishing accuracy. Do not select an operating point on the reporting test set. Existing in-sample fit statistics inside model artifacts are historical training/sanity metadata, not current independent accuracy claims. Source-aware leakage and label distributions need explicit evaluation.

`evaluation/metrics.py` validates aligned finite scores and binary labels. ROC groups tied scores and uses `score >= threshold`; AUC uses average ranks for ties. Single-class AUC is undefined. Use confidence intervals and describe underpowered slices. An interval does not fix dataset contamination or deployment drift.

## Load tooling

`benchmarks/load/assess_bench.py` is the fixed-workload HTTP harness; it validates the workload hash and records warm-up, concurrency, percentiles, errors and metadata. `benchmarks/load/build_workload.py` builds the workload from a supplied suite. The workload is local/ignored; rebuilding against changed upstream data is not proof of the retained hash. Do not run the builder merely to overwrite an expected hash.

With a matching workload and an already running, identified server:

```sh
python benchmarks/load/assess_bench.py --api-key "$GATEKEEPER_BENCH_API_KEY" --concurrency 1,4,16 --duration 60 --warmup 10 --judge-label enabled
```

The duration and concurrency values are example run parameters, not measured results. Set the judge label to match reality. Verify server settings, source and model identities separately: harness-process metadata does not attest a remote server. Report 401/429/errors and admission controls rather than treating them as successful inference throughput. Keep raw result files private or ignored.

`scripts/load_test.py` is a general endpoint smoke/load utility, not a canonical model benchmark. `scripts/live_pentest.py` is an operator-run adversarial probe; its existence is not a completed penetration-test claim. Run such tools only against an owned isolated service with dedicated credentials/state.

## Source verification

```sh
python -m scripts.verify --profile unit --output _evidence/unit
```

The unit profile copies only public source into owned temporary state and runs lint, types and non-slow tests, including strict hashes for all replay inputs. It does not require the private training dataset. The release profile adds slow tests/replay and requires the separately provisioned pinned models and canonical runtime. It cannot silently accept missing models or a degraded tier. Training reconstruction/provenance is a separate, explicit check.

Real Redis backend acceptance is available through `python -m scripts.verify_redis --output _evidence/redis` with Docker. It owns its container and checks concurrency/outage/restart, not full application replicas. Source tests and these backend checks do not certify container operation or production deployment.

The manual utilities `tests/benchmark.py` and `tests/eval_harness.py` use external dataset/live or cache workflows. They are not the strict frozen-oracle replay, and their results must identify dataset version, judge and cache state. `tests/smoke_test_risk.py` is a diagnostic utility, not a measured security acceptance test.
