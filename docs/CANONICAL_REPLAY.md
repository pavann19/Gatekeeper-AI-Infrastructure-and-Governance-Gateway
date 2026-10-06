# Canonical decision profile

`config/canonical_replay.json` defines the Phase 1.1 frozen-oracle CPU profile.
It is a correctness check against 300 frozen decisions, not an independent
accuracy estimate. The oracle comes from commit `5984269`; it is not regenerated.

Run from the repository root in a separate environment using
`requirements-canonical-replay.txt`, then:

```powershell
python -m scripts.canonical_replay --output D:\private-evidence\gatekeeper-replay
```

The slow test runs this command in a fresh subprocess. Override the default
slow-test exclusion to run the entire suite:

```powershell
python -m pytest -o addopts="" --cov=core --cov=api --cov-fail-under=65
```

The critical CPU inference versions are pinned separately from the broad legacy
requirements, whose NumPy constraint differs. Optional ONNX/Optimum packages
are outside this profile. Phase 1.1 used an isolated environment inheriting
installed dependencies, with Transformers and Hugging Face Hub installed into
that environment only. The critical versions were checked at runtime; a fresh
installation of the complete deployment dependency set was not validated.

## Required assets and behavior

- All eight model features in the profile are required, plus the anchor feature.
  A missing model, wrong revision, weight hash, artifact hash, runtime version,
  changed setting or degraded fusion tier fails the gate. Ordinary optional
  production tiers are separate configurations.
- Default anchor encoder: `sentence-transformers/all-mpnet-base-v2` at
  `e8c3b32edf5434bc2275fc9bab85f82640a19130`.
- The multilingual head, fusion artifact, policy files, frozen corpus and
  evaluation suite are identified by SHA256 with text line endings normalized.
  The ignored `data/eval_suite.jsonl` must exist and match the profile; its
  absence is an explicit error. Obtain the matching private evaluation asset
  from the maintained dataset workflow; do not substitute a different suite.
- Seed 42, two Torch threads, deterministic Torch algorithms, per-class
  nine-feature fusion, parallel detector evaluation and domain guardrail off.
- Cache reads/writes are bypassed. Both judge paths are explicitly unavailable;
  ambiguous decisions remain fail-closed HIGH. No online judge is substituted.
- `unitary/toxic-bert` explicitly lowercases according to its pinned
  `bert-base-uncased` tokenizer declaration. Transformers 5.2.0 is required for
  the verified Prompt Guard 2 whitespace preprocessing. Model weights and
  decision thresholds are unchanged.

## Measurement and claim boundary

This is the single reference profile for future comparable replay, benchmark,
calibration and published decision claims. Measurements must record this profile,
its runtime versions, artifact hashes, dataset identity, judge/cache state and
security configuration fingerprint. The shared preflight is
`scripts.canonical_replay.prepare_environment()`; decision measurements must also
use `isolated_decisions()` to retain the declared judge/cache state. A different
tier, live judge, cache mode, dataset or runtime requires a separately named
configuration and cannot be described as canonical-equivalent.

Existing exploratory benchmark/calibration tools and historical artifacts have
not been rerun or migrated into comparable measurements in Phase 1.1. Existing
headline numbers are unchanged. In particular, the historical claim that
Transformers 4.57.6 and 5.x produce equivalent detector scores is stale. This
phase validates replay correctness; evaluation integrity and renewed accuracy
claims remain separate work.

Cache assessment identity includes core source, policies, model artifacts,
settings, model availability, tenant/raw prompt identities and critical runtime
package versions. Changing tokenizer code or inference libraries invalidates old
assessment versions while HIGH stays monotonic within an unchanged version.
