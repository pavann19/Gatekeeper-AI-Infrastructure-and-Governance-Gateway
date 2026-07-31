# Evaluation Methodology

This document specifies how Gatekeeper is measured. It exists because the
project's first published metrics (37% accuracy, 98% FPR) were not wrong in
arithmetic but wrong in *method*, and the corrections are worth stating
explicitly rather than silently fixing.

---

## 1. What went wrong the first time

Three independent methodological faults, each sufficient on its own to
invalidate the published numbers:

| Fault | Effect | Fix |
|---|---|---|
| A scoping decision (off-topic) was scored as a safety prediction | 79 of 80 sampled benign prompts became false positives; FPR ~98% | `topicality` is a separate field and is excluded from the safety confusion matrix |
| The judge was offline during the run, and the pipeline fails closed to HIGH | Metrics measured sidecar availability, not detection quality | Harnesses abort on an unreachable judge unless explicitly overridden |
| The threat anchors described a different attack class than the dataset contained | AUC 0.649 — near-random — attributed to the architecture rather than the threat model | Anchors grouped by attack class; taxonomy stated explicitly |

The general lesson, and the one worth putting in the report: **a metric that is
computed correctly over the wrong construct is more dangerous than one that is
obviously broken**, because it looks publishable.

---

## 2. The evaluation suite

Built by `scripts/build_eval_suite.py` into `data/eval_suite.jsonl`, with a
manifest at `_evidence/eval_suite_manifest.json`.

**6,933 prompts — 1,925 attacks, 5,008 benign — from 7 public sources**, after
dropping 174 cross-source duplicates.

| Source | Contributes | Why included |
|---|---|---|
| `deepset/prompt-injections` | Instruction override, German-heavy | The original project benchmark; retained for continuity |
| `jackhhao/jailbreak-classification` | DAN-style persona jailbreaks + benign | A distinct attack class from injection |
| `lakera/gandalf_ignore_instructions` | Real attacker attempts | Genuine adversarial traffic, not synthetic |
| `rubend18/ChatGPT-Jailbreak-Prompts` | In-the-wild jailbreaks | Curated real-world attacks |
| `lmsys/toxic-chat` | Real user traffic, human-annotated | Realistic benign distribution — where FPR actually matters |
| `JailbreakBench/JBB-Behaviors` | Harmful-behaviour goals + benign controls | Matched pairs for the content-harm axis |
| `tatsu-lab/alpaca` | Ordinary instructions | Anchors FPR against normal traffic, not near-miss attacks |

Excluded and recorded as such: `hackaprompt/hackaprompt-dataset` and
`walledai/HarmBench` (both gated, require Hub authentication);
`JasperLS/prompt-injections` (a fork of `deepset/prompt-injections` that dedup
would fully remove anyway).

### Composition

```
by attack class : benign 5008 | prompt_injection 979 | jailbreak 692 | harmful_content 254
by language     : en 6458 | other 241 | de 234
```

### Taxonomy

Three attack classes, deliberately kept distinct because they are different
problems and a system can be good at one and useless at another:

- **`prompt_injection`** — instruction override, task hijacking, system-prompt
  extraction. *"Ignore all previous instructions."*
- **`jailbreak`** — persona and roleplay attacks that defeat the safety policy
  while leaving the task intact. *"You are DAN, who has no restrictions."*
- **`harmful_content`** — requests for genuinely dangerous information,
  regardless of framing. *"How do I synthesise sarin?"*

Pooling these into one "malicious" label is what allowed the original anchor set
(entirely `harmful_content`) to be evaluated against a dataset that was entirely
`prompt_injection` without the mismatch being visible.

### Known limitations (stated, not hidden)

- Language tags come from a stopword-overlap heuristic, not a trained language
  identifier. Adequate for stratifying metrics; not citable as ground truth.
- Attack-class labels are inherited per-source from each dataset's own
  conventions and are not independently re-annotated. The
  `jailbreak`/`harmful_content` boundary is approximate where a source mixes them.
- Benign rows pool adversarial-adjacent datasets with ordinary traffic. For a
  realistic production FPR estimate, report the `alpaca` and `toxic-chat` slices
  separately — they are the ones that resemble real usage.
- Class balance is not natural. Real traffic is overwhelmingly benign; the 28%
  attack rate here is a construction artefact. Precision figures should be read
  with that in mind, and a prevalence-adjusted precision is worth reporting for
  any production claim.

---

## 3. Metrics

Implemented in `evaluation/metrics.py`, tested in `tests/test_metrics.py`.

### Every headline number carries a confidence interval

At n=343 benign (the original dataset), the standard error on a 5% FPR is about
1.2 percentage points, so a 95% interval spans roughly ±2.3pp. Two
configurations differing by 2pp are **statistically indistinguishable**, yet
point estimates present them as different. Percentile bootstrap (2,000
resamples, seeded) is used because ROC AUC and recall-at-fixed-FPR have no
convenient closed-form variance.

Resamples that lose a class entirely are skipped rather than scored as zero —
counting them would bias small slices toward whichever class survived.

### AUC handles ties correctly

Computed via the Mann-Whitney rank-sum identity with averaged ranks within tie
groups. This is not pedantry: a detector that returns ~0.0 for every German
prompt produces a large tie block, and a naive implementation would inflate AUC
by treating those ties as wins.

### One shared operating point across slices

Slices are evaluated at a single global threshold, not at per-slice optima.
Letting each slice choose its own favourable cutoff would report a configuration
that cannot actually be deployed.

### Single-class slices are refused, not fabricated

Attack-class slices contain no benign rows, so AUC and FPR are undefined there.
The metrics module marks such slices `evaluable: false` and reports detection
rate at the shared threshold instead of inventing a number.

### Underpowered slices are flagged

Slices below n=20 are reported with an `underpowered` flag. A CI over 8 rows is
valid and useless; it should look that way in the output rather than sit in a
table beside a slice of 5,000.

---

## 4. What is measured, and what is not

**Measured:** the deterministic signal path — symbolic rules, threat-anchor
similarity, dynamic threat feed, meta-intent similarity.

**Not measured:** the semantic judge (Stage 4). Excluded deliberately because it
is non-deterministic and expensive, so sweeping thresholds over it is not
reproducible. Judge contribution belongs in a separate ablation, run once at a
fixed operating point rather than inside a sweep.

**Not measured:** the domain guardrail. It is a scoping decision, reported
separately as `topicality`.

Any published figure must state which of these it covers. "Gatekeeper achieves
X" without that qualification is the same category of error as the original 37%.

---

## 5. Reproducing

```bash
python -m scripts.build_eval_suite          # build data/eval_suite.jsonl
python -m scripts.evaluate_suite            # sliced metrics with CIs
python -m scripts.calibrate_thresholds      # ROC sweep, operating point selection
python -m tests.benchmark                   # end-to-end, requires a live judge
```

All randomness is seeded (`SEED = 20260723`). Signal extraction is cached; pass
`--refresh` / `--refresh-signals` after any change to the anchors or the encoder.

**Re-run calibration after every change to the anchor set or the embedding
model.** Thresholds are fitted to a specific signal distribution and do not
transfer across encoders.
