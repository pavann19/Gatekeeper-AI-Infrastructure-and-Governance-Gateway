"""
Leakage-free validation of the multilingual_head's FUSION-level contribution
(issue #4's first requirement) -- the gap scripts.sweep_fusion_variants'
`plus_ml_5`/`all_7`/`plus_ml_9` numbers explicitly warned about:

    "Feeding an OOF-generated feature into a second CV over the same rows
    is the standard stacking protocol, but it is mildly optimistic -- the
    fusion's training folds contain rows whose head-scores came from a
    head fitted on data overlapping the fusion's own test fold."

METHODOLOGY: A THREE-WAY DISJOINT SPLIT
------------------------------------------
    A (40%) -- fits the multilingual head ONLY. Never touched again.
    B (30%) -- the head scores B (genuinely out-of-sample for the head,
               since B was never in A). Fusion is fit on B using those
               scores alongside the other 8 detectors.
    C (30%) -- held out from BOTH fits. The head scores C (also
               out-of-sample), the fusion (fit on B) scores C. This is
               the number that actually answers "does adding this feature
               help", with no stacking optimism anywhere in the chain.

A control fusion (the shipped 8 features, no multilingual_head) is fit on
the SAME B and evaluated on the SAME C, so the comparison isolates the
head's marginal contribution rather than conflating it with "a different,
possibly luckier split."

Usage:
    python -m scripts.validate_multilingual_head_nested
"""
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.metrics import bootstrap_ci, fmt_ci, recall_at_fpr, roc_auc
from scripts.analyze_german_by_task import INJECTION_SOURCES
from scripts.sweep_fusion_variants import PLUS_GERMAN_TOX_8, load_scores

SUITE_FILE = os.path.join("data", "eval_suite.jsonl")
EMBED_FILE = os.path.join(
    "_evidence", "embeddings",
    "sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2.npy",
)
EMBED_IDS_FILE = EMBED_FILE.replace(".npy", ".ids.json")
REPORT_FILE = os.path.join("_evidence", "multilingual_head_nested_validation.json")
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEED = 20260824
FPR_BUDGET = 0.05
N_BOOT = 500


def load_suite():
    with open(SUITE_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_embeddings(rows):
    """Reuses the cache scripts.build_multilingual_feature already wrote,
    re-embedding only if the suite has changed since."""
    if os.path.exists(EMBED_FILE) and os.path.exists(EMBED_IDS_FILE):
        with open(EMBED_IDS_FILE, "r", encoding="utf-8") as f:
            cached_ids = json.load(f)
        if cached_ids == [r["id"] for r in rows]:
            print(f"  embeddings cached -> {EMBED_FILE}")
            return np.load(EMBED_FILE)
    print(f"  embedding {len(rows)} rows with {MODEL_ID} ...")
    model = SentenceTransformer(MODEL_ID)
    vecs = model.encode([r["text"] for r in rows], batch_size=64,
                        show_progress_bar=True, convert_to_numpy=True,
                        normalize_embeddings=True)
    os.makedirs(os.path.dirname(EMBED_FILE), exist_ok=True)
    np.save(EMBED_FILE, vecs)
    with open(EMBED_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump([r["id"] for r in rows], f)
    return vecs


def report_by_slice(scores, labels, idx_de_inj, idx_de_off, tag):
    out = {}
    labels = np.asarray(labels)
    for name, idx in (("pooled", range(len(labels))),
                      ("de_injection", idx_de_inj), ("de_offensive", idx_de_off)):
        idx = list(idx)
        if len(idx) < 20 or len(set(labels[idx])) < 2:
            continue
        s = [scores[i] for i in idx]
        lab = [int(labels[i]) for i in idx]
        auc = bootstrap_ci(s, lab, roc_auc, n_boot=N_BOOT)
        rec = bootstrap_ci(s, lab, lambda a, b: recall_at_fpr(a, b, budget=FPR_BUDGET), n_boot=N_BOOT)
        print(f"  [{tag}] {name:<14} n={len(idx):<6} AUC={fmt_ci(auc, pct=False):<24} "
              f"recall@{FPR_BUDGET:.0%}FPR={fmt_ci(rec)}")
        out[name] = {"n": len(idx), "n_attack": int(sum(lab)), "auc": auc, "recall_at_fpr": rec}
    return out


def main():
    rows = load_suite()
    y_all = np.array([r["label"] for r in rows])
    vecs = load_embeddings(rows)

    caches = {n: load_scores(n) for n in PLUS_GERMAN_TOX_8}
    usable = [i for i, r in enumerate(rows) if all(r["id"] in caches[n] for n in PLUS_GERMAN_TOX_8)]
    print(f"Usable rows (all 8 base features cached): {len(usable)} of {len(rows)}")

    idx_A, idx_rest = train_test_split(usable, test_size=0.60, random_state=SEED,
                                       stratify=y_all[usable])
    idx_B, idx_C = train_test_split(idx_rest, test_size=0.50, random_state=SEED,
                                    stratify=y_all[idx_rest])
    print(f"Split: A(head fit)={len(idx_A)}  B(fusion fit)={len(idx_B)}  C(final eval)={len(idx_C)}")

    # --- Stage 1: fit the multilingual head on A only ---------------------
    head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    head.fit(vecs[idx_A], y_all[idx_A])
    head_scores_B = head.predict_proba(vecs[idx_B])[:, 1]
    head_scores_C = head.predict_proba(vecs[idx_C])[:, 1]

    # --- Stage 2: fit two fusions on B -- with and without the head -------
    def build_X(indices, feats, head_scores=None):
        X = [[caches[f][rows[i]["id"]] for f in feats] for i in indices]
        if head_scores is not None:
            X = [row + [hs] for row, hs in zip(X, head_scores)]
        return np.array(X)

    fusion_control = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    fusion_control.fit(build_X(idx_B, PLUS_GERMAN_TOX_8), y_all[idx_B])

    fusion_with_head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    fusion_with_head.fit(build_X(idx_B, PLUS_GERMAN_TOX_8, head_scores_B), y_all[idx_B])

    # --- Stage 3: evaluate BOTH on C, which neither fit has ever seen -----
    scores_control_C = fusion_control.predict_proba(build_X(idx_C, PLUS_GERMAN_TOX_8))[:, 1]
    scores_with_head_C = fusion_with_head.predict_proba(
        build_X(idx_C, PLUS_GERMAN_TOX_8, head_scores_C))[:, 1]

    langs_C = [rows[i]["language"] for i in idx_C]
    sources_C = [rows[i]["source"] for i in idx_C]
    de_inj_C = [i for i, (la, src) in enumerate(zip(langs_C, sources_C))
               if la == "de" and src in INJECTION_SOURCES]
    de_off_C = [i for i, (la, src) in enumerate(zip(langs_C, sources_C))
               if la == "de" and src not in INJECTION_SOURCES]
    y_C = y_all[idx_C]

    print("\n=== HELD-OUT C (neither the head nor either fusion has seen these rows) ===")
    control = report_by_slice(scores_control_C, y_C, de_inj_C, de_off_C, "control (8 features)")
    with_head = report_by_slice(scores_with_head_C, y_C, de_inj_C, de_off_C, "with multilingual_head (9)")

    report = {
        "seed": SEED, "fpr_budget": FPR_BUDGET,
        "n_head_fit": len(idx_A), "n_fusion_fit": len(idx_B), "n_final_eval": len(idx_C),
        "control_8_feature": control,
        "with_multilingual_head_9_feature": with_head,
    }
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport -> {REPORT_FILE}")


if __name__ == "__main__":
    main()
