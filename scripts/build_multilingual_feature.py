"""
Builds a PURPOSE-BUILT multilingual attack-detection feature, instead of
hoping an English-trained classifier generalises to German.

THE REASONING THAT LED HERE
---------------------------
Three separate attempts to close the German gap by adjusting how existing
detectors are USED all failed, and the pattern in the failures is the
finding:

  - German-specific THRESHOLD on protectai_injection: bought FPR back to
    budget by giving up 15 points of recall. Moving a cutoff cannot create
    separability that is not in the score distribution.
  - SWAPPING deepset_injection for protectai_injection: regressed pooled
    and English badly; German unchanged.
  - Language-conditional WEIGHTS (fusion refit on German rows only):
    AUC 0.670 vs the global model's 0.671. Exactly zero.

That last one is decisive. If refitting the weights on the target
population changes nothing, the weights were never the problem -- the
features simply do not carry German signal. Every remaining fix has to add
INFORMATION, not redistribute it. Adding deepset_injection as a 5th feature
confirmed the frame (German AUC 0.671 -> 0.721) but plateaus there, because
every detector in the pool is English-trained.

WHAT THIS DOES DIFFERENTLY
--------------------------
The suite now carries 4,221 German rows (1,655 attacks) -- enough to TRAIN
against rather than only to evaluate on. This script embeds every row with
a genuinely multilingual sentence encoder (one shared representation space
across languages, so German and English attacks with the same intent land
near each other) and fits a classifier on those embeddings. The resulting
probability is then available as a fusion feature whose German competence
does not depend on English-language transfer.

LEAKAGE CONTROL, WHICH IS THE WHOLE METHODOLOGICAL RISK
--------------------------------------------------------
A classifier trained on suite rows and then measured on suite rows will
flatter itself. Two guards:

  1. A held-out TEST split (default 20%) is separated FIRST and never
     participates in fitting, scaling, or model selection.
  2. Within the training portion, the reported number is out-of-fold
     (StratifiedKFold, project-standard seed/folds), so even the
     development-time figure is not in-sample.

The test-split number is the honest one and is what this script leads with.
Everything else is diagnostic.

Usage:
    python -m scripts.build_multilingual_feature
    python -m scripts.build_multilingual_feature --model <hf-model-id>
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.metrics import bootstrap_ci, fmt_ci, recall_at_fpr, roc_auc

SUITE_FILE = os.path.join("data", "eval_suite.jsonl")
EMBED_DIR = os.path.join("_evidence", "embeddings")
SCORES_DIR = os.path.join("_evidence", "detector_scores")
REPORT_FILE = os.path.join("_evidence", "multilingual_feature.json")
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
FEATURE_NAME = "multilingual_head"
SEED = 20260723
FOLDS = 5
FPR_BUDGET = 0.05
N_BOOT = 500
TEST_FRACTION = 0.20


def load_suite():
    with open(SUITE_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def embed_suite(rows, model_id, batch_size=64):
    """Embeds every row, caching to .npy so re-runs and downstream scripts
    are free. Cache is keyed by model id, so switching encoders does not
    silently reuse the previous one's vectors."""
    os.makedirs(EMBED_DIR, exist_ok=True)
    slug = model_id.replace("/", "__")
    vec_path = os.path.join(EMBED_DIR, f"{slug}.npy")
    ids_path = os.path.join(EMBED_DIR, f"{slug}.ids.json")

    if os.path.exists(vec_path) and os.path.exists(ids_path):
        with open(ids_path, "r", encoding="utf-8") as f:
            cached_ids = json.load(f)
        if cached_ids == [r["id"] for r in rows]:
            print(f"  embeddings cached -> {vec_path}")
            return np.load(vec_path)
        print("  cached embeddings do not match current suite - re-embedding")

    from sentence_transformers import SentenceTransformer
    print(f"  loading {model_id} ...")
    model = SentenceTransformer(model_id)
    texts = [r["text"] for r in rows]
    print(f"  embedding {len(texts)} rows ...")
    vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=True)
    np.save(vec_path, vecs)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump([r["id"] for r in rows], f)
    print(f"  saved -> {vec_path}")
    return vecs


def report_by_language(scores, labels, langs, prefix="   "):
    out = {}
    for lang in ("en", "de", "other"):
        idx = [i for i, la in enumerate(langs) if la == lang]
        if len(idx) < 50:
            continue
        s = [scores[i] for i in idx]
        lab = [int(labels[i]) for i in idx]
        if len(set(lab)) < 2:
            continue
        auc = bootstrap_ci(s, lab, roc_auc, n_boot=N_BOOT)
        rec = bootstrap_ci(s, lab, lambda a, b: recall_at_fpr(a, b, budget=FPR_BUDGET),
                           n_boot=N_BOOT)
        print(f"{prefix}{lang:<8} n={len(idx):<6} AUC={fmt_ci(auc, pct=False):<24} "
              f"recall@{FPR_BUDGET:.0%}FPR={fmt_ci(rec)}")
        out[lang] = {"n": len(idx), "n_attack": int(sum(lab)), "auc": auc, "recall_at_fpr": rec}
    return out


def leave_one_source_out(rows, vecs, y, langs):
    """
    The test that separates "fits this data" from "actually works".

    A random held-out split still draws test rows from sources the model
    trained on, so it cannot distinguish a model that learned what an
    attack looks like from one that learned each dataset's house style
    (annotation quirks, prompt templates, scraping artefacts). Holding out
    an ENTIRE SOURCE and training on the rest asks the question that
    matters for deployment: does this work on German text from somewhere it
    has never seen?

    Reported per German-bearing source. A model that only memorised source
    style collapses here; one that generalises degrades gracefully.
    """
    print("\n[LEAVE-ONE-SOURCE-OUT -- generalisation to unseen sources]")
    sources = sorted({r["source"] for r in rows})
    de_idx_all = [i for i, la in enumerate(langs) if la == "de"]
    de_sources = sorted({rows[i]["source"] for i in de_idx_all})

    out = {}
    for src in sources:
        if src not in de_sources:
            continue
        test_idx = [i for i, r in enumerate(rows) if r["source"] == src and langs[i] == "de"]
        train_idx = [i for i, r in enumerate(rows) if r["source"] != src]
        test_labels = [int(y[i]) for i in test_idx]
        if len(test_idx) < 50 or len(set(test_labels)) < 2:
            print(f"   {src:<52} skipped (n={len(test_idx)}, single-class or too small)")
            continue

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
        clf.fit(vecs[train_idx], y[train_idx])
        scores = clf.predict_proba(vecs[test_idx])[:, 1]
        auc = bootstrap_ci(list(scores), test_labels, roc_auc, n_boot=N_BOOT)
        print(f"   {src:<52} n={len(test_idx):<5} de AUC={fmt_ci(auc, pct=False)}")
        out[src] = {"n": len(test_idx), "n_attack": int(sum(test_labels)), "de_auc": auc}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    rows = load_suite()
    y = np.array([r["label"] for r in rows])
    langs = [r["language"] for r in rows]
    print(f"Suite: {len(rows)} rows ({int(y.sum())} attacks)")

    vecs = embed_suite(rows, args.model)
    print(f"  embedding dim: {vecs.shape[1]}")

    # --- Held-out test split, separated BEFORE anything is fitted --------
    idx_all = np.arange(len(rows))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=TEST_FRACTION, random_state=SEED, stratify=y)
    print(f"\nSplit: {len(idx_train)} train / {len(idx_test)} held-out test")

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))

    # Development-time OOF view (diagnostic only).
    print("\n[out-of-fold on the TRAIN portion -- diagnostic]")
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof_train = cross_val_predict(clf, vecs[idx_train], y[idx_train], cv=cv,
                                  method="predict_proba")[:, 1]
    oof_by_lang = report_by_language(
        oof_train, y[idx_train], [langs[i] for i in idx_train])

    # --- The honest number: fit on train, score the untouched test split -
    print("\n[HELD-OUT TEST -- the number that counts]")
    clf.fit(vecs[idx_train], y[idx_train])
    test_scores = clf.predict_proba(vecs[idx_test])[:, 1]
    test_by_lang = report_by_language(
        test_scores, y[idx_test], [langs[i] for i in idx_test])
    pooled_auc = bootstrap_ci(list(test_scores), [int(v) for v in y[idx_test]],
                              roc_auc, n_boot=N_BOOT)
    print(f"   {'pooled':<8} n={len(idx_test):<6} AUC={fmt_ci(pooled_auc, pct=False)}")

    loso = leave_one_source_out(rows, vecs, y, langs)

    # --- Emit as a fusion feature, using OOF over the FULL suite so no row
    #     ever receives a score from a model that saw its own label.
    print("\nEmitting full-suite out-of-fold scores as a fusion feature ...")
    oof_full = cross_val_predict(clf, vecs, y, cv=cv, method="predict_proba")[:, 1]
    os.makedirs(SCORES_DIR, exist_ok=True)
    feature_path = os.path.join(SCORES_DIR, f"{FEATURE_NAME}.jsonl")
    with open(feature_path, "w", encoding="utf-8") as f:
        for r, s in zip(rows, oof_full):
            f.write(json.dumps({"id": r["id"], "score": float(s)}) + "\n")
    print(f"  -> {feature_path}")

    report = {
        "model": args.model,
        "feature_name": FEATURE_NAME,
        "seed": SEED, "folds": FOLDS, "fpr_budget": FPR_BUDGET,
        "test_fraction": TEST_FRACTION,
        "n_rows": len(rows), "n_train": len(idx_train), "n_test": len(idx_test),
        "embedding_dim": int(vecs.shape[1]),
        "oof_train_by_language": oof_by_lang,
        "held_out_test_by_language": test_by_lang,
        "held_out_test_pooled_auc": pooled_auc,
        "leave_one_source_out_german": loso,
    }
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report -> {REPORT_FILE}")


if __name__ == "__main__":
    main()
