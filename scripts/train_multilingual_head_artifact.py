"""Refit and persist a multilingual embedding head as plain JSON.

Scaler, coefficients and intercept define runtime inference without pickle.
All-data training metrics are not holdout evidence. Requires matching local
embedding/dataset inputs; see docs/EVALUATION.md."""
import json
import os
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SUITE_FILE = os.path.join("data", "eval_suite.jsonl")
EMBED_FILE = os.path.join(
    "_evidence", "embeddings",
    "sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2.npy",
)
EMBED_IDS_FILE = EMBED_FILE.replace(".npy", ".ids.json")
ARTIFACT_FILE = os.path.join("models", "multilingual_head.json")
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_suite():
    with open(SUITE_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_or_build_embeddings(rows):
    if os.path.exists(EMBED_FILE) and os.path.exists(EMBED_IDS_FILE):
        with open(EMBED_IDS_FILE, "r", encoding="utf-8") as f:
            cached_ids = json.load(f)
        if cached_ids == [r["id"] for r in rows]:
            print(f"  embeddings cached -> {EMBED_FILE}")
            return np.load(EMBED_FILE)
    from sentence_transformers import SentenceTransformer
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


def main():
    rows = load_suite()
    y = [r["label"] for r in rows]
    vecs = load_or_build_embeddings(rows)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(vecs)
    model = LogisticRegression(max_iter=3000, C=1.0)
    model.fit(X_scaled, y)

    artifact = {
        "version": 1,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "embedding_dim": int(vecs.shape[1]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "training": {
            "n_rows": len(rows),
            "n_attack": int(sum(y)),
            "n_benign": int(len(y) - sum(y)),
            "note": "Refit on 100% of data/eval_suite.jsonl -- this artifact's own "
                    "score against that suite is contaminated by construction, not "
                    "an accuracy estimate. Unbiased numbers: scripts.build_"
                    "multilingual_feature's held-out-test-split (German AUC 0.826) "
                    "and leave-one-source-out (0.94-0.98 on unseen German injection "
                    "sources, 0.615 on germeval18); scripts.validate_multilingual_"
                    "head_nested's fully disjoint 3-way split (fusion-level pooled "
                    "AUC 0.923->0.945, held-out C never touched by head or fusion).",
        },
    }
    os.makedirs(os.path.dirname(ARTIFACT_FILE), exist_ok=True)
    with open(ARTIFACT_FILE, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print(f"Trained on {len(rows)} rows ({sum(y)} attack / {len(y) - sum(y)} benign)")
    print(f"Embedding dim: {vecs.shape[1]}")
    print(f"Artifact -> {ARTIFACT_FILE}")


if __name__ == "__main__":
    main()
