"""
Measures whether CACHE_SIMILARITY_THRESHOLD (0.95) is actually safe, and finds
a threshold that is, on the exact dataset the end-to-end benchmark uses.

WHY THIS EXISTS
---------------
The live benchmark (docs/ENGINEERING_ASSESSMENT.md §1g/1h) showed warm-cache
recall collapsing to roughly the OLD anchors-only cold-cache number, and FPR
roughly tripling, versus the fused cold-cache pass on the identical prompts.
The suspect is CACHE_SIMILARITY_THRESHOLD=0.95: two prompts that share almost
all their tokens but differ in exactly the part that matters (a swapped
payload, an opposite instruction) can easily exceed 0.95 cosine similarity in
embedding space, since the embedding is dominated by the shared surrounding
text. If that is what's happening, the cache is silently serving one prompt's
verdict for a DIFFERENT prompt with a different true label.

This script measures it directly: for every prompt in the benchmark dataset,
find its nearest OTHER prompt by cosine similarity and check whether they share
a label. Sweeping candidate thresholds shows the actual safety/hit-rate
tradeoff, so the fix is chosen from measurement, not a round number.

Usage:
    python -m scripts.diagnose_cache_threshold
"""
import numpy as np
from datasets import load_dataset

from core.embeddings import get_embedding

CANDIDATE_THRESHOLDS = [0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]


def main():
    ds = load_dataset("deepset/prompt-injections", split="train")
    prompts, labels = list(ds["text"]), list(ds["label"])
    n = len(prompts)
    print(f"Dataset: {n} prompts ({sum(labels)} malicious / {n - sum(labels)} benign)")

    print("Embedding all prompts...")
    vecs = []
    for p in prompts:
        v = get_embedding(p)
        v = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
        vecs.append(v)
    X = np.array(vecs, dtype="float32")
    X = X / np.linalg.norm(X, axis=1, keepdims=True)  # L2-normalize for cosine via dot product

    sims = X @ X.T
    np.fill_diagonal(sims, -1.0)  # exclude self

    nearest_idx = sims.argmax(axis=1)
    nearest_sim = sims[np.arange(n), nearest_idx]

    print(f"\n{'threshold':>10} {'pairs >= thr':>13} {'%same-label':>12} "
          f"{'%diff-label (DANGEROUS)':>25} {'cache hit rate':>16}")
    for thr in CANDIDATE_THRESHOLDS:
        at_or_above = nearest_sim >= thr
        n_hit = int(at_or_above.sum())
        if n_hit == 0:
            print(f"{thr:>10.3f} {0:>13} {'--':>12} {'--':>25} {0.0:>15.1%}")
            continue
        same_label = labels == np.array(labels)[nearest_idx]
        same_label_at_thr = same_label[at_or_above]
        pct_same = same_label_at_thr.mean()
        pct_diff = 1 - pct_same
        hit_rate = n_hit / n
        flag = "  <-- UNSAFE" if pct_diff > 0.01 else ""
        print(f"{thr:>10.3f} {n_hit:>13} {pct_same:>11.1%} {pct_diff:>24.1%} "
              f"{hit_rate:>15.1%}{flag}")

    # Show the worst offenders at the CURRENT default (0.95) for concrete proof.
    print(f"\n{'-'*90}")
    print("WORST COLLISIONS at threshold=0.95 (different label, high similarity):")
    print(f"{'-'*90}")
    mismatches = [
        (nearest_sim[i], prompts[i], labels[i], prompts[nearest_idx[i]], labels[nearest_idx[i]])
        for i in range(n)
        if nearest_sim[i] >= 0.95 and labels[i] != labels[nearest_idx[i]]
    ]
    mismatches.sort(key=lambda x: -x[0])
    for sim, p1, l1, p2, l2 in mismatches[:8]:
        print(f"\n  similarity={sim:.4f}")
        print(f"    query  (label={l1}): {p1[:90]!r}")
        print(f"    cached (label={l2}): {p2[:90]!r}")

    if not mismatches:
        print("  (none found at 0.95 — inspect a lower threshold band instead)")


if __name__ == "__main__":
    main()
