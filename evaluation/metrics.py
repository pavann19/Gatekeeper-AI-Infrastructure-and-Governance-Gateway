"""
Metrics with uncertainty quantification.

WHY BOOTSTRAP CONFIDENCE INTERVALS
----------------------------------
The original evaluation reported point estimates from 343 benign prompts. At
that size the standard error on a false-positive rate near 5% is roughly 1.2
percentage points, so a 95% interval spans about +/-2.3pp. Two configurations
differing by 2pp are therefore indistinguishable, yet a point estimate presents
them as different. Every comparison in this project reports an interval so that
"A beats B" is a claim the data can actually support.

Bootstrap is used rather than a closed form because the statistics of interest
(ROC AUC, recall at a fixed FPR budget) have no convenient analytic variance,
and because the same machinery then applies to every slice without special
cases.

No sklearn/scipy dependency: AUC, PR and the percentile bootstrap are all short
enough to implement directly, which also keeps the CI install lean.
"""
import math
import random

DEFAULT_BOOTSTRAP = 2000
DEFAULT_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Point statistics
# ---------------------------------------------------------------------------

def roc_auc(scores, labels):
    """
    ROC AUC via the rank-sum (Mann-Whitney U) identity, with correct handling
    of tied scores. Ties matter here: a detector that returns 0.0 for every
    German prompt produces a large tie block, and ignoring it would inflate AUC.
    """
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n = len(pairs)
    if n == 0:
        return float("nan")

    # Average ranks within tie groups.
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1

    pos_ranks = sum(r for r, (_, lab) in zip(ranks, pairs) if lab)
    n_pos = sum(1 for _, lab in pairs if lab)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def roc_points(scores, labels):
    """Returns [(fpr, tpr, threshold)] sorted by descending threshold."""
    ordered = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(1 for _, lab in ordered if lab)
    n_neg = len(ordered) - n_pos
    if not n_pos or not n_neg:
        return []
    pts = [(0.0, 0.0, math.inf)]
    tp = fp = 0
    for score, lab in ordered:
        if lab:
            tp += 1
        else:
            fp += 1
        pts.append((fp / n_neg, tp / n_pos, score))
    return pts


def recall_at_fpr(scores, labels, budget=0.05):
    """Best achievable recall while keeping FPR at or below `budget`."""
    pts = roc_points(scores, labels)
    if not pts:
        return float("nan")
    return max((tpr for fpr, tpr, _ in pts if fpr <= budget), default=0.0)


def threshold_at_fpr(scores, labels, budget=0.05):
    """The score cutoff achieving `recall_at_fpr`."""
    pts = roc_points(scores, labels)
    feasible = [(tpr, thr) for fpr, tpr, thr in pts if fpr <= budget]
    if not feasible:
        return float("nan")
    return max(feasible)[1]


def confusion(scores, labels, threshold):
    tp = fp = tn = fn = 0
    for s, lab in zip(scores, labels):
        pred = s >= threshold
        if lab and pred:
            tp += 1
        elif not lab and pred:
            fp += 1
        elif not lab and not pred:
            tn += 1
        else:
            fn += 1
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def derived(cm):
    tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
    }


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(scores, labels, statistic, n_boot=DEFAULT_BOOTSTRAP,
                 alpha=DEFAULT_ALPHA, seed=20260723):
    """
    Percentile bootstrap CI for `statistic(scores, labels)`.

    Resamples rows with replacement. Draws where one class vanishes are skipped
    rather than counted as zero — they carry no information about the statistic
    and including them would bias small slices toward whichever class survived.

    Returns {point, lo, hi, n_boot_effective, ...}.
    """
    point = statistic(scores, labels)
    n = len(scores)
    if n == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0, "n_boot_effective": 0}

    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        s = [scores[i] for i in idx]
        lab = [labels[i] for i in idx]
        if len(set(lab)) < 2:
            continue
        value = statistic(s, lab)
        if not (isinstance(value, float) and math.isnan(value)):
            samples.append(value)

    if not samples:
        return {"point": point, "lo": float("nan"), "hi": float("nan"),
                "n": n, "n_boot_effective": 0}

    samples.sort()
    lo_i = int((alpha / 2) * len(samples))
    hi_i = min(int((1 - alpha / 2) * len(samples)), len(samples) - 1)
    return {
        "point": point,
        "lo": samples[lo_i],
        "hi": samples[hi_i],
        "n": n,
        "n_boot_effective": len(samples),
        "ci_level": 1 - alpha,
    }


def fmt_ci(ci, pct=True, decimals=None):
    """
    '74.5% [70.1, 78.9]' — the format used in printed tables.

    Proportions default to 1 decimal; ratios (AUC) default to 3, because at 1
    decimal an AUC interval renders as "0.9 [0.9, 0.9]", which hides both the
    estimate and its width.
    """
    if ci is None or (isinstance(ci.get("point"), float) and math.isnan(ci["point"])):
        return "n/a"
    scale, suffix = (100, "%") if pct else (1, "")
    if decimals is None:
        decimals = 1 if pct else 3
    return (f"{ci['point'] * scale:.{decimals}f}{suffix} "
            f"[{ci['lo'] * scale:.{decimals}f}, {ci['hi'] * scale:.{decimals}f}]")


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------

def evaluate_slice(scores, labels, fpr_budget=0.05, n_boot=DEFAULT_BOOTSTRAP,
                   threshold=None):
    """Full metric bundle for one slice, with CIs on AUC and recall@FPR."""
    n_pos = sum(1 for lab in labels if lab)
    n_neg = len(labels) - n_pos
    out = {
        "n": len(labels),
        "n_attack": n_pos,
        "n_benign": n_neg,
        "evaluable": n_pos > 0 and n_neg > 0,
    }
    if not out["evaluable"]:
        # A slice with only one class cannot support AUC or FPR. Reporting a
        # number here would be meaningless, so say so explicitly instead.
        out["note"] = (
            "single-class slice: AUC and FPR are undefined. "
            + ("Only attacks present — report recall only."
               if n_pos else "Only benign present — report FPR only.")
        )
        if n_pos and threshold is not None:
            out["recall_at_threshold"] = sum(1 for s in scores if s >= threshold) / n_pos
        if n_neg and threshold is not None:
            out["fpr_at_threshold"] = sum(1 for s in scores if s >= threshold) / n_neg
        return out

    out["auc"] = bootstrap_ci(scores, labels, roc_auc, n_boot=n_boot)
    out["recall_at_fpr"] = bootstrap_ci(
        scores, labels,
        lambda s, lab: recall_at_fpr(s, lab, budget=fpr_budget),
        n_boot=n_boot,
    )
    out["fpr_budget"] = fpr_budget

    thr = threshold if threshold is not None else threshold_at_fpr(scores, labels, fpr_budget)
    if not (isinstance(thr, float) and (math.isnan(thr) or math.isinf(thr))):
        cm = confusion(scores, labels, thr)
        out["threshold"] = thr
        out["confusion_matrix"] = cm
        out.update(derived(cm))
    return out


def evaluate_by(records, score_key, label_key, slice_key, fpr_budget=0.05,
                n_boot=DEFAULT_BOOTSTRAP, threshold=None, min_n=20):
    """
    Metrics per value of `slice_key` (e.g. attack_class, language, source).

    Slices below `min_n` are reported but flagged — a CI computed over 8 rows is
    technically valid and practically useless, and it should look that way in
    the output rather than sitting in a table beside a slice of 5,000.
    """
    groups = {}
    for r in records:
        groups.setdefault(r[slice_key], []).append(r)

    out = {}
    for value, rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        scores = [r[score_key] for r in rows]
        labels = [r[label_key] for r in rows]
        result = evaluate_slice(scores, labels, fpr_budget=fpr_budget,
                                n_boot=n_boot, threshold=threshold)
        if len(rows) < min_n:
            result["underpowered"] = True
            result["warning"] = f"n={len(rows)} < {min_n}; interval is too wide to act on"
        out[value] = result
    return out
