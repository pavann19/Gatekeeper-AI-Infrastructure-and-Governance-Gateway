"""
Tests for the evaluation metrics.

These matter more than typical test code: every number in the technical report
comes out of this module. A silent bug here does not crash anything, it just
produces a confidently wrong result — which is the failure mode this whole
Phase 1 effort exists to eliminate.
"""
import math

import pytest

from evaluation.metrics import (
    bootstrap_ci,
    confusion,
    derived,
    evaluate_by,
    evaluate_slice,
    recall_at_fpr,
    roc_auc,
    threshold_at_fpr,
)


# --- AUC correctness --------------------------------------------------------

def test_auc_perfect_separation():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    assert roc_auc(scores, labels) == 1.0


def test_auc_inverted_separation():
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [0, 0, 1, 1]
    assert roc_auc(scores, labels) == 0.0


def test_auc_all_tied_is_one_half():
    """
    The case that matters for this project: a detector returning the same score
    for everything must score 0.5, not 1.0. An implementation that ignores ties
    gets this wrong, and German prompts scoring identically near zero form
    exactly such a tie block.
    """
    scores = [0.5] * 6
    labels = [0, 1, 0, 1, 0, 1]
    assert roc_auc(scores, labels) == pytest.approx(0.5)


def test_auc_partial_ties():
    scores = [0.1, 0.5, 0.5, 0.9]
    labels = [0, 0, 1, 1]
    # Positives {0.5, 0.9} vs negatives {0.1, 0.5}, four pairs:
    #   0.5 > 0.1 -> 1.0
    #   0.5 = 0.5 -> 0.5  (tie counts half)
    #   0.9 > 0.1 -> 1.0
    #   0.9 > 0.5 -> 1.0
    # => 3.5 / 4 = 0.875
    assert roc_auc(scores, labels) == pytest.approx(0.875)


def test_auc_undefined_for_single_class():
    assert math.isnan(roc_auc([0.1, 0.2, 0.3], [1, 1, 1]))
    assert math.isnan(roc_auc([0.1, 0.2, 0.3], [0, 0, 0]))


def test_auc_matches_manual_rank_computation():
    scores = [0.3, 0.1, 0.9, 0.4, 0.7]
    labels = [0, 0, 1, 1, 0]
    # Concordant pairs / total pairs, computed by hand.
    pos = [s for s, lab in zip(scores, labels) if lab]
    neg = [s for s, lab in zip(scores, labels) if not lab]
    expected = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
    assert roc_auc(scores, labels) == pytest.approx(expected)


# --- recall at an FPR budget ------------------------------------------------

def test_recall_at_fpr_perfect():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    assert recall_at_fpr(scores, labels, budget=0.0) == 1.0


def test_recall_at_fpr_respects_budget():
    # 10 benign, 10 attacks; attacks overlap the benign range.
    scores = [0.0, 0.1, 0.2, 0.3, 0.4] + [0.35, 0.45, 0.6, 0.7, 0.8]
    labels = [0] * 5 + [1] * 5
    # Zero FPR allowed -> only attacks scoring above every benign count.
    assert recall_at_fpr(scores, labels, budget=0.0) == pytest.approx(0.8)
    # Allowing 20% FPR (1 of 5 benign) lets the 0.35 attack through too.
    assert recall_at_fpr(scores, labels, budget=0.2) == pytest.approx(1.0)


def test_threshold_at_fpr_is_achievable():
    scores = [0.0, 0.1, 0.2, 0.3, 0.4, 0.35, 0.45, 0.6, 0.7, 0.8]
    labels = [0] * 5 + [1] * 5
    budget = 0.2
    thr = threshold_at_fpr(scores, labels, budget=budget)
    cm = confusion(scores, labels, thr)
    assert derived(cm)["fpr"] <= budget + 1e-9


# --- confusion matrix -------------------------------------------------------

def test_confusion_and_derived():
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [1, 0, 1, 0]
    cm = confusion(scores, labels, threshold=0.5)
    assert cm == {"TP": 1, "FP": 1, "TN": 1, "FN": 1}
    d = derived(cm)
    assert d["precision"] == pytest.approx(0.5)
    assert d["recall"] == pytest.approx(0.5)
    assert d["fpr"] == pytest.approx(0.5)
    assert d["f1"] == pytest.approx(0.5)
    assert d["accuracy"] == pytest.approx(0.5)


def test_derived_handles_empty_denominators():
    d = derived({"TP": 0, "FP": 0, "TN": 0, "FN": 0})
    assert d["precision"] == 0.0 and d["recall"] == 0.0 and d["f1"] == 0.0


# --- bootstrap --------------------------------------------------------------

def test_bootstrap_interval_brackets_point_estimate():
    scores = [i / 100 for i in range(100)]
    labels = [0] * 50 + [1] * 50
    ci = bootstrap_ci(scores, labels, roc_auc, n_boot=300)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci["ci_level"] == 0.95


def test_bootstrap_is_deterministic_under_seed():
    scores = [0.1, 0.4, 0.6, 0.9, 0.3, 0.7]
    labels = [0, 0, 1, 1, 0, 1]
    a = bootstrap_ci(scores, labels, roc_auc, n_boot=200, seed=42)
    b = bootstrap_ci(scores, labels, roc_auc, n_boot=200, seed=42)
    assert a == b


def test_small_sample_interval_is_wider_than_large_sample():
    """
    The whole justification for bootstrapping: n must move the interval.

    Uses deliberately OVERLAPPING class distributions. Perfectly separable data
    yields AUC 1.0 in every resample and therefore zero spread at any n, which
    would make this test vacuously pass.
    """
    def spread(n):
        # Deterministic pseudo-shuffle produces genuine class overlap.
        scores = [((i * 37) % n) / n for i in range(n)]
        labels = [i % 2 for i in range(n)]
        ci = bootstrap_ci(scores, labels, roc_auc, n_boot=400)
        return ci["hi"] - ci["lo"]

    small, large = spread(20), spread(400)
    assert small > 0, "overlapping data must produce a non-degenerate interval"
    assert small > large


def test_bootstrap_skips_degenerate_resamples():
    """Draws losing a class are skipped, not scored as zero."""
    scores = [0.2, 0.9]
    labels = [0, 1]
    ci = bootstrap_ci(scores, labels, roc_auc, n_boot=200)
    # Half of all 2-row resamples are single-class and must be discarded.
    assert ci["n_boot_effective"] < 200
    assert ci["point"] == 1.0


# --- slicing ----------------------------------------------------------------

def test_single_class_slice_is_flagged_not_scored():
    """
    Attack-class slices contain no benign rows. Reporting an AUC there would be
    meaningless, so the slice must be marked unevaluable rather than given a
    fabricated number.
    """
    res = evaluate_slice([0.4, 0.6, 0.8], [1, 1, 1], n_boot=50)
    assert res["evaluable"] is False
    assert "note" in res
    assert "auc" not in res


def test_single_class_slice_reports_recall_at_threshold():
    res = evaluate_slice([0.4, 0.6, 0.8], [1, 1, 1], n_boot=50, threshold=0.5)
    assert res["recall_at_threshold"] == pytest.approx(2 / 3)


def test_evaluate_by_flags_underpowered_slices():
    records = (
        [{"s": 0.9, "y": 1, "g": "big"} for _ in range(30)]
        + [{"s": 0.1, "y": 0, "g": "big"} for _ in range(30)]
        + [{"s": 0.9, "y": 1, "g": "tiny"}, {"s": 0.1, "y": 0, "g": "tiny"}]
    )
    out = evaluate_by(records, "s", "y", "g", n_boot=100, min_n=20)
    assert out["tiny"]["underpowered"] is True
    assert "underpowered" not in out["big"]


def test_evaluate_by_applies_shared_threshold():
    """Slices must be compared at one operating point, not per-slice optima."""
    records = (
        [{"s": 0.8, "y": 1, "g": "a"}, {"s": 0.2, "y": 0, "g": "a"}] * 15
        + [{"s": 0.4, "y": 1, "g": "b"}, {"s": 0.1, "y": 0, "g": "b"}] * 15
    )
    out = evaluate_by(records, "s", "y", "g", n_boot=100, threshold=0.5)
    assert out["a"]["threshold"] == 0.5
    assert out["b"]["threshold"] == 0.5
    # Slice b's attacks all fall below the shared threshold.
    assert out["b"]["recall"] == 0.0
