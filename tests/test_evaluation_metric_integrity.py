import itertools
import math

import pytest

from evaluation.metrics import confusion, recall_at_fpr, roc_auc, roc_points, threshold_at_fpr


def test_ties_do_not_create_unattainable_operating_points():
    for labels in ([1, 0], [0, 1]):
        assert roc_points([0.5, 0.5], labels) == [(0, 0, math.inf), (1, 1, 0.5)]
        assert recall_at_fpr([0.5, 0.5], labels, 0) == 0
        assert threshold_at_fpr([0.5, 0.5], labels, 0) == math.inf
        assert roc_auc([0.5, 0.5], labels) == 0.5


def test_every_roc_point_matches_its_confusion_matrix():
    rows = [(0.9, 1), (0.9, 0), (0.4, 1), (0.2, 0)]
    for perm in itertools.permutations(rows):
        scores, labels = map(list, zip(*perm))
        for fpr, tpr, threshold in roc_points(scores, labels):
            cm = confusion(scores, labels, threshold)
            assert fpr == cm['FP'] / 2
            assert tpr == cm['TP'] / 2
        for budget in (0, 0.5, 1):
            cm = confusion(scores, labels, threshold_at_fpr(scores, labels, budget))
            assert cm['FP'] / 2 <= budget
            assert cm['TP'] / 2 == recall_at_fpr(scores, labels, budget)


@pytest.mark.parametrize('scores,labels', [([1], []), ([math.nan], [1]), ([math.inf], [0]), ([0], [2])])
def test_invalid_inputs_are_not_silently_truncated(scores, labels):
    for metric in (roc_auc, roc_points):
        with pytest.raises(ValueError):
            metric(scores, labels)


def test_empty_and_single_class():
    assert math.isnan(roc_auc([], []))
    assert roc_points([1], [1]) == []
    assert math.isnan(threshold_at_fpr([1], [1]))
