import pytest

from rag_judge_eval.metrics import binary_kappa, cohens_kappa


def test_cohens_kappa_matches_nominal_three_class_example():
    human = ["correct", "correct", "partially_correct", "wrong"]
    judge = ["correct", "partially_correct", "partially_correct", "wrong"]
    assert cohens_kappa(human, judge) == pytest.approx(0.6363636364)


def test_kappa_is_undefined_for_empty_or_constant_marginals():
    assert cohens_kappa([], []) is None
    assert cohens_kappa(["correct", "correct"], ["correct", "correct"]) is None
    assert binary_kappa([True, True], [True, True]) is None


def test_kappa_rejects_mismatched_or_invalid_labels():
    with pytest.raises(ValueError, match="equal length"):
        cohens_kappa(["correct"], [])
    with pytest.raises(ValueError, match="answer labels"):
        cohens_kappa(["pending"], ["correct"])
