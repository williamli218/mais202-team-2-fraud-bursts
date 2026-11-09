"""Known-answer tests for your track's headline metric (D1.5).

Run them with `pytest tests/test_metrics.py`. Only the cases for the track
in config.py run; the rest skip. They fail until the functions in
src/metrics.py are written, and a failing message names the function and the
case. The expected values were computed from a reference implementation that
follows the docstrings in src/metrics.py to the letter.

Don't edit this file. If you believe a case is wrong, tell your TPM, and
bring the input and the value you get.
"""

import math

import numpy as np
import pytest

import config
from conftest import same, skip_unless_track
from src import metrics

# ---------------------------------------------------------------------------
# Cases. Each is (name, kwargs). The expected values are at the bottom.
# ---------------------------------------------------------------------------

# Tracks 3, 4, 6 and 10 share the recall-at-fixed-FPR family. The budget
# (max_fpr) is part of each case; your track's own budget is 0.01 (Tracks 3
# and 6) or 0.05 (Tracks 4 and 10).
FPR_FAMILY_CASES = [
    ("clean separation, budget forces zero false positives",
     dict(y_true=[0, 0, 0, 0, 1, 1], scores=[.1, .2, .3, .4, .8, .9], max_fpr=0.01)),
    ("budget allows one false positive out of ten",
     dict(y_true=[0] * 10 + [1, 1, 1],
          scores=[.05, .1, .15, .2, .25, .3, .35, .4, .45, .95, .3, .6, .97], max_fpr=0.10)),
    ("ties at the budget: three legit share the score just under it",
     dict(y_true=[0, 0, 0, 0, 1, 1], scores=[.5, .5, .5, .1, .5, .9], max_fpr=0.5)),
    ("no fraud in the slice",
     dict(y_true=[0, 0], scores=[.1, .2], max_fpr=0.01)),
    ("the trivial baseline: every score is zero",
     dict(y_true=[0, 0, 0, 1], scores=[0, 0, 0, 0], max_fpr=0.05)),
    ("no legitimate rows at all",
     dict(y_true=[1, 1], scores=[.3, .7], max_fpr=0.01)),
]

TRACK1_CASES = [
    ("clean separation",
     dict(y_true=[0, 0, 0, 0, 1, 1], scores=[.1, .2, .3, .4, .8, .9], target_recall=0.85)),
    ("one fraud buried under legit scores",
     dict(y_true=[0, 0, 0, 0, 0, 1, 1, 1, 1], scores=[.1, .5, .2, .3, .6, .9, .8, .7, .4], target_recall=0.85)),
    ("ties at the cut: fraud and legit share 0.5",
     dict(y_true=[0, 0, 1, 1, 0, 1], scores=[.5, .5, .5, .9, .1, .5], target_recall=0.85)),
    ("target hit exactly: 17 of 20",
     dict(y_true=[1] * 20 + [0, 0, 0, 0], scores=list(range(1, 21)) + [3.5, 4, 10, 0], target_recall=0.85)),
    ("no fraud in the slice",
     dict(y_true=[0, 0, 0], scores=[.1, .2, .3], target_recall=0.85)),
    ("the trivial baseline: every score is zero",
     dict(y_true=[0, 0, 0, 1], scores=[0, 0, 0, 0], target_recall=0.85)),
]

TRACK2_CASES = [
    ("two attacked cards, held mid-attack",
     dict(y_true=[1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
          scores=[.2, .3, .9, .1, .3, .2, .1, .4, .6, .4, .1],
          card_ids=["A", "A", "A", "A", "B", "B", "C", "C", "C", "D", "D"],
          max_legit_hold_rate=0.01)),
    ("never held: attacked card scores below the threshold",
     dict(y_true=[1, 1, 1, 0], scores=[.1, .1, .1, .2], card_ids=["A", "A", "A", "B"], max_legit_hold_rate=0.01)),
    ("ties: legit cards' max equals the attacked card's first score",
     dict(y_true=[1, 1, 0, 0], scores=[.5, .7, .5, .5], card_ids=["A", "A", "B", "C"], max_legit_hold_rate=0.01)),
    ("no attacked cards",
     dict(y_true=[0, 0, 0], scores=[.1, .2, .3], card_ids=["A", "A", "B"], max_legit_hold_rate=0.01)),
    ("the trivial baseline: every score is zero",
     dict(y_true=[1, 0, 1, 0, 0], scores=[0, 0, 0, 0, 0], card_ids=["A", "A", "A", "B", "B"], max_legit_hold_rate=0.01)),
    ("budget allows one of two legit cards to be held",
     dict(y_true=[1, 1, 0, 1, 0, 0], scores=[.4, .5, .1, .7, .3, .6], card_ids=["A", "A", "A", "A", "B", "C"], max_legit_hold_rate=0.5)),
]

TRACK4_RATIO_CASES = [
    ("equal false positive rates",
     dict(y_true=[0, 0, 0, 0, 0, 0, 1, 1], flagged=[1, 0, 0, 1, 0, 0, 1, 1],
          group=[True, True, True, False, False, False, True, False])),
    ("group flagged twice as often",
     dict(y_true=[0] * 8, flagged=[1, 1, 0, 0, 1, 0, 0, 0],
          group=[True, True, True, True, False, False, False, False])),
    ("nothing flagged outside the group, something inside",
     dict(y_true=[0, 0, 0, 0], flagged=[1, 0, 0, 0], group=[True, True, False, False])),
    ("nothing flagged anywhere",
     dict(y_true=[0, 0, 0, 0], flagged=[0, 0, 0, 0], group=[True, True, False, False])),
    ("no legitimate rows in the group",
     dict(y_true=[1, 1, 0, 0], flagged=[1, 1, 0, 0], group=[True, True, False, False])),
    ("flagged fraud doesn't count as a false positive",
     dict(y_true=[0, 1, 0, 1], flagged=[0, 1, 0, 1], group=[True, True, False, False])),
]

TRACK6_LATENCY_CASES = [
    ("twenty identical timings", dict(seconds=[0.001] * 20)),
    ("ten increasing timings", dict(seconds=[i / 1000 for i in range(1, 11)])),
    ("a single timing", dict(seconds=[0.5])),
    ("no timings", dict(seconds=[])),
]

TRACK7_CASES = [
    ("top three of six",
     dict(y_true=[1, 0, 1, 0, 0, 1], scores=[.9, .8, .7, .6, .5, .1], k=3)),
    ("ties broken by position: the earlier row wins",
     dict(y_true=[0, 1, 1, 0], scores=[.5, .5, .5, .1], k=2)),
    ("k larger than the number of rows",
     dict(y_true=[1, 0, 1], scores=[.3, .2, .1], k=100)),
    ("no fraud in the slice",
     dict(y_true=[0, 0, 0], scores=[.1, .2, .3], k=2)),
    ("the trivial baseline: every score is zero",
     dict(y_true=[0, 0, 1, 0], scores=[0, 0, 0, 0], k=2)),
]

TRACK9_F1_CASES = [
    ("one of each: tp, fp, fn, plus an unknown row",
     dict(y_true=[1, 1, 0, 0, -1], y_pred=[1, 0, 1, 0, 1])),
    ("perfect", dict(y_true=[1, 0, 1], y_pred=[1, 0, 1])),
    ("nothing illicit and nothing predicted", dict(y_true=[0, 0], y_pred=[0, 0])),
    ("nothing illicit but something predicted", dict(y_true=[0, 0], y_pred=[1, 0])),
    ("unknown rows are ignored", dict(y_true=[-1, -1, 1], y_pred=[1, 1, 1])),
]

TRACK9_STEP_CASES = [
    ("three steps, one with no illicit and nothing predicted",
     dict(y_true=[1, 0, 1, 0, -1, 0], y_pred=[1, 1, 0, 0, 1, 0], steps=[1, 1, 2, 2, 3, 3])),
    ("a step with only unknown rows is left out",
     dict(y_true=[1, 0, -1], y_pred=[1, 0, 1], steps=[1, 1, 2])),
]

# Which cases each track runs, as (function name, case list).
CASES_BY_TRACK = {
    1: [("fpr_at_recall", TRACK1_CASES)],
    2: [("median_frauds_before_hold", TRACK2_CASES)],
    3: [("threshold_at_fpr", FPR_FAMILY_CASES), ("recall_at_fpr", FPR_FAMILY_CASES)],
    4: [("threshold_at_fpr", FPR_FAMILY_CASES), ("recall_at_fpr", FPR_FAMILY_CASES), ("fpr_ratio", TRACK4_RATIO_CASES)],
    6: [("threshold_at_fpr", FPR_FAMILY_CASES), ("recall_at_fpr", FPR_FAMILY_CASES), ("p95_latency_ms", TRACK6_LATENCY_CASES)],
    7: [("precision_at_k", TRACK7_CASES), ("recall_at_k", TRACK7_CASES)],
    9: [("f1_illicit", TRACK9_F1_CASES), ("f1_per_step", TRACK9_STEP_CASES)],
    10: [("threshold_at_fpr", FPR_FAMILY_CASES), ("recall_at_fpr", FPR_FAMILY_CASES)],
}


def _all_cases():
    seen, out = set(), []
    for track, pairs in CASES_BY_TRACK.items():
        for func, cases in pairs:
            for name, kwargs in cases:
                key = (func, name)
                if key not in seen:
                    seen.add(key)
                    out.append((track, func, name, kwargs))
    return out


def _tracks_for(func, name):
    return [t for t, pairs in CASES_BY_TRACK.items()
            for f, cases in pairs if f == func for n, _ in cases if n == name]


@pytest.mark.parametrize(
    "func,name,kwargs",
    [(f, n, k) for _, f, n, k in _all_cases()],
    ids=[f"{f}: {n}" for _, f, n, _ in _all_cases()],
)
def test_metric(func, name, kwargs):
    skip_unless_track(*_tracks_for(func, name))
    expected = EXPECTED[(func, name)]
    fn = getattr(metrics, func)
    try:
        got = fn(**{k: (np.asarray(v) if isinstance(v, list) else v) for k, v in kwargs.items()})
    except NotImplementedError as e:
        pytest.fail(f"Implement {func} in src/metrics.py (D1.5). {e}", pytrace=False)
    if not same(got, expected):
        pytest.fail(
            f"{func} in src/metrics.py (D1.5) returned {got!r} on the case \"{name}\"; expected {expected!r}.\n"
            f"Inputs: {kwargs}",
            pytrace=False,
        )


def test_inputs_can_be_lists_or_series():
    """A metric must accept plain lists and pandas Series, not only arrays."""
    skip_unless_track(3, 4, 6, 10)
    import pandas as pd
    y = [0, 0, 0, 0, 1, 1]
    s = [.1, .2, .3, .4, .8, .9]
    try:
        a = metrics.recall_at_fpr(y, s, 0.01)
        b = metrics.recall_at_fpr(pd.Series(y), pd.Series(s), 0.01)
    except NotImplementedError as e:
        pytest.fail(f"Implement recall_at_fpr in src/metrics.py (D1.5). {e}", pytrace=False)
    assert same(a, b), "recall_at_fpr gives different answers for a list and a Series with the same values"
    assert same(a, 1.0), f"recall_at_fpr returned {a!r} on an easy case; expected 1.0"


# ---------------------------------------------------------------------------
# Expected values. Generated from the staff reference implementation; the
# block between the markers is rewritten by a script, so don't edit by hand.
# nan means "undefined for this input", inf means "no threshold flags little enough".
# ---------------------------------------------------------------------------
nan, inf = float("nan"), float("inf")

# --- EXPECTED (generated) ---
EXPECTED = {
    ('fpr_at_recall', 'clean separation'): 0.0,
    ('fpr_at_recall', 'one fraud buried under legit scores'): 0.4,
    ('fpr_at_recall', 'ties at the cut: fraud and legit share 0.5'): 0.666666666667,
    ('fpr_at_recall', 'target hit exactly: 17 of 20'): 0.5,
    ('fpr_at_recall', 'no fraud in the slice'): nan,
    ('fpr_at_recall', 'the trivial baseline: every score is zero'): 1.0,
    ('median_frauds_before_hold', 'two attacked cards, held mid-attack'): 1.5,
    ('median_frauds_before_hold', 'never held: attacked card scores below the threshold'): 3.0,
    ('median_frauds_before_hold', "ties: legit cards' max equals the attacked card's first score"): 1.0,
    ('median_frauds_before_hold', 'no attacked cards'): nan,
    ('median_frauds_before_hold', 'the trivial baseline: every score is zero'): 2.0,
    ('median_frauds_before_hold', 'budget allows one of two legit cards to be held'): 0.0,
    ('threshold_at_fpr', 'clean separation, budget forces zero false positives'): 0.8,
    ('threshold_at_fpr', 'budget allows one false positive out of ten'): 0.6,
    ('threshold_at_fpr', 'ties at the budget: three legit share the score just under it'): 0.9,
    ('threshold_at_fpr', 'no fraud in the slice'): inf,
    ('threshold_at_fpr', 'the trivial baseline: every score is zero'): inf,
    ('threshold_at_fpr', 'no legitimate rows at all'): 0.3,
    ('recall_at_fpr', 'clean separation, budget forces zero false positives'): 1.0,
    ('recall_at_fpr', 'budget allows one false positive out of ten'): 0.666666666667,
    ('recall_at_fpr', 'ties at the budget: three legit share the score just under it'): 0.5,
    ('recall_at_fpr', 'no fraud in the slice'): nan,
    ('recall_at_fpr', 'the trivial baseline: every score is zero'): 0.0,
    ('recall_at_fpr', 'no legitimate rows at all'): 1.0,
    ('fpr_ratio', 'equal false positive rates'): 1.0,
    ('fpr_ratio', 'group flagged twice as often'): 2.0,
    ('fpr_ratio', 'nothing flagged outside the group, something inside'): inf,
    ('fpr_ratio', 'nothing flagged anywhere'): 1.0,
    ('fpr_ratio', 'no legitimate rows in the group'): nan,
    ('fpr_ratio', "flagged fraud doesn't count as a false positive"): 1.0,
    ('p95_latency_ms', 'twenty identical timings'): 1.0,
    ('p95_latency_ms', 'ten increasing timings'): 9.55,
    ('p95_latency_ms', 'a single timing'): 500.0,
    ('p95_latency_ms', 'no timings'): nan,
    ('precision_at_k', 'top three of six'): 0.666666666667,
    ('precision_at_k', 'ties broken by position: the earlier row wins'): 0.5,
    ('precision_at_k', 'k larger than the number of rows'): 0.666666666667,
    ('precision_at_k', 'no fraud in the slice'): 0.0,
    ('precision_at_k', 'the trivial baseline: every score is zero'): 0.0,
    ('recall_at_k', 'top three of six'): 0.666666666667,
    ('recall_at_k', 'ties broken by position: the earlier row wins'): 0.5,
    ('recall_at_k', 'k larger than the number of rows'): 1.0,
    ('recall_at_k', 'no fraud in the slice'): nan,
    ('recall_at_k', 'the trivial baseline: every score is zero'): 0.0,
    ('f1_illicit', 'one of each: tp, fp, fn, plus an unknown row'): 0.5,
    ('f1_illicit', 'perfect'): 1.0,
    ('f1_illicit', 'nothing illicit and nothing predicted'): nan,
    ('f1_illicit', 'nothing illicit but something predicted'): 0.0,
    ('f1_illicit', 'unknown rows are ignored'): 1.0,
    ('f1_per_step', 'three steps, one with no illicit and nothing predicted'): {1: 0.666666666667, 2: 0.0, 3: nan},
    ('f1_per_step', 'a step with only unknown rows is left out'): {1: 1.0},
}
# --- END EXPECTED ---
