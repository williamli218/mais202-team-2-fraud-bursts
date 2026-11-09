"""Your track's headline metric. Every function here is yours to write (D1.5).

Which functions your track needs:

    Track 1   fpr_at_recall
    Track 2   median_frauds_before_hold
    Track 3   threshold_at_fpr, recall_at_fpr
    Track 4   threshold_at_fpr, recall_at_fpr, fpr_ratio
    Track 6   threshold_at_fpr, recall_at_fpr, p95_latency_ms
    Track 7   precision_at_k, recall_at_k
    Track 9   f1_illicit, f1_per_step
    Track 10  threshold_at_fpr, recall_at_fpr

Work against the tests: `pytest tests/test_metrics.py` runs the cases for
your track (set in config.py) and skips the rest. Each docstring below is the
full definition, including what to do with ties and with a slice that has no
fraud in it. The tests check those cases on purpose, because the obvious
implementation gets them wrong. Read the docstring, write the function, run
the tests, repeat until green.

Conventions used throughout:
  - y_true is 0/1 (1 = fraud, phishing or illicit), scores are floats where
    higher means more suspicious, and "flagged at threshold t" means score >= t.
  - Inputs can be lists, numpy arrays or pandas Series; convert with np.asarray.
  - Rates are fractions (0.012), not percentages.
  - "No fraud in the slice" returns float('nan') where the quantity is
    undefined, never an exception. A metric that crashes on an empty slice
    takes the whole month loop (Track 10) or per-step report (Track 9) down
    with it.
"""

import numpy as np  # noqa: F401  (you'll need it)

_TODO = "TODO D1.5: implement {name} in src/metrics.py. The docstring is the spec; tests/test_metrics.py has the cases."


# ---------------------------------------------------------------------------
# Recall at a fixed false positive rate  (Tracks 3, 4, 6, 10)
# ---------------------------------------------------------------------------

def threshold_at_fpr(y_true, scores, max_fpr):
    """The threshold that flags as much as possible while keeping FPR <= max_fpr.

    Candidates are every distinct score value, plus +inf (flag nothing).
    FPR at t = (legitimate rows with score >= t) / (legitimate rows). Return
    the smallest candidate t whose FPR is <= max_fpr. Lower t flags more, so
    this is the most aggressive threshold the budget allows.

    Ties: all rows sharing a score are flagged together or not at all, which
    is why candidates are distinct values rather than ranks.
    Edge cases: no legitimate rows -> FPR is 0 at every t -> return min(scores).
    Empty input -> float('inf').
    """
    raise NotImplementedError(_TODO.format(name="threshold_at_fpr"))


def recall_at_fpr(y_true, scores, max_fpr):
    """Share of fraud caught at threshold_at_fpr(y_true, scores, max_fpr).

    recall = (fraud rows with score >= t) / (fraud rows).
    No fraud rows -> float('nan').
    """
    raise NotImplementedError(_TODO.format(name="recall_at_fpr"))


# ---------------------------------------------------------------------------
# Track 1: FPR at a fixed recall
# ---------------------------------------------------------------------------

def fpr_at_recall(y_true, scores, target_recall=0.85):
    """Share of legitimate rows flagged when the system catches target_recall of fraud.

    Candidates are every distinct score value. Pick the largest t whose
    recall (fraud with score >= t / all fraud) is >= target_recall, then
    return the FPR at that t. Largest t = flags the least while still hitting
    the target. Lower is better.

    No fraud rows -> float('nan'). No legitimate rows -> 0.0.
    """
    raise NotImplementedError(_TODO.format(name="fpr_at_recall"))


# ---------------------------------------------------------------------------
# Track 2: frauds that get through before a card is held
# ---------------------------------------------------------------------------

def median_frauds_before_hold(y_true, scores, card_ids, max_legit_hold_rate=0.01):
    """Median number of fraudulent transactions that get through on an attacked
    card before the card is held, with at most max_legit_hold_rate of
    legitimate cards wrongly held.

    Rows must be in time order within each card, oldest first (the loader
    returns them that way). A card is attacked if any of its rows has
    y_true == 1, legitimate otherwise. Holding rule: a card is held at its
    first row with score >= t, and that row and everything after it is
    blocked.

    Choose t: candidates are every distinct score plus +inf. The legitimate
    hold rate at t is the share of legitimate cards whose maximum score is
    >= t. Take the smallest t with rate <= max_legit_hold_rate.

    Then for each attacked card, count its fraud rows that come before the
    hold row (all of them if the card is never held). Return the median of
    those counts over attacked cards. No attacked cards -> float('nan').
    """
    raise NotImplementedError(_TODO.format(name="median_frauds_before_hold"))


# ---------------------------------------------------------------------------
# Track 4: fairness check
# ---------------------------------------------------------------------------

def fpr_ratio(y_true, flagged, group):
    """FPR in the group / FPR outside it, from binary decisions.

    flagged is 0/1 or bool (the decision, after you've picked a threshold);
    group is bool, True for the group of interest (customer_age > 50). FPR in
    a group = flagged legitimate rows in it / legitimate rows in it.

    Return fpr(group) / fpr(not group). Closer to 1 is fairer.
    Either group has no legitimate rows -> float('nan').
    fpr(not group) is 0 -> 1.0 if fpr(group) is also 0, else float('inf').
    """
    raise NotImplementedError(_TODO.format(name="fpr_ratio"))


# ---------------------------------------------------------------------------
# Track 6: scoring latency
# ---------------------------------------------------------------------------

def p95_latency_ms(seconds):
    """95th percentile of a list of per-link scoring times, in milliseconds.

    seconds: times in seconds, one per scored link (time.perf_counter()
    before and after a single predict() call). Use numpy's default percentile
    (linear interpolation). Empty -> float('nan').
    """
    raise NotImplementedError(_TODO.format(name="p95_latency_ms"))


# ---------------------------------------------------------------------------
# Track 7: the analyst's top-k alerts
# ---------------------------------------------------------------------------

def precision_at_k(y_true, scores, k=100):
    """Share of the k highest-scoring rows that are fraud.

    Order rows by score, highest first, with ties broken by position: the
    row that comes first in the input wins (np.argsort(-scores,
    kind="stable")). If there are fewer than k rows, use all of them and
    divide by that count. Empty input -> float('nan').
    """
    raise NotImplementedError(_TODO.format(name="precision_at_k"))


def recall_at_k(y_true, scores, k=100):
    """Share of all fraud that lands in the k highest-scoring rows.

    Same ordering and tie rule as precision_at_k. No fraud rows -> float('nan').
    """
    raise NotImplementedError(_TODO.format(name="recall_at_k"))


# ---------------------------------------------------------------------------
# Track 9: F1 on the illicit class
# ---------------------------------------------------------------------------

def f1_illicit(y_true, y_pred):
    """F1 for the illicit class, from binary predictions.

    y_true: 1 illicit, 0 licit, -1 unknown. Unknown rows are ignored.
    y_pred: 1 = predicted illicit.
    F1 = 2 * tp / (2 * tp + fp + fn). If that denominator is 0 (no illicit
    rows and nothing predicted illicit) -> float('nan').
    """
    raise NotImplementedError(_TODO.format(name="f1_illicit"))


def f1_per_step(y_true, y_pred, steps):
    """{time_step: f1_illicit over that step's rows}, with steps in ascending order.

    Steps with no labelled rows at all are left out. A step with labelled rows
    but nothing illicit and nothing predicted gets nan, from f1_illicit.
    """
    raise NotImplementedError(_TODO.format(name="f1_per_step"))
