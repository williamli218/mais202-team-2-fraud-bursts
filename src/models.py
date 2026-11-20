"""Models. The trivial baseline is provided; everything else is yours (D2).

The trivial baseline calls every row legitimate. Its purpose is to be scored
with your metric and with accuracy, side by side, so that the gap is on the
page before you train anything real (D1.5).
"""

import numpy as np


def always_legitimate(n_rows: int) -> np.ndarray:
    """Scores for the trivial baseline: 0.0 for every row.

    With every score tied at 0, no threshold can separate fraud from
    legitimate. Under your track's metric that reads as "catches nothing"
    (or, on Track 1, "blocks everything to catch 85%"). Under accuracy it
    reads as 99% on most tracks. Both are correct, and that's the point.
    """
    return np.zeros(n_rows, dtype="float32")


# TODO D2.1: a simple baseline model (logistic regression, a small tree, or
# a rule), trained on your training split, scored on validation. Save it
# small enough to commit under models/ if the demo will use it.
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_logistic_regression(X_train, y_train):
  model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
  )
  model.fit(X_train, y_train)
  return model

# TODO D2.2: your main model. Put training in a function so a teammate can
# rerun it from a notebook, and log every run to experiments/log.csv.
