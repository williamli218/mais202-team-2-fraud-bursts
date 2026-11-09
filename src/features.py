"""Features (D1.4). One worked example is here to show the shape of a feature
function; the three features your proposal needs are yours to write.

Rules that the example follows and yours must too:

  1. A feature for one row may only use information that existed when that
     row happened. Past transactions on the same card: fine. The next
     transaction, or anything computed over the whole file: not fine.
  2. Anything learned from data (an average, a count per merchant, a
     vocabulary) is computed on the training pool only, then applied to
     validation and test. Compute it on all rows and you've used the test set.
  3. One function per feature, taking a DataFrame and returning a Series
     aligned with it. Small functions are easy to test and easy to explain
     in the report.
"""

import numpy as np
import pandas as pd


def example_amount_is_round(df: pd.DataFrame, amount_col: str = "amt") -> pd.Series:
    """1 if the amount is a whole number of dollars, else 0.

    The example, not a feature you should rely on. It shows the shape: one
    row in, one value out, nothing looked up from other rows, so there is no
    way for it to see the future.
    """
    amount = df[amount_col].astype("float64")
    return (np.isclose(amount, np.round(amount))).astype("int8")


# TODO D1.4: write at least three features of your own below, one function
# each, and say in the report why each should help and why it can't see the
# future. If your track works with text (Track 6), write the function that
# turns a hostname into model input here instead, and explain what it keeps
# and what it throws away.


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the feature table your model trains on.

    Call your feature functions here so that the notebook, the tests and
    predict() all build features the same way. Start with the example and
    replace it as you go.
    """
    out = pd.DataFrame(index=df.index)
    # TODO D1.4: replace the example with your features.
    if "amt" in df.columns:
        out["amount_is_round"] = example_amount_is_round(df)
    return out
