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
def card_time_order(df):
  d = pd.DataFrame({"cc_num": df["cc_num"],
  "time": pd.to_datetime(df["trans_date_trans_time"]),
  "amount": df["amt"].astype("float64")}, index=df.index)

  return d.sort_values(["cc_num", "time"], kind="stable")

def seconds_since_prev_transaction(df):
  """
  Log of seconds since this card's previous transaction.
  Bursts mean small gaps. First transaction on a card gets 30 days.
  """
  d = card_time_order(df)
  gap = d.groupby("cc_num", observed=True)["time"].diff().dt.total_seconds()
  gap = gap.fillna(30 * 24 * 3600).clip(lower=0)
  return np.log1p(gap).astype("float32").reindex(df.index)

def transaction_count_past_24h(df):
  """
  Number of transactions on this card in the 24 hours up to and including this one.
  """
  d = card_time_order(df)
  count = d.groupby("cc_num", observed=True).rolling("24h", on="time")["amount"].count()
  return pd.Series(count.to_numpy(), index=d.index).astype("float32").reindex(df.index)

def spend_past_24h(df):
  """
  Total dollars spent on this card in the 24 hours up to and including this one.
  """
  d = card_time_order(df)
  s = d.groupby("cc_num", observed=True).rolling("24h", on="time")["amount"].sum()
  return pd.Series(s.to_numpy(), index=d.index).astype("float32").reindex(df.index)

def hour_of_day(df: pd.DataFrame) -> pd.Series:
    """
    Hour of the transaction, 0-23. Uses only the row itself.
    """
    return pd.to_datetime(df["trans_date_trans_time"]).dt.hour.astype("int8")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the feature table your model trains on.

    Call your feature functions here so that the notebook, the tests and
    predict() all build features the same way. Start with the example and
    replace it as you go.
    """
    out = pd.DataFrame(index=df.index)
    # TODO D1.4: replace the example with your features.
    out["log_seconds_since_prev"] = seconds_since_prev_transaction(df)
    out["transaction_count_past_24h"] = transaction_count_past_24h(df)
    out["spend_past_24h"] = spend_past_24h(df)
    out["hour_of_day"] = hour_of_day(df)
    out["amt"] = df["amt"].astype("float32")
    return out
