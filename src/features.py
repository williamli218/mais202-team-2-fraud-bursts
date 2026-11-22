"""Features (D1.4, extended in D2). Each feature uses only the current row
and the same card's earlier rows, so none can see the future, and nothing
is learned across the dataset.
"""

import numpy as np
import pandas as pd


def example_amount_is_round(df: pd.DataFrame, amount_col: str = "amt") -> pd.Series:
    """1 if the amount is a whole number of dollars, else 0. (Template example.)"""
    amount = df[amount_col].astype("float64")
    return (np.isclose(amount, np.round(amount))).astype("int8")


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

def is_night(df):
  """
  1 if the charge happens between 22:00 and 04:59, else 0. Uses only the row itself.
  """
  hour = pd.to_datetime(df["trans_date_trans_time"]).dt.hour
  return ((hour >= 22) | (hour <= 4)).astype("int8")

def amount_vs_recent_mean(df, n=10):
  """
  This amount divided by the mean of the card's previous n amounts.
  shift(1) leaves out the current charge, so only the past is used.
  A card's first transaction gets 1.0.
  """
  d = card_time_order(df)
  prev_mean = d.groupby("cc_num", observed=True)["amount"].transform(
    lambda s: s.shift(1).rolling(n, min_periods=1).mean()
  )
  return (d["amount"] / prev_mean).fillna(1.0).astype("float32").reindex(df.index)

CATEGORIES = [
  "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
  "health_fitness", "home", "kids_pets", "misc_net", "misc_pos",
  "personal_care", "shopping_net", "shopping_pos", "travel",
]

def category_one_hot(df):
  """
  One 0/1 column per merchant category. The list is fixed (Sparkov's 14
  categories), not learned from the data, so validation and test get exactly
  the same columns. Uses only the row itself.
  """
  cat = df["category"].astype(str)
  return pd.DataFrame(
    {f"cat_{c}": (cat == c).astype("int8") for c in CATEGORIES}, index=df.index
  )

def build_features(df: pd.DataFrame) -> pd.DataFrame:
  """Assemble the feature table your model trains on."""
  out = pd.DataFrame(index=df.index)
  out["log_seconds_since_prev"] = seconds_since_prev_transaction(df)
  out["transaction_count_past_24h"] = transaction_count_past_24h(df)
  out["log_spend_past_24h"] = np.log1p(spend_past_24h(df))
  out["log_amt"] = np.log1p(df["amt"].astype("float32"))
  out["amount_vs_recent_mean"] = amount_vs_recent_mean(df)
  out["is_night"] = is_night(df)
  out = out.join(category_one_hot(df))
  return out