"""Models. The trivial baseline is provided; everything else is yours (D2).

The trivial baseline calls every row legitimate. Its purpose is to be scored
with your metric and with accuracy, side by side, so that the gap is on the
page before you train anything real (D1.5).
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import config
import json


def always_legitimate(n_rows: int) -> np.ndarray:
    """Scores for the trivial baseline: 0.0 for every row.

    With every score tied at 0, no threshold can separate fraud from
    legitimate. Under your track's metric that reads as "catches nothing"
    (or, on Track 1, "blocks everything to catch 85%"). Under accuracy it
    reads as 99% on most tracks. Both are correct, and that's the point.
    """
    return np.zeros(n_rows, dtype="float32")


def train_logistic_regression(X_train, y_train):
  model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.SEED),
  )
  model.fit(X_train, y_train)
  return model


def load_logistic_regression(path="models/logistic_regression.json"):
  with open(path) as f:
    return json.load(f)


def logistic_regression_scores(params, X):
  """
  Fraud probability per row from saved logistic regression parameters.
  """
  X = X[params["features"]].to_numpy(dtype=float)
  z = (X - np.array(params["mean"])) / np.array(params["scale"])
  logit = z @ np.array(params["coef"]) + params["intercept"]
  return 1 / (1 + np.exp(-logit))


def smooth_scores(scores, card_ids, k=3):
  """
  Card-level score: mean of the card's last k transaction scores, the
  current one included. Rows must be in time order within each card. Uses
  only current and past scores, so it can't see the future.
  """
  v = pd.DataFrame({"card": np.asarray(card_ids), "s": np.asarray(scores, dtype=float)})
  return v.groupby("card", sort=False)["s"].transform(
    lambda x: x.rolling(k, min_periods=1).mean()
  ).to_numpy()
  
# TODO D2.2: your main model. Put training in a function so a teammate can
# rerun it from a notebook, and log every run to experiments/log.csv.
import torch
from torch import nn


class FraudLSTM(nn.Module):
  """Reads a card's last n transactions (oldest first, padding at the start)
  and returns one logit: how likely the newest transaction is fraud."""

  def __init__(self, n_features, hidden_dim=32, num_layers=1):
    super().__init__()
    self.lstm = nn.LSTM(n_features, hidden_dim, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_dim, 1)

  def forward(self, x):                        # x: (batch, n, features)
    lstm_out, _ = self.lstm(x)                 # hidden state starts at 0 for every window
    return self.fc(lstm_out[:, -1, :]).squeeze(-1)   # last step only -> one logit per window

import copy
from torch.utils.data import TensorDataset, DataLoader


def train_lstm(X_tr, y_tr, X_va, y_va, hidden_dim=32, learning_rate=1e-3,
               batch_size=2048, max_epochs=15, patience=3):
  """Train FraudLSTM with early stopping on validation loss.
  Class imbalance: pos_weight = (#legit / #fraud) in the loss.
  Returns (model with the best weights, history of losses per epoch)."""
  torch.manual_seed(config.SEED)
  np.random.seed(config.SEED)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = FraudLSTM(X_tr.shape[2], hidden_dim).to(device)

  n_pos = int(y_tr.sum())
  pos_weight = torch.tensor((len(y_tr) - n_pos) / n_pos, dtype=torch.float32, device=device)
  criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)   # sigmoid + weighted BCE in one
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype("float32"))),
    batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(config.SEED))
  valid_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va.astype("float32"))),
    batch_size=8192)

  best_loss, best_state, patience_counter, history = float("inf"), None, 0, []
  for epoch in range(max_epochs):
    model.train()
    total = 0.0
    for inputs, labels in train_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()
      loss = criterion(model(inputs), labels)
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
      optimizer.step()
      total += loss.item() * len(labels)
    train_loss = total / len(y_tr)

    model.eval()
    total = 0.0
    with torch.no_grad():
      for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        total += criterion(model(inputs), labels).item() * len(labels)
    valid_loss = total / len(y_va)

    history.append({"epoch": epoch + 1, "train_loss": train_loss, "valid_loss": valid_loss})
    print(f"Epoch {epoch + 1}/{max_epochs}, train loss {train_loss:.4f}, valid loss {valid_loss:.4f}")

    if valid_loss < best_loss:
      best_loss, best_state, patience_counter = valid_loss, copy.deepcopy(model.state_dict()), 0
    else:
      patience_counter += 1
      if patience_counter >= patience:
        print(f"Early stopping: no improvement for {patience} epochs.")
        break

  model.load_state_dict(best_state)
  return model, history

def lstm_scores(model, X, batch_size=8192):
  """Fraud probability for each window (its newest transaction)."""
  model.eval()
  device = next(model.parameters()).device
  out = []
  with torch.no_grad():
    for i in range(0, len(X), batch_size):
      xb = torch.from_numpy(X[i:i + batch_size]).to(device)
      out.append(torch.sigmoid(model(xb)).cpu().numpy())
  return np.concatenate(out)