"""Download, load and split the data for every track.

This file is provided and tested. Don't edit it; if something here is wrong
for your track, tell your TPM so the fix reaches every team.

Each track has one loader, `load_trackNN()`, that returns the fixed test
split for that track: `(train_pool, test)` in most cases. The test set is
not to be touched until Deliverable 3. Your validation set comes out of the
train pool, and that part is yours (D1.2).

Every loader looks in `data/` first and only downloads when the files aren't
there. Kaggle downloads need KAGGLE_USERNAME and KAGGLE_KEY in the
environment; the starter notebooks read them from Colab Secrets.

Memory: the two big files (Sparkov train, 1.3M rows; PaySim, 6.3M rows) are
read with `usecols` and small dtypes. A full default `read_csv` on either can
eat most of what a free Colab session has.
"""

from __future__ import annotations

import io
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import config

SEED = config.SEED


# --------------------------------------------------------------------------
# Finding and downloading files
# --------------------------------------------------------------------------

KAGGLE_SLUGS = {
    "sparkov": "kartik2112/fraud-detection",
    "paysim": "ealaxi/paysim1",
    "baf": "sgpjesus/bank-account-fraud-dataset-neurips-2022",
    "ulb": "mlg-ulb/creditcardfraud",
    "elliptic": "ellipticco/elliptic-data-set",
}

PHIUSIIL_ZIP_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
PHIUSIIL_CSV = "PhiUSIIL_Phishing_URL_Dataset.csv"


def _dataset_dir(data_dir, name) -> Path:
    d = Path(data_dir if data_dir is not None else config.DATA_DIR) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _find_local(folder: Path, pattern: str):
    """First file under `folder` matching a glob pattern, or None."""
    hits = sorted(folder.rglob(pattern))
    return hits[0] if hits else None


def _kaggle_file(data_dir, dataset: str, pattern: str) -> Path:
    """Path to one file of a Kaggle dataset, downloading the dataset if needed.

    Looks under data/<dataset>/ first. On a miss it calls kagglehub, which
    caches the download in the home folder, and copies the matching file into
    data/<dataset>/ so the next call is a local read.
    """
    folder = _dataset_dir(data_dir, dataset)
    hit = _find_local(folder, pattern)
    if hit is not None:
        return hit

    import kagglehub  # imported here so the tests don't need it

    print(f"Downloading {KAGGLE_SLUGS[dataset]} from Kaggle (first time only)...")
    cache = Path(kagglehub.dataset_download(KAGGLE_SLUGS[dataset]))
    hit = _find_local(cache, pattern)
    if hit is None:
        raise FileNotFoundError(
            f"Downloaded {KAGGLE_SLUGS[dataset]} but found no file matching {pattern!r} "
            f"under {cache}. Files there: {[p.name for p in cache.rglob('*') if p.is_file()][:20]}"
        )
    dest = folder / hit.name
    shutil.copy2(hit, dest)
    return dest


def _phiusiil_file(data_dir) -> Path:
    """The PhiUSIIL CSV, downloaded from UCI as a zip on first use."""
    folder = _dataset_dir(data_dir, "phiusiil")
    hit = _find_local(folder, PHIUSIIL_CSV)
    if hit is not None:
        return hit
    print("Downloading the PhiUSIIL dataset from UCI (first time only, about 15 MB)...")
    with urllib.request.urlopen(PHIUSIIL_ZIP_URL, timeout=120) as r:
        payload = r.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if not names:
            raise FileNotFoundError(f"No CSV inside the UCI zip; contents: {z.namelist()}")
        z.extract(names[0], folder)
    hit = _find_local(folder, "*.csv")
    return hit


def _downcast_floats(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32 and int64 -> int32 where the values fit. Halves the
    memory of a numeric table and changes nothing a model can see."""
    for col in df.select_dtypes(include="float64").columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include="int64").columns:
        if df[col].min() >= -2**31 and df[col].max() < 2**31:
            df[col] = df[col].astype("int32")
    return df


# --------------------------------------------------------------------------
# Quick look, used by every starter notebook
# --------------------------------------------------------------------------

def quick_look(df: pd.DataFrame, label_col: str, name: str = "data") -> None:
    """Print shape, columns, memory and the fraud count and rate. No plots here;
    plots are Deliverable 1's exploration section, and they're yours."""
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"{name}: {len(df):,} rows x {df.shape[1]} columns, {mem_mb:,.0f} MB in memory")
    print("columns:", ", ".join(df.columns))
    labels = df[label_col]
    if labels.isin([0, 1, -1]).all() and (labels == -1).any():
        labelled = labels != -1
        print(f"labelled rows: {labelled.sum():,} of {len(df):,}; unknown: {(~labelled).sum():,}")
        labels = labels[labelled]
    n_pos = int((labels == 1).sum())
    print(f"{label_col} == 1: {n_pos:,} of {len(labels):,} ({100 * n_pos / max(len(labels), 1):.3f}%)")


# --------------------------------------------------------------------------
# Tracks 1 and 2: Sparkov card transactions
# --------------------------------------------------------------------------

# Names and street addresses are dropped: they're synthetic, identify nobody,
# and no feature should come from them. The row-number column is dropped too.
SPARKOV_DTYPES = {
    "cc_num": str,            # an ID, not a number; keep it as text
    "merchant": "category",
    "category": "category",
    "amt": "float32",
    "gender": "category",
    "city": "category",
    "state": "category",
    "zip": "int32",
    "lat": "float32",
    "long": "float32",
    "city_pop": "int32",
    "job": "category",
    "dob": str,
    "trans_num": str,         # unique transaction ID; useful for leakage checks
    "unix_time": "int64",
    "merch_lat": "float32",
    "merch_long": "float32",
    "is_fraud": "int8",
}
SPARKOV_COLS = ["trans_date_trans_time", *SPARKOV_DTYPES]


def _read_sparkov(path: Path, nrows=None) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=SPARKOV_COLS, dtype=SPARKOV_DTYPES, nrows=nrows)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    # Oldest first. Several helpers assume this order.
    return df.sort_values(["unix_time", "trans_num"], kind="stable").reset_index(drop=True)


def load_track01(data_dir=None, nrows=None):
    """Track 1. Returns (train_pool, test).

    train_pool is fraudTrain.csv (2019-01 to 2020-06) and test is fraudTest.csv
    (2020-06 to 2020-12). The files are already split by time, so this is a
    train-on-the-past, test-on-the-future split. Carve your validation set
    from the end of train_pool, not at random.

    nrows limits how much of each file is read, for quick experiments only.
    """
    train = _read_sparkov(_kaggle_file(data_dir, "sparkov", "fraudTrain.csv"), nrows)
    test = _read_sparkov(_kaggle_file(data_dir, "sparkov", "fraudTest.csv"), nrows)
    return train, test


def load_track02(data_dir=None, nrows=None):
    """Track 2. Same files and split as Track 1; see make_sequences() for the
    per-card windows."""
    return load_track01(data_dir, nrows)


def make_sequences(df: pd.DataFrame, n: int = 10, feature_cols=("amt",),
                   card_col: str = "cc_num", time_col: str = "unix_time",
                   label_col: str = "is_fraud"):
    """Turn a transactions table into one fixed-length window per transaction.

    For each transaction, the window holds that card's last `n` transactions
    ending at it, oldest first. A card's early transactions, before it has
    `n` of them, are padded with zeros at the front of the window.

    Returns:
        X:    float32 array of shape (rows, n, len(feature_cols))
        y:    int8 array of shape (rows,), the label of the last transaction in each window
        meta: DataFrame aligned with X: card, time, label and n_valid (how many
              of the n rows are real transactions rather than padding)

    Only numeric feature columns work here. Encode categoricals first (D1.4).
    The output is roughly rows * n * features * 4 bytes: 1.3M rows, n=10 and
    5 features is about 260 MB. Keep n small while you're experimenting.
    """
    feature_cols = list(feature_cols)
    df = df.sort_values([card_col, time_col], kind="stable")
    values = df[feature_cols].to_numpy(dtype="float32")
    labels = df[label_col].to_numpy(dtype="int8")
    cards = df[card_col].to_numpy()

    # Boundaries where the card changes, so each card can be windowed on its own.
    change = np.flatnonzero(cards[1:] != cards[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(df)]])

    windows, n_valid = [], []
    pad = np.zeros((n - 1, len(feature_cols)), dtype="float32")
    for s, e in zip(starts, ends):
        padded = np.vstack([pad, values[s:e]])
        # sliding_window_view gives (rows, features, n); move n to the middle.
        w = np.lib.stride_tricks.sliding_window_view(padded, (n, len(feature_cols)))[:, 0]
        windows.append(w)
        n_valid.append(np.minimum(np.arange(1, e - s + 1), n))

    X = np.concatenate(windows).astype("float32")
    meta = pd.DataFrame({
        card_col: cards,
        time_col: df[time_col].to_numpy(),
        label_col: labels,
        "n_valid": np.concatenate(n_valid).astype("int16"),
    }, index=df.index)
    return X, labels, meta


# --------------------------------------------------------------------------
# Track 3: PaySim transfers
# --------------------------------------------------------------------------

PAYSIM_DTYPES = {
    "step": "int16",
    "type": "category",
    "amount": "float32",
    "nameOrig": str,
    "oldbalanceOrg": "float32",   # yes, "Org"; that's how the file spells it
    "newbalanceOrig": "float32",
    "nameDest": str,
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}

# Test = this step and later. Steps are hours over 30 days (1 to 743). The cut
# sits late enough to leave plenty of fraud in the test period and early enough
# that legitimate traffic is still dense; volume thins out in the last third.
TRACK03_TEST_FROM_STEP = 300


def load_track03(data_dir=None, types=("TRANSFER", "CASH_OUT"), keep_names=True, nrows=None):
    """Track 3. Returns (train_pool, test), split by `step`.

    types: which transaction types to keep. Fraud only ever occurs in TRANSFER
        and CASH_OUT, so those two are the default (about 2.8M of the 6.3M
        rows). Pass None to keep everything.
    keep_names: nameOrig and nameDest are account IDs stored as text and cost
        about 400 MB each. They're needed for leakage checks and for any
        feature that looks at an account's history; set False to save memory
        once you know you don't need them.

    A warning that matters for D1.4: the four balance columns are written
    after the transaction, and the simulator cancels transactions it flags, so
    for many frauds the balances don't move. A feature built on them looks
    brilliant and would be useless in production, where you score before the
    transaction happens.
    """
    path = _kaggle_file(data_dir, "paysim", "*.csv")
    usecols = list(PAYSIM_DTYPES)
    if not keep_names:
        usecols = [c for c in usecols if c not in ("nameOrig", "nameDest")]
    df = pd.read_csv(path, usecols=usecols, dtype=PAYSIM_DTYPES, nrows=nrows)
    if types is not None:
        df = df[df["type"].isin(list(types))].copy()
        df["type"] = df["type"].cat.remove_unused_categories()
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    test_mask = df["step"] >= TRACK03_TEST_FROM_STEP
    return df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


# --------------------------------------------------------------------------
# Tracks 4 and 10: Bank Account Fraud (Base.csv)
# --------------------------------------------------------------------------

BAF_DTYPES = {
    "fraud_bool": "int8",
    "month": "int8",
    "customer_age": "int16",
    "payment_type": "category",
    "employment_status": "category",
    "housing_status": "category",
    "source": "category",
    "device_os": "category",
}


def _read_baf(data_dir, nrows=None) -> pd.DataFrame:
    path = _kaggle_file(data_dir, "baf", "Base.csv")
    df = pd.read_csv(path, dtype=BAF_DTYPES, nrows=nrows)
    df = _downcast_floats(df)
    return df.sort_values("month", kind="stable").reset_index(drop=True)


def load_track04(data_dir=None, nrows=None):
    """Track 4. Returns (train_pool, test): months 0 to 5, and months 6 and 7.

    `month` is the only time column. customer_age is in ten-year bins (20
    means 20 to 29), which matters for the fairness part of the metric.
    """
    df = _read_baf(data_dir, nrows)
    test_mask = df["month"] >= 6
    return df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


TRACK10_FIRST_TRAIN_MONTHS = 3


def load_track10(data_dir=None, nrows=None):
    """Track 10. Returns (train_pool, deployment) for the month-by-month loop.

    train_pool is months 0 to 2. deployment is a dict {month: DataFrame} for
    months 3 to 7, in order. The starter notebook loops over deployment,
    scoring one month at a time, the way a deployed model meets new data. No
    row from a deployment month may be used to train until that month has
    been scored.
    """
    df = _read_baf(data_dir, nrows)
    train = df[df["month"] < TRACK10_FIRST_TRAIN_MONTHS].reset_index(drop=True)
    deployment = {
        int(m): df[df["month"] == m].reset_index(drop=True)
        for m in sorted(df["month"].unique()) if m >= TRACK10_FIRST_TRAIN_MONTHS
    }
    return train, deployment


# --------------------------------------------------------------------------
# Track 6: PhiUSIIL phishing URLs
# --------------------------------------------------------------------------

_SCHEME = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.IGNORECASE)


def extract_hostname(url: str) -> str:
    """'https://www.Example.com:443/login?x=1' -> 'example.com'.

    Lowercases, drops the scheme, anything after the first '/', '?' or '#',
    a user@ prefix, a port, and a leading 'www.'. The result is what a model
    for this track sees; the full URL gives the answer away (see load_track06).
    """
    s = str(url).strip().lower()
    s = _SCHEME.sub("", s)
    s = re.split(r"[/?#]", s, maxsplit=1)[0]
    if "@" in s:
        s = s.rsplit("@", 1)[1]
    if s.count(":") == 1 and s.rsplit(":", 1)[1].isdigit():
        s = s.rsplit(":", 1)[0]
    if s.startswith("www."):
        s = s[4:]
    return s.strip(".")


def clean_hostnames(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw PhiUSIIL rows -> one row per hostname with a single label.

    The raw label is 1 for legitimate and 0 for phishing. It's flipped here so
    that is_phishing == 1 means phishing, like every other track's label.
    Hostnames that appear with both labels (link shorteners, mostly) are
    dropped: the hostname alone can't decide them.
    """
    df = pd.DataFrame({
        "hostname": raw["URL"].map(extract_hostname),
        "is_phishing": (1 - raw["label"].astype("int8")).astype("int8"),
    })
    df = df[df["hostname"] != ""]
    labels_per_host = df.groupby("hostname")["is_phishing"].nunique()
    single = labels_per_host.index[labels_per_host == 1]
    df = df[df["hostname"].isin(single)].drop_duplicates("hostname")
    return df.sort_values("hostname").reset_index(drop=True)


def load_track06(data_dir=None, test_size: float = 0.15):
    """Track 6. Returns (train_pool, test): hostname and is_phishing only.

    Why hostnames: every legitimate URL in the raw file is written as a bare
    'https://www.<domain>' homepage, and most phishing URLs aren't. A model
    on full URLs learns that formatting. So the URL is reduced to its
    hostname, duplicates are collapsed, and ambiguous hostnames dropped.

    There are no dates, so the test set is a random 15%, stratified so that
    the phishing share matches the pool. Your validation set comes out of the
    85% the same way (D1.2). No hostname is in two parts.
    """
    from sklearn.model_selection import train_test_split

    raw = pd.read_csv(_phiusiil_file(data_dir), usecols=["URL", "label"])
    df = clean_hostnames(raw)
    pool, test = train_test_split(
        df, test_size=test_size, stratify=df["is_phishing"], random_state=SEED
    )
    return pool.reset_index(drop=True), test.reset_index(drop=True)


# --------------------------------------------------------------------------
# Track 7: ULB card transactions, two days
# --------------------------------------------------------------------------

ULB_DTYPES = {**{f"V{i}": "float32" for i in range(1, 29)},
              "Time": "float32", "Amount": "float32", "Class": "int8"}
SECONDS_PER_DAY = 86_400


def load_track07(data_dir=None):
    """Track 7. Returns (train_pool, test): day 1 and day 2, split on Time.

    Time is seconds since the first transaction, over two days. The labels
    (Class) are for scoring only on this track: the system ranks transactions
    without training on them.
    """
    path = _kaggle_file(data_dir, "ulb", "creditcard.csv")
    df = pd.read_csv(path, dtype=ULB_DTYPES)
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    test_mask = df["Time"] >= SECONDS_PER_DAY
    return df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


# --------------------------------------------------------------------------
# Track 9: Elliptic transaction graph
# --------------------------------------------------------------------------

N_ELLIPTIC_FEATURES = 165
TRACK09_TEST_FROM_STEP = 35


def load_track09(data_dir=None):
    """Track 9. Returns (train_pool, test, edges).

    The three files are joined into one table: txId, time_step, feat_0 to
    feat_164, and label (1 illicit, 0 licit, -1 unknown). Train pool is time
    steps 1 to 34, test is 35 to 49. Both keep the unknown rows: the metric
    uses labelled transactions only, but neighbour features need every node.
    edges is the full edge list (txId1 -> txId2) for building those features.
    """
    feats_path = _kaggle_file(data_dir, "elliptic", "elliptic_txs_features.csv")
    classes_path = _kaggle_file(data_dir, "elliptic", "elliptic_txs_classes.csv")
    edges_path = _kaggle_file(data_dir, "elliptic", "elliptic_txs_edgelist.csv")

    dtypes = {0: "int64", 1: "int16", **{i: "float32" for i in range(2, N_ELLIPTIC_FEATURES + 2)}}
    feats = pd.read_csv(feats_path, header=None, dtype=dtypes)
    feats.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(N_ELLIPTIC_FEATURES)]

    classes = pd.read_csv(classes_path, dtype={"txId": "int64", "class": str})
    classes["label"] = classes["class"].map({"1": 1, "2": 0, "unknown": -1}).astype("int8")

    df = feats.merge(classes[["txId", "label"]], on="txId", how="left")
    df["label"] = df["label"].fillna(-1).astype("int8")
    df = df.sort_values(["time_step", "txId"], kind="stable").reset_index(drop=True)

    edges = pd.read_csv(edges_path, dtype={"txId1": "int64", "txId2": "int64"})

    test_mask = df["time_step"] >= TRACK09_TEST_FROM_STEP
    return df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True), edges


# --------------------------------------------------------------------------
# One entry point, used by the tests and app.py
# --------------------------------------------------------------------------

LOADERS = {
    1: load_track01, 2: load_track02, 3: load_track03, 4: load_track04,
    6: load_track06, 7: load_track07, 9: load_track09, 10: load_track10,
}

# Which column is the label, which identifies a row or entity, and which
# carries time, per track. Used by the leakage test and the starters.
TRACK_COLUMNS = {
    1: {"label": "is_fraud", "id": "trans_num", "time": "unix_time"},
    2: {"label": "is_fraud", "id": "trans_num", "time": "unix_time"},
    3: {"label": "isFraud", "id": None, "time": "step"},
    4: {"label": "fraud_bool", "id": None, "time": "month"},
    6: {"label": "is_phishing", "id": "hostname", "time": None},
    7: {"label": "Class", "id": None, "time": "Time"},
    9: {"label": "label", "id": "txId", "time": "time_step"},
    10: {"label": "fraud_bool", "id": None, "time": "month"},
}


def load_track(track: int | None = None, data_dir=None, **kwargs):
    """Call the loader for `track` (default: config.TRACK)."""
    track = config.require_track() if track is None else track
    return LOADERS[track](data_dir=data_dir, **kwargs)
