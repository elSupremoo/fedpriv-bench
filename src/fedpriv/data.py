"""Loading and leakage-safe splitting of the MSU/ORNL power-system attack data.

The "three-class" release (triple.7z) contains 15 CSV files (data1..data15).
Each file is one of the 15 independently generated scenario sets described
in the dataset README, each a 1% random sample of the same 37 scenarios.
Rows inside a file are i.i.d. snapshots (128 PMU / log features), sorted by
class, NOT a contiguous time series, so sliding windows are not meaningful.

Leakage control: we split by *file* (scenario set), never by row. Rows from
the same simulated event run are highly correlated, so a random row split
lets near-duplicates land on both sides. All preprocessing statistics
(imputation medians, standardisation) are fit on the training files only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["Attack", "Natural", "NoEvents"]
LABEL_COL = "marker"

# Default split by scenario-set file id (1..15).
TRAIN_FILES = tuple(range(1, 11))  # data1..data10
VAL_FILES = (11, 12)  # used only for hyper-parameter choices (FedProx mu)
TEST_FILES = (13, 14, 15)  # never used for any decision


@dataclass
class Split:
    X: np.ndarray  # float32 [n, d]
    y: np.ndarray  # int64 [n]
    group: np.ndarray  # int64 [n], source file id


@dataclass
class Dataset:
    train: Split
    val: Split
    test: Split
    feature_names: list[str]


def load_files(data_dir: str | Path, file_ids) -> pd.DataFrame:
    frames, groups = [], []
    for i in file_ids:
        df = pd.read_csv(Path(data_dir) / f"data{i}.csv")
        frames.append(df)
        groups.append(np.full(len(df), i, dtype=np.int64))
    out = pd.concat(frames, ignore_index=True)
    return pd.concat([out, pd.Series(np.concatenate(groups), name="__group")], axis=1)


def _to_xy(df: pd.DataFrame):
    y = df[LABEL_COL].map({c: k for k, c in enumerate(CLASSES)})
    if y.isna().any():
        raise ValueError(f"unexpected labels: {set(df[LABEL_COL]) - set(CLASSES)}")
    X = df.drop(columns=[LABEL_COL, "__group"]).to_numpy(dtype=np.float64)
    X[~np.isfinite(X)] = np.nan  # impedance columns contain +/-inf
    return X, y.to_numpy(np.int64).copy(), df["__group"].to_numpy(np.int64).copy()


class Preprocessor:
    """Median imputation + standardisation, fit on training rows only."""

    def fit(self, X: np.ndarray) -> "Preprocessor":
        self.median = np.nanmedian(X, axis=0)
        self.median = np.where(np.isnan(self.median), 0.0, self.median)
        Xi = np.where(np.isnan(X), self.median, X)
        self.mean = Xi.mean(axis=0)
        self.std = Xi.std(axis=0)
        self.std[self.std < 1e-8] = 1.0  # constant columns
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xi = np.where(np.isnan(X), self.median, X)
        Z = (Xi - self.mean) / self.std
        # A few raw values are extreme; clip so no single feature dominates.
        return np.clip(Z, -10.0, 10.0).astype(np.float32)


def load_dataset(
    data_dir: str | Path,
    train_files=TRAIN_FILES,
    val_files=VAL_FILES,
    test_files=TEST_FILES,
) -> Dataset:
    assert not set(train_files) & set(test_files), "train/test overlap"
    assert not set(val_files) & set(test_files), "val/test overlap"
    assert not set(train_files) & set(val_files), "train/val overlap"
    parts = {}
    names = None
    for name, ids in (("train", train_files), ("val", val_files), ("test", test_files)):
        if not len(ids):  # no validation files requested: evaluate on test only
            continue
        df = load_files(data_dir, ids)
        names = [c for c in df.columns if c not in (LABEL_COL, "__group")]
        parts[name] = _to_xy(df)
    pre = Preprocessor().fit(parts["train"][0])
    out = {k: Split(pre.transform(X), y, g) for k, (X, y, g) in parts.items()}
    return Dataset(out["train"], out.get("val", out["test"]), out["test"], names)


def random_row_split(data_dir: str | Path, file_ids, test_frac: float, seed: int) -> Dataset:
    """The internship-style split (random rows), kept ONLY for the leakage
    comparison in scripts/leakage_check.py. Do not use for headline results."""
    df = load_files(data_dir, file_ids)
    X, y, g = _to_xy(df)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    n_test = int(round(test_frac * len(y)))
    te, tr = perm[:n_test], perm[n_test:]
    pre = Preprocessor().fit(X[tr])
    tr_s = Split(pre.transform(X[tr]), y[tr], g[tr])
    te_s = Split(pre.transform(X[te]), y[te], g[te])
    names = [c for c in df.columns if c not in (LABEL_COL, "__group")]
    return Dataset(tr_s, te_s, te_s, names)
