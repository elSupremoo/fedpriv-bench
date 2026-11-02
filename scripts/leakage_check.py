"""How much does a random ROW split inflate scores versus a split by scenario SET?

Same data (all 15 files), same ~80/20 proportion, same models; only the split
differs:
  * set split:        train on files 1-12, test on files 13-15
  * random-row split: random 20% of rows from all 15 files as test
Models: the project MLP (centralized, 40 epochs, same optimizer) and a
RandomForest(200 trees) as a model-agnostic check. 3 seeds each.

Usage:  python scripts/leakage_check.py [--device cuda]
Writes results/leakage_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedpriv.data import load_dataset, random_row_split  # noqa: E402
from fedpriv.metrics import scores  # noqa: E402
from fedpriv.train import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data", default=str(ROOT / "data/raw/triple"))
    args = ap.parse_args()
    from sklearn.ensemble import RandomForestClassifier

    t0 = time.time()
    out = {"mlp": {}, "rf": {}}
    cfg = dict(mode="central", rounds=40, local_epochs=1, batch_size=256, lr=0.05, momentum=0.9, hidden=256)
    for split in ("set", "random_row"):
        out["mlp"][split], out["rf"][split] = [], []
        for seed in (0, 1, 2):
            if split == "set":
                d = load_dataset(args.data, train_files=range(1, 13), val_files=(), test_files=(13, 14, 15))
            else:
                d = random_row_split(args.data, range(1, 16), test_frac=0.2, seed=seed)
            # (val is unused here; point it at test so the run() signature is satisfied)
            d.val = d.test
            r = run({**cfg, "seed": seed}, d, device=args.device)
            out["mlp"][split].append({"final": r["test"]["final"], "last10_mean": r["test"]["last10_mean"]})
            rf = RandomForestClassifier(200, n_jobs=4, random_state=seed).fit(d.train.X, d.train.y)
            out["rf"][split].append(scores(d.test.y, rf.predict(d.test.X)))
            print(split, seed, "mlp acc %.4f" % r["test"]["last10_mean"]["acc"],
                  "rf acc %.4f" % out["rf"][split][-1]["acc"], flush=True)
    out["wall_time_s"] = time.time() - t0
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "results/leakage_check.json", "w"), indent=1)
    for m in ("mlp", "rf"):
        for split in ("set", "random_row"):
            rows = [x["last10_mean"] if m == "mlp" else x for x in out[m][split]]
            acc = np.array([x["acc"] for x in rows])
            f1 = np.array([x["macro_f1"] for x in rows])
            print(f"{m:4s} {split:10s} acc {acc.mean():.4f}±{acc.std(ddof=1):.4f}  macroF1 {f1.mean():.4f}±{f1.std(ddof=1):.4f}")


if __name__ == "__main__":
    main()
