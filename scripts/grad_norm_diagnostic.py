"""Why does per-sample clipping hurt the rare classes? Measure per-sample
gradient norms by class during ordinary (non-private) centralized training.

Trains the project MLP on the training files (lr 0.05, momentum 0.9, batch 256)
and, at 0 / 2 / 12 epochs, computes per-sample gradient L2 norms (torch.func
vmap) on 4,000 random training rows, reporting the median and 90th percentile
per class and the fraction of each class's gradients that clipping at C would
shrink. Seeds 0, 1, 2. CPU only (tiny).

Usage:  python scripts/grad_norm_diagnostic.py
Writes results/grad_norms.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedpriv.data import CLASSES, load_dataset  # noqa: E402
from fedpriv.model import MLP  # noqa: E402
from fedpriv.train import local_sgd  # noqa: E402


def per_sample_norms(model, X, y):
    params = {k: v.detach() for k, v in model.named_parameters()}

    def loss(p, x, t):
        return F.cross_entropy(functional_call(model, p, (x[None],)), t[None])

    g = vmap(grad(loss), in_dims=(None, 0, 0))(params, X, y)
    return torch.sqrt(sum((v.flatten(1) ** 2).sum(1) for v in g.values()))


def main():
    torch.set_num_threads(4)
    d = load_dataset(ROOT / "data/raw/triple")
    X = torch.from_numpy(d.train.X)
    y = torch.from_numpy(d.train.y)
    out = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        model = MLP(X.shape[1], 256, len(CLASSES))
        idx = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))[:4000]
        gen = torch.Generator().manual_seed(seed)
        done = 0
        for target in (0, 2, 12):
            if target > done:
                local_sgd(model, X, y, epochs=target - done, batch_size=256, lr=0.05, momentum=0.9, gen=gen)
                done = target
            n = per_sample_norms(model, X[idx], y[idx]).numpy()
            yy = y[idx].numpy()
            row = {"seed": seed, "epochs": target}
            for c, name in enumerate(CLASSES):
                nc = n[yy == c]
                row[name] = {
                    "median": float(np.median(nc)),
                    "p90": float(np.percentile(nc, 90)),
                    "frac_above_C1": float((nc > 1.0).mean()),
                    "mean_shrink_factor_C1": float(np.mean(np.minimum(1.0, 1.0 / nc))),
                }
            out.append(row)
            print(seed, target, {k: round(v["median"], 2) for k, v in row.items() if isinstance(v, dict)}, flush=True)
    json.dump(out, open(ROOT / "results/grad_norms.json", "w"), indent=1)

    print("\nmean over seeds (median per-sample grad norm | mean factor by which clipping at C=1 scales it)")
    for target in (0, 2, 12):
        rows = [r for r in out if r["epochs"] == target]
        print(f"epochs={target:2d} " + "  ".join(
            f"{c}: {np.mean([r[c]['median'] for r in rows]):.2f} | x{np.mean([r[c]['mean_shrink_factor_C1'] for r in rows]):.2f}"
            for c in CLASSES))


if __name__ == "__main__":
    main()
