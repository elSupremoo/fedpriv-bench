"""Run every configuration in a YAML sweep file, one at a time.

Each run is saved to results/runs/<name>.json; existing files are skipped, so
the sweep is resumable. Before each run the script checks free GPU memory and
waits if the (shared) GPU is short on memory.

Usage:  python scripts/run_sweep.py configs/main.yaml [--device cuda]
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedpriv.data import load_dataset  # noqa: E402
from fedpriv.train import run  # noqa: E402


def run_name(cfg: dict) -> str:
    parts = [cfg.get("tag", ""), cfg["mode"]]
    if cfg["mode"] != "central":
        parts.append(f"a{cfg['alpha']}")
    if cfg["mode"] == "fedprox":
        parts.append(f"mu{cfg['mu']}")
    if cfg["mode"] == "dp_fedavg":
        parts.append(f"eps{cfg['target_epsilon']}")
    parts.append(f"s{cfg['seed']}")
    return "_".join(p for p in parts if p)


def expand(spec: dict) -> list[dict]:
    base = spec.get("base", {})
    out = []
    for block in spec["runs"]:
        block = {**base, **block}
        grid_keys = [k for k, v in block.items() if isinstance(v, list)]
        for values in itertools.product(*(block[k] for k in grid_keys)):
            cfg = dict(block)
            cfg.update(dict(zip(grid_keys, values)))
            out.append(cfg)
    return out


def wait_for_gpu(min_free_gb: float, device: str):
    if not device.startswith("cuda"):
        return
    import torch

    while True:
        free, total = torch.cuda.mem_get_info()
        if free / 2**30 >= min_free_gb:
            return
        print(f"  GPU busy: {free / 2**30:.1f} GB free < {min_free_gb} GB, waiting 60 s", flush=True)
        time.sleep(60)


def environment() -> dict:
    import numpy
    import opacus
    import sklearn
    import torch

    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "opacus": opacus.__version__,
        "numpy": numpy.__version__,
        "sklearn": sklearn.__version__,
        "cpu": platform.processor(),
    }
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data", default=str(ROOT / "data/raw/triple"))
    ap.add_argument("--out", default=str(ROOT / "results/runs"))
    ap.add_argument("--min-free-gb", type=float, default=3.0)
    args = ap.parse_args()

    spec = yaml.safe_load(open(args.config))
    cfgs = expand(spec)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cfgs if not (out / f"{run_name(c)}.json").exists()]
    print(f"{len(cfgs)} configs, {len(todo)} to run", flush=True)
    if not todo:
        return
    data = load_dataset(args.data)
    env = environment()
    for i, cfg in enumerate(todo, 1):
        name = run_name(cfg)
        wait_for_gpu(args.min_free_gb, args.device)
        print(f"[{i}/{len(todo)}] {name}", flush=True)
        res = run(cfg, data, device=args.device)
        res["name"] = name
        res["device"] = args.device
        res["environment"] = env
        t = res["test"]
        print(
            f"   {res['wall_time_s']:.0f}s  test acc final {t['final']['acc']:.4f} "
            f"last10 {t['last10_mean']['acc']:.4f}  macroF1 last10 {t['last10_mean']['macro_f1']:.4f}"
            + (f"  eps_max {res['epsilon_max']:.3f}" if "epsilon_max" in res else ""),
            flush=True,
        )
        with open(out / f"{name}.json", "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
