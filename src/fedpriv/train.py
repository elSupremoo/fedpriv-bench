"""Centralized, FedAvg, FedProx and DP-FedAvg (Opacus DP-SGD on clients).

Everything is hand-written except the per-sample gradient / clipping / noise
machinery and the RDP accountant, which come from Opacus.

Compute matching: an FL run with R rounds and E local epochs (all K clients
participate every round) processes R * E * N_train sample-gradients. The
centralized baseline is trained for R * E epochs over the same N_train rows
with the same batch size and optimizer, i.e. the same number of
sample-gradient passes, and is evaluated every E epochs so that its curve has
the same R points as the FL curve.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import evaluate
from .model import MLP
from .partition import class_histogram, make_partition


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def weighted_average(states: list[dict], weights) -> dict:
    """Sample-weighted parameter average: sum_k (n_k / sum n) * w_k."""
    w = torch.as_tensor(np.asarray(weights, dtype=np.float64))
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be non-negative with positive sum")
    w = w / w.sum()
    out = {}
    for key in states[0]:
        ref = states[0][key]
        if not torch.is_floating_point(ref):
            out[key] = ref.clone()
            continue
        acc = torch.zeros_like(ref, dtype=torch.float64)
        for wk, s in zip(w, states):
            acc += wk.item() * s[key].to(torch.float64)
        out[key] = acc.to(ref.dtype)
    return out


# --------------------------------------------------------------------------
# Local training
# --------------------------------------------------------------------------
def local_sgd(model, X, y, *, epochs, batch_size, lr, momentum, gen, prox_mu=0.0, global_params=None):
    """Plain mini-batch SGD (fresh momentum buffer each call).

    With prox_mu > 0 this is the FedProx local solver: the gradient of
    (mu/2)||w - w_global||^2, i.e. mu * (w - w_global), is added to each step.
    Returns the number of sample-gradients computed.
    """
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    n = len(y)
    seen = 0
    for _ in range(epochs):
        perm = torch.randperm(n, device=X.device, generator=gen)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(X[idx]), y[idx])
            loss.backward()
            if prox_mu > 0:
                with torch.no_grad():
                    for p, g in zip(model.parameters(), global_params):
                        p.grad.add_(p.detach() - g, alpha=prox_mu)
            opt.step()
            seen += len(idx)
    return seen


class DPClient:
    """Per-client DP-SGD state: Poisson sampling rate, noise multiplier, accountant."""

    def __init__(self, n, *, batch_size, rounds, local_epochs, target_epsilon, delta, max_grad_norm):
        from opacus.accountants import RDPAccountant
        from opacus.accountants.utils import get_noise_multiplier

        self.n = n
        self.q = min(1.0, batch_size / n)
        self.steps_per_epoch = max(1, int(round(1.0 / self.q)))
        self.total_steps = rounds * local_epochs * self.steps_per_epoch
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        if math.isinf(target_epsilon):
            self.sigma = 0.0  # clipping only, no noise
        else:
            self.sigma = float(
                get_noise_multiplier(
                    target_epsilon=target_epsilon,
                    target_delta=delta,
                    sample_rate=self.q,
                    steps=self.total_steps,
                    accountant="rdp",
                    epsilon_tolerance=0.01,
                )
            )
        self.accountant = RDPAccountant()
        self.empty_batches = 0

    def epsilon(self) -> float:
        if self.sigma == 0.0:
            return float("inf")
        return float(self.accountant.get_epsilon(self.delta))


def dp_local_sgd(gsm, client: DPClient, X, y, *, epochs, lr, momentum, gen):
    """DP-SGD with Poisson sampling, per-sample clipping and Gaussian noise
    (Opacus GradSampleModule + DPOptimizer). Accountant stepped per step."""
    from opacus.optimizers import DPOptimizer

    gsm.train()
    base = torch.optim.SGD(gsm.parameters(), lr=lr, momentum=momentum)
    opt = DPOptimizer(
        base,
        noise_multiplier=client.sigma,
        max_grad_norm=client.max_grad_norm,
        expected_batch_size=max(1, int(round(client.q * client.n))),
        loss_reduction="mean",
    )
    seen = 0
    for _ in range(epochs):
        for _ in range(client.steps_per_epoch):
            mask = torch.rand(client.n, device=X.device, generator=gen) < client.q
            idx = mask.nonzero().squeeze(1)
            # The accountant always counts the step (conservative).
            client.accountant.step(noise_multiplier=client.sigma, sample_rate=client.q)
            if idx.numel() == 0:
                client.empty_batches += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(gsm(X[idx]), y[idx])
            loss.backward()
            opt.step()
            seen += idx.numel()
    opt.zero_grad(set_to_none=True)
    return seen


# --------------------------------------------------------------------------
# Experiment driver
# --------------------------------------------------------------------------
def _seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _summary(history: list[dict], last: int = 10) -> dict:
    tail = history[-last:]
    return {
        "final": history[-1],
        f"last{last}_mean": {
            "acc": float(np.mean([h["acc"] for h in tail])),
            "macro_f1": float(np.mean([h["macro_f1"] for h in tail])),
            "recall": np.mean([h["recall"] for h in tail], axis=0).tolist(),
        },
    }


def run(cfg: dict, data, device: str = "cuda") -> dict:
    """Run one configuration. ``data`` is a fedpriv.data.Dataset.

    cfg keys: mode (central|fedavg|fedprox|dp_fedavg), seed, alpha (None/'iid'
    or float), num_clients, rounds, local_epochs, batch_size, lr, momentum,
    mu (fedprox), target_epsilon / delta / max_grad_norm (dp_fedavg).
    """
    t0 = time.time()
    _seed_all(cfg["seed"])
    dev = torch.device(device)
    num_classes = int(data.train.y.max()) + 1
    Xtr = torch.from_numpy(data.train.X).to(dev)
    ytr = torch.from_numpy(data.train.y).to(dev)
    Xva = torch.from_numpy(data.val.X).to(dev)
    Xte = torch.from_numpy(data.test.X).to(dev)
    gen = torch.Generator(device=dev).manual_seed(cfg["seed"])

    model = MLP(Xtr.shape[1], cfg.get("hidden", 256), num_classes).to(dev)
    R, E, B = cfg["rounds"], cfg["local_epochs"], cfg["batch_size"]
    hist_val, hist_test = [], []
    total_seen = 0
    extra: dict = {}

    def log_eval():
        hist_val.append(evaluate(model, Xva, data.val.y, num_classes))
        hist_test.append(evaluate(model, Xte, data.test.y, num_classes))

    mode = cfg["mode"]
    if mode == "central":
        for _ in range(R):
            total_seen += local_sgd(
                model, Xtr, ytr, epochs=E, batch_size=B, lr=cfg["lr"], momentum=cfg["momentum"], gen=gen
            )
            log_eval()
    else:
        parts = make_partition(data.train.y, cfg["num_clients"], cfg["alpha"], cfg["seed"], cfg.get("min_client_size", 100))
        sizes = [len(p) for p in parts]
        extra["client_sizes"] = sizes
        extra["client_class_hist"] = class_histogram(data.train.y, parts, num_classes).tolist()
        cX = [Xtr[torch.from_numpy(p).to(dev)] for p in parts]
        cy = [ytr[torch.from_numpy(p).to(dev)] for p in parts]

        if mode == "dp_fedavg":
            from opacus import GradSampleModule

            dp_clients = [
                DPClient(
                    n,
                    batch_size=B,
                    rounds=R,
                    local_epochs=E,
                    target_epsilon=float(cfg["target_epsilon"]),
                    delta=cfg["delta"],
                    max_grad_norm=cfg["max_grad_norm"],
                )
                for n in sizes
            ]
            extra["noise_multipliers"] = [c.sigma for c in dp_clients]
            local = MLP(Xtr.shape[1], cfg.get("hidden", 256), num_classes).to(dev)
            gsm = GradSampleModule(local, loss_reduction="mean")
        else:
            local = copy.deepcopy(model)

        for _ in range(R):
            global_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            states = []
            for k in range(len(parts)):
                if mode == "dp_fedavg":
                    local.load_state_dict(global_state)
                    total_seen += dp_local_sgd(
                        gsm, dp_clients[k], cX[k], cy[k], epochs=E, lr=cfg["lr"], momentum=cfg["momentum"], gen=gen
                    )
                else:
                    local.load_state_dict(global_state)
                    mu = cfg.get("mu", 0.0) if mode == "fedprox" else 0.0
                    gparams = [global_state[n] for n, _ in local.named_parameters()] if mu > 0 else None
                    total_seen += local_sgd(
                        local, cX[k], cy[k], epochs=E, batch_size=B, lr=cfg["lr"],
                        momentum=cfg["momentum"], gen=gen, prox_mu=mu, global_params=gparams,
                    )
                states.append({k2: v.detach().clone() for k2, v in local.state_dict().items()})
            model.load_state_dict(weighted_average(states, sizes))
            log_eval()

        if mode == "dp_fedavg":
            eps = [c.epsilon() for c in dp_clients]
            extra["epsilon_per_client"] = eps
            extra["epsilon_max"] = max(eps)
            extra["empty_batches"] = sum(c.empty_batches for c in dp_clients)

    return {
        "config": cfg,
        "n_train": int(len(ytr)),
        "sample_grad_passes": int(total_seen),
        "wall_time_s": time.time() - t0,
        "val": {"history": hist_val, **_summary(hist_val)},
        "test": {"history": hist_test, **_summary(hist_test)},
        **extra,
    }
