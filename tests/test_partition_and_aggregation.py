import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedpriv.data import Preprocessor  # noqa: E402
from fedpriv.model import MLP  # noqa: E402
from fedpriv.partition import dirichlet_partition, iid_partition  # noqa: E402
from fedpriv.train import local_sgd, weighted_average  # noqa: E402


def _labels(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(3, size=n, p=[0.7, 0.25, 0.05])


@pytest.mark.parametrize("alpha", [0.1, 0.5, 100.0])
def test_dirichlet_is_a_partition(alpha):
    y = _labels()
    parts = dirichlet_partition(y, 10, alpha, seed=1, min_size=50)
    allidx = np.concatenate(parts)
    assert len(parts) == 10
    assert len(allidx) == len(y)  # nothing dropped
    assert len(np.unique(allidx)) == len(y)  # nothing duplicated
    assert min(len(p) for p in parts) >= 50


def test_dirichlet_deterministic_per_seed():
    y = _labels()
    a = dirichlet_partition(y, 10, 0.5, seed=3)
    b = dirichlet_partition(y, 10, 0.5, seed=3)
    c = dirichlet_partition(y, 10, 0.5, seed=4)
    assert all(np.array_equal(x, z) for x, z in zip(a, b))
    assert not all(np.array_equal(x, z) for x, z in zip(a, c))


def test_small_alpha_is_more_skewed_than_large_alpha():
    y = _labels(20000)

    def mean_max_class_share(alpha):
        shares = []
        for s in range(5):
            for p in dirichlet_partition(y, 10, alpha, seed=s, min_size=20):
                shares.append(np.bincount(y[p], minlength=3).max() / len(p))
        return np.mean(shares)

    assert mean_max_class_share(0.1) > mean_max_class_share(100.0) + 0.1


def test_iid_partition_sizes():
    parts = iid_partition(1003, 10, seed=0)
    sizes = [len(p) for p in parts]
    assert sum(sizes) == 1003 and max(sizes) - min(sizes) <= 1
    assert len(np.unique(np.concatenate(parts))) == 1003


def test_weighted_average_matches_manual():
    torch.manual_seed(0)
    models = [MLP(8, 16, 3) for _ in range(3)]
    states = [m.state_dict() for m in models]
    w = [10, 30, 60]
    avg = weighted_average(states, w)
    for k in states[0]:
        expected = 0.1 * states[0][k] + 0.3 * states[1][k] + 0.6 * states[2][k]
        assert torch.allclose(avg[k], expected, atol=1e-6)


def test_weighted_average_identical_models_is_identity_and_ignores_weights():
    s = MLP(8, 16, 3).state_dict()
    avg = weighted_average([s, s, s], [1, 5, 100])
    for k in s:
        assert torch.allclose(avg[k], s[k], atol=1e-6)


def test_weighted_average_rejects_bad_weights():
    s = MLP(8, 16, 3).state_dict()
    with pytest.raises(ValueError):
        weighted_average([s, s], [0, 0])


def test_fedavg_one_client_equals_local_training():
    """With K=1, one FedAvg round must equal plain local SGD (sanity check)."""
    torch.manual_seed(0)
    X = torch.randn(200, 8)
    y = torch.randint(0, 3, (200,))
    m1 = MLP(8, 16, 3)
    m2 = MLP(8, 16, 3)
    m2.load_state_dict(m1.state_dict())
    kw = dict(epochs=2, batch_size=32, lr=0.1, momentum=0.9)
    local_sgd(m1, X, y, gen=torch.Generator().manual_seed(1), **kw)
    local_sgd(m2, X, y, gen=torch.Generator().manual_seed(1), **kw)
    avg = weighted_average([m2.state_dict()], [200])
    for k, v in m1.state_dict().items():
        assert torch.allclose(avg[k], v, atol=1e-6)


def test_fedprox_mu_pulls_towards_global():
    torch.manual_seed(0)
    X = torch.randn(256, 8)
    y = torch.randint(0, 3, (256,))
    g = MLP(8, 16, 3)
    gp = [p.detach().clone() for p in g.parameters()]

    def drift(mu):
        m = MLP(8, 16, 3)
        m.load_state_dict(g.state_dict())
        local_sgd(m, X, y, epochs=5, batch_size=32, lr=0.1, momentum=0.0,
                  gen=torch.Generator().manual_seed(0), prox_mu=mu, global_params=gp)
        return sum(((p - q) ** 2).sum().item() for p, q in zip(m.parameters(), gp))

    assert drift(1.0) < drift(0.0)


def test_preprocessor_fits_on_train_only_and_handles_inf():
    tr = np.array([[1.0, np.inf], [3.0, 2.0], [5.0, 4.0]])
    tr[~np.isfinite(tr)] = np.nan
    te = np.array([[100.0, np.nan]])
    p = Preprocessor().fit(tr)
    z = p.transform(te)
    assert np.isfinite(z).all()
    assert z[0, 0] == pytest.approx(10.0)  # (100-3)/std, clipped at 10
    assert z[0, 1] == pytest.approx((3.0 - p.mean[1]) / p.std[1])  # imputed with train median
