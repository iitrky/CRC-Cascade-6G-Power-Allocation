"""Finite-difference gradient check of every model (run: python -m tests.test_autodiff)."""
import numpy as np
from cspa.env import Scenario, generate, features
from cspa.models import IAAN, MLP, GNN, rate_loss


def check(model, F, H, n_probe=12, eps=1e-6):
    loss = rate_loss(H, model.forward(F))
    loss.backward()
    rng = np.random.default_rng(1)
    worst = 0.0
    for p in model.params():
        if p.grad is None:
            continue
        for _ in range(2):
            i = tuple(rng.integers(0, s) for s in p.value.shape)
            g = p.grad[i]
            old = p.value[i]
            p.value[i] = old + eps; lp = rate_loss(H, model.forward(F)).value
            p.value[i] = old - eps; lm = rate_loss(H, model.forward(F)).value
            p.value[i] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - g) / (abs(num) + abs(g) + 1e-8))
    return worst


if __name__ == '__main__':
    sc = Scenario(K=6)
    Ht, He = generate(sc, 4, np.random.default_rng(0))
    F = features(He)
    for m in [IAAN(d=8, heads=2, layers=2), MLP(6, hidden=(16, 16)), GNN(d=8, layers=2)]:
        print(type(m).__name__, 'max rel. grad error = %.2e' % check(m, F, He))


def test_gradients():
    """pytest entry point: every model's analytic gradient matches finite differences."""
    sc = Scenario(K=6)
    Ht, He = generate(sc, 4, np.random.default_rng(0))
    F = features(He)
    for m in [IAAN(d=8, heads=2, layers=2), MLP(6, hidden=(16, 16)), GNN(d=8, layers=2)]:
        assert check(m, F, He) < 1e-3, type(m).__name__
