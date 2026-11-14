"""Fast end-to-end smoke test of the full CRC-Cascade pipeline on a tiny problem.

Runs in well under a minute on a laptop CPU:  python -m pytest -q   or   python -m tests.test_pipeline
It checks shapes, value ranges and the basic properties the paper relies on; it does
NOT reproduce the paper's numbers (use reproduce.sh for that).
"""
import numpy as np
from cspa.env import Scenario, generate, features, sum_rate
from cspa.optim import wmmse, max_power
from cspa.models import IAAN
from cspa.selective import (ensemble_outputs, select_and_features, RiskScorer, FEATURE_SETS,
                            regret, crc_threshold, hp_threshold, online_recalibration)
from cspa.cascade import stage_outputs, calibrate_t, assign, path_losses, cost, groupwise_crc

K, ALPHA = 8, 0.01


def _data(n, seed, sc=Scenario(K=K)):
    Ht, He = generate(sc, n, np.random.default_rng(seed))
    Pref, _ = wmmse(He)
    return Ht, He, Pref


def test_simulator_and_wmmse():
    Ht, He, Pref = _data(50, 0)
    assert Ht.shape == He.shape == (50, K, K) and (Ht > 0).all()
    assert Pref.shape == (50, K) and (Pref >= 0).all() and (Pref <= 1).all()
    # WMMSE should beat full power on average (on the channel it optimised)
    assert sum_rate(He, Pref).mean() >= sum_rate(He, max_power(He)).mean()


def test_end_to_end_pipeline():
    rng = np.random.default_rng(0)
    Ht, He, Pref = _data(400, 1)
    models = [IAAN(d=16, heads=2, layers=2, seed=s).fit(features(He), He, Pref, iters=60, bs=64,
                                                       rng=np.random.default_rng(s)) for s in range(2)]
    splits = {}
    for name, seed, n in [('sc', 2, 200), ('cal', 3, 200), ('te', 4, 200)]:
        ht, he, pr = _data(n, seed)
        P, R = ensemble_outputs(models, he)
        p_hat, f = select_and_features(P, R, he)
        splits[name] = dict(ht=ht, he=he, pr=pr, p=p_hat, f=f, l=regret(ht, p_hat, pr))
    for d in splits.values():
        assert ((d['l'] >= 0) & (d['l'] <= 1)).all()
    scorer = RiskScorer(FEATURE_SETS['full']).fit(splits['sc']['f'], splits['sc']['l'])
    s = {k: scorer.score(v['f']) for k, v in splits.items()}

    # single-threshold CRC and LTT-EB
    lam = crc_threshold(s['cal'], splits['cal']['l'], ALPHA)
    lam_hp = hp_threshold(s['cal'], splits['cal']['l'], ALPHA)
    assert lam_hp <= lam or np.isinf(lam_hp)          # LTT is the more conservative rule

    # three-stage cascade with monotone envelope (Theorem 2)
    st = {k: stage_outputs(v['he'], v['ht'], v['p'], v['pr'], warm_iters=5) for k, v in splits.items()}
    assert (st['cal']['env1'] >= st['cal']['env2']).all()
    a = 0.05
    t = calibrate_t(s['cal'], st['cal'], a, ALPHA)
    L, k = path_losses(st['te'], s['te'], t, a)
    assert set(np.unique(k)) <= {0, 1, 2} and (L >= 0).all()
    # cascade envelope loss is non-decreasing in t (needed for CRC)
    grid = np.linspace(s['cal'].min() - 1, s['cal'].max() + 1, 25)
    Rg = [path_losses(st['cal'], s['cal'], tt, a)[0].mean() for tt in grid]
    assert np.all(np.diff(Rg) >= -1e-12)
    assert cost(k, 5).mean() <= 100

    # group-wise (Mondrian) CRC and online recalibration run
    g = np.digitize(splits['cal']['f']['rate_per_link'], np.quantile(splits['cal']['f']['rate_per_link'], [1/3, 2/3]))
    assert groupwise_crc(s['cal'], splits['cal']['l'], g, ALPHA, 3).shape == (3,)
    acc, aud, lams = online_recalibration(s['te'], splits['te']['l'], lam, ALPHA, 0.01, 0.1, rng)
    assert acc.shape == aud.shape == lams.shape == (200,)


if __name__ == '__main__':
    test_simulator_and_wmmse(); test_end_to_end_pipeline(); print('pipeline smoke test passed')
