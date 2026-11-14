"""Stage 1: generate data sets and train all learning models (python -m scripts.train)."""
import os, pickle, time
import numpy as np
from multiprocessing import Pool
from cspa.env import Scenario, generate, features
from cspa.optim import wmmse
from cspa.models import IAAN, GNN, MLP

OUT = os.environ.get('CSPA_OUT', 'results')
N_TRAIN, N_SCORER, N_CAL, N_TEST = 20000, 5000, 5000, 10000
M_ENS = 5
IT_IAAN, IT_GNN, IT_MLP = 4000, 3000, 8000


def make(sc, n, seed):
    Ht, He = generate(sc, n, np.random.default_rng(seed))
    Pref, _ = wmmse(He)
    return dict(Ht=Ht, He=He, Pref=Pref)


def job(spec):
    kind, seed = spec
    d = pickle.load(open(f'{OUT}/data_train.pkl', 'rb'))
    F, He, Pref = features(d['He']), d['He'], d['Pref']
    rng = np.random.default_rng(100 + seed)
    t = time.time()
    if kind == 'iaan':
        idx = rng.integers(0, len(F), len(F))     # bootstrap resample per member
        m = IAAN(seed=seed).fit(F[idx], He[idx], Pref[idx], iters=IT_IAAN, rng=rng)
    elif kind == 'iaan_nobias':
        m = IAAN(seed=seed, gain_bias=False).fit(F, He, Pref, iters=IT_IAAN, rng=rng)
    elif kind == 'gnn':
        m = GNN(seed=seed).fit(F, He, Pref, iters=IT_GNN, rng=rng)
    elif kind == 'mlp':
        m = MLP(He.shape[1], seed=seed).fit(F, He, Pref, iters=IT_MLP, unsup=False, rng=rng, lr=1e-3)
    pickle.dump(m, open(f'{OUT}/model_{kind}_{seed}.pkl', 'wb'))
    return kind, seed, time.time() - t


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    sc = Scenario()
    t = time.time()
    for name, n, seed in [('train', N_TRAIN, 1), ('scorer', N_SCORER, 2), ('cal', N_CAL, 3), ('test', N_TEST, 4)]:
        pickle.dump(make(sc, n, seed), open(f'{OUT}/data_{name}.pkl', 'wb'))
    shifts = {
        'denser_K30': sc.shift(K=30),
        'sparser_K10': sc.shift(K=10),
        'small_hall': sc.shift(area=20.0),
        'csi_err_0.3': sc.shift(csi_err=0.3),
        'heavy_shadow': sc.shift(sigma_nlos=10.0, sigma_los=6.0),
    }
    for i, (name, s) in enumerate(shifts.items()):
        pickle.dump(make(s, 6000, 50 + i), open(f'{OUT}/data_ood_{name}.pkl', 'wb'))
    print(f'data done in {time.time() - t:.1f}s', flush=True)
    jobs = [('iaan', s) for s in range(M_ENS)] + [('iaan_nobias', 0), ('gnn', 0), ('mlp', 0)]
    with Pool(2) as pool:
        for kind, seed, dt in pool.imap_unordered(job, jobs):
            print(f'trained {kind} seed {seed} in {dt:.0f}s', flush=True)
