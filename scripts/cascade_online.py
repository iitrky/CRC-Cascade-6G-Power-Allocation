"""Online recalibration of the cascade path parameter t under shift (python -m scripts.cascade_online)."""
import os, json, glob, pickle
import numpy as np
from cspa.selective import RiskScorer, FEATURE_SETS
from cspa.cascade import stage_outputs, assign, cost
from scripts.evaluate import pipeline, load, ens

OUT = os.environ.get('CSPA_OUT', 'results')
R = json.load(open(f'{OUT}/results_cascade.json'))
A, WARM, ETA, BETA = 0.003, R['warm_iters'], 0.01, 0.1
sc = load('scorer'); S = pipeline(sc, ens)
scorer = RiskScorer(FEATURE_SETS['full']).fit(S['feats'], S['loss'])
from cspa.cascade import calibrate_t, choose_offset, path_losses
from cspa.env import sum_rate
GRID_ON = [0.0, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05]  # all stages reachable
stS = stage_outputs(sc['He'], sc['Ht'], S['p'], sc['Pref'], WARM)
a = choose_offset(scorer.score(S['feats']), stS, A, WARM, GRID_ON)
cal, te = load('cal'), load('test'); C, T = pipeline(cal, ens), pipeline(te, ens)
stC = stage_outputs(cal['He'], cal['Ht'], C['p'], cal['Pref'], WARM)
stT = stage_outputs(te['He'], te['Ht'], T['p'], te['Pref'], WARM)
sT = scorer.score(T['feats'])
t0 = calibrate_t(scorer.score(C['feats']), stC, a, A)
L, k = path_losses(stT, sT, t0, a, 'actual')
rate = np.where(k == 0, stT['r1'], np.where(k == 1, stT['r2'], stT['r3']))
R['cascade_bounded'] = dict(offset_a=float(a), t=float(t0), risk=float(L.mean()), iters=float(cost(k, WARM).mean()),
                            share=[float((k == i).mean()) for i in range(3)], pct=float(100 * rate.mean() / stT['r3'].mean()))
print('bounded offset in-dist', R['cascade_bounded'], flush=True)
rng = np.random.default_rng(11)
R['cascade_ood_online'] = {}
for f in sorted(glob.glob(f'{OUT}/data_ood_*.pkl')):
    name = f.split('data_ood_')[1][:-4]; d = pickle.load(open(f, 'rb'))
    O = pipeline(d, ens); s = scorer.score(O['feats'])
    st = stage_outputs(d['He'], d['Ht'], O['p'], d['Pref'], WARM)
    reps = []
    for rep in range(20):
        perm = rng.permutation(len(s)); t = t0; L = np.zeros(len(s)); C = np.zeros(len(s)); aud = 0
        for i, j in enumerate(perm):
            k = assign(s[j:j + 1], t, a)[0]
            loss = (st['l1'][j], st['l2'][j], 0.0)[k]; env = (st['env1'][j], st['env2'][j], 0.0)[k]
            L[i] = loss; C[i] = (0, WARM, 100)[k]
            est = 0.0
            if k < 2 and rng.random() < BETA:
                aud += 1; est = env / BETA
            t = t + ETA * (A - est)
        reps.append((L.mean(), C.mean() + 100 * aud / len(s), aud / len(s)))
    r = np.array(reps)
    Lo, ko = path_losses(st, s, t0, a, 'actual')
    R['cascade_ood_online'][name] = dict(offline_risk=float(Lo.mean()), offline_iters=float(cost(ko, WARM).mean()), risk=float(r[:, 0].mean()), risk_std=float(r[:, 0].std()),
                                         iters_incl_audit=float(r[:, 1].mean()), audit=float(r[:, 2].mean()))
    print(name, R['cascade_ood_online'][name], flush=True)
json.dump(R, open(f'{OUT}/results_cascade.json', 'w'), indent=1)
