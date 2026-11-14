"""Stage 2b: risk-controlled cascade and interference-conditional risk control
(python -m scripts.evaluate_cascade). Writes results/results_cascade.json."""
import os, json, glob, pickle
import numpy as np
from cspa.env import sum_rate
from cspa.selective import RiskScorer, FEATURE_SETS, crc_threshold, regret
from cspa.cascade import (stage_outputs, assign, path_losses, cost, calibrate_t,
                          choose_offset, groupwise_crc)
from scripts.evaluate import pipeline, load, ens

OUT = os.environ.get('CSPA_OUT', 'results')
ALPHAS = [0.001, 0.002, 0.003, 0.004, 0.005]
WARM = 10
GRID = [0.0, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 1.0]
rng = np.random.default_rng(7)
R = {'warm_iters': WARM}

sc_d, cal_d, te_d = load('scorer'), load('cal'), load('test')
S, C, T = pipeline(sc_d, ens), pipeline(cal_d, ens), pipeline(te_d, ens)
scorer = RiskScorer(FEATURE_SETS['full']).fit(S['feats'], S['loss'])
sS, sC, sT = scorer.score(S['feats']), scorer.score(C['feats']), scorer.score(T['feats'])
stS = stage_outputs(sc_d['He'], sc_d['Ht'], S['p'], sc_d['Pref'], WARM)
stC = stage_outputs(cal_d['He'], cal_d['Ht'], C['p'], cal_d['Pref'], WARM)
stT = stage_outputs(te_d['He'], te_d['Ht'], T['p'], te_d['Pref'], WARM)
r_ref = stT['r3'].mean()
r_nn = sum_rate(te_d['Ht'], T['p'])

# fixed (non-selective) strategies for reference
R['fixed'] = {
    'NN only': dict(risk=float(T['loss'].mean()), pct=float(100 * r_nn.mean() / r_ref), iters=0),
    'NN + 1-step probe': dict(risk=float(stT['l1'].mean()), pct=float(100 * stT['r1'].mean() / r_ref), iters=0),
    f'NN + {WARM}-it. warm WMMSE (always)': dict(risk=float(stT['l2'].mean()),
                                                 pct=float(100 * stT['r2'].mean() / r_ref), iters=WARM),
    'Cold WMMSE (always)': dict(risk=0.0, pct=100.0, iters=100),
}
print(R['fixed'], flush=True)

R['cascade'], R['single'], R['single_probe'] = {}, {}, {}
for a_ in ALPHAS:
    # single threshold, network vs cold WMMSE (Sec. 4.4)
    lam = crc_threshold(sC, C['loss'], a_); acc = sT <= lam
    R['single'][str(a_)] = dict(risk=float(np.mean(T['loss'] * acc)), defer=float(1 - acc.mean()),
                                iters=float(100 * (1 - acc.mean())),
                                pct=float(100 * np.where(acc, r_nn, stT['r3']).mean() / r_ref))
    # single threshold, (network + probe) vs cold WMMSE
    lam = crc_threshold(sC, stC['l1'], a_); acc = sT <= lam
    R['single_probe'][str(a_)] = dict(risk=float(np.mean(stT['l1'] * acc)), defer=float(1 - acc.mean()),
                                      iters=float(100 * (1 - acc.mean())),
                                      pct=float(100 * np.where(acc, stT['r1'], stT['r3']).mean() / r_ref))
    # cascade: offset chosen on D_sc, t calibrated on D_cal (CRC and LTT-EB)
    a = choose_offset(sS, stS, a_, WARM, GRID)
    res = {'offset_a': float(a)}
    for m in ['crc', 'ltt']:
        t = calibrate_t(sC, stC, a, a_, method=m)
        L, k = path_losses(stT, sT, t, a, 'actual')
        Le, _ = path_losses(stT, sT, t, a, 'env')
        rate = np.where(k == 0, stT['r1'], np.where(k == 1, stT['r2'], stT['r3']))
        res[m] = dict(t=float(t), risk=float(L.mean()), env_risk=float(Le.mean()),
                      iters=float(cost(k, WARM).mean()),
                      share=[float((k == i).mean()) for i in range(3)],
                      pct=float(100 * rate.mean() / r_ref))
    R['cascade'][str(a_)] = res
    print('alpha', a_, 'single', R['single'][str(a_)], '\n  cascade', res, flush=True)

# validity of cascade-CRC over 300 resplits (offset fixed on D_sc)
pool_s = np.concatenate([sC, sT])
pool = {k: np.concatenate([stC[k], stT[k]]) for k in ['env1', 'env2', 'l1', 'l2']}
n = len(sC)
R['cascade_validity'] = {}
for a_ in ALPHAS:
    a = R['cascade'][str(a_)]['offset_a']
    risks, envs, its = [], [], []
    for rep in range(300):
        perm = rng.permutation(len(pool_s)); ci, ti = perm[:n], perm[n:]
        stc = {k: v[ci] for k, v in pool.items()}; stt = {k: v[ti] for k, v in pool.items()}
        t = calibrate_t(pool_s[ci], stc, a, a_)
        L, k = path_losses(stt, pool_s[ti], t, a, 'actual')
        Le, _ = path_losses(stt, pool_s[ti], t, a, 'env')
        risks.append(L.mean()); envs.append(Le.mean()); its.append(cost(k, WARM).mean())
    R['cascade_validity'][str(a_)] = dict(mean=float(np.mean(risks)), std=float(np.std(risks)),
                                          viol=float(np.mean(np.array(risks) > a_)),
                                          env_mean=float(np.mean(envs)),
                                          env_viol=float(np.mean(np.array(envs) > a_)),
                                          iters=float(np.mean(its)))
    print('cascade validity', a_, R['cascade_validity'][str(a_)], flush=True)

# interference-conditional (group-wise) risk control, alpha = 0.003
A = 0.003
q = np.quantile(S['feats']['rate_per_link'], [1 / 3, 2 / 3])
gC, gT = np.digitize(C['feats']['rate_per_link'], q), np.digitize(T['feats']['rate_per_link'], q)
names = ['Interference-limited (low rate/link)', 'Medium', 'Noise-limited (high rate/link)']
lam_m = crc_threshold(sC, C['loss'], A)
lam_g = groupwise_crc(sC, C['loss'], gC, A, 3)
R['conditional'] = {'groups': names, 'base_regret': [float(T['loss'][gT == i].mean()) for i in range(3)]}
pool_g = np.concatenate([gC, gT]); pool_l = np.concatenate([C['loss'], T['loss']])
for nm, fn in [('marginal', lambda s, l, g: np.full(3, crc_threshold(s, l, A))),
               ('groupwise', lambda s, l, g: groupwise_crc(s, l, g, A, 3))]:
    rk = [[] for _ in range(3)]; df = [[] for _ in range(3)]
    for rep in range(300):
        perm = rng.permutation(len(pool_s)); ci, ti = perm[:n], perm[n:]
        lam = fn(pool_s[ci], pool_l[ci], pool_g[ci])
        for i in range(3):
            m = pool_g[ti] == i; acc = pool_s[ti][m] <= lam[i]
            rk[i].append(np.mean(pool_l[ti][m] * acc)); df[i].append(1 - acc.mean())
    R['conditional'][nm] = [dict(mean=float(np.mean(rk[i])), viol=float(np.mean(np.array(rk[i]) > A)),
                                 defer=float(np.mean(df[i]))) for i in range(3)]
    R['conditional'][nm + '_overall_defer'] = float(np.mean([np.mean(df[i]) for i in range(3)]))
    print('conditional', nm, R['conditional'][nm], flush=True)

# cascade on OOD shifts (offline calibration, alpha = 0.003)
R['cascade_ood'] = {}
a = R['cascade']['0.003']['offset_a']; t = R['cascade']['0.003']['crc']['t']
for f in sorted(glob.glob(f'{OUT}/data_ood_*.pkl')):
    name = f.split('data_ood_')[1][:-4]; d = pickle.load(open(f, 'rb'))
    O = pipeline(d, ens); so = scorer.score(O['feats'])
    st = stage_outputs(d['He'], d['Ht'], O['p'], d['Pref'], WARM)
    L, k = path_losses(st, so, t, a, 'actual')
    rate = np.where(k == 0, st['r1'], np.where(k == 1, st['r2'], st['r3']))
    R['cascade_ood'][name] = dict(risk=float(L.mean()), iters=float(cost(k, WARM).mean()),
                                  pct=float(100 * rate.mean() / st['r3'].mean()),
                                  warm_always_risk=float(st['l2'].mean()))
    print('cascade OOD', name, R['cascade_ood'][name], flush=True)

json.dump(R, open(f'{OUT}/results_cascade.json', 'w'), indent=1)
print('done')
