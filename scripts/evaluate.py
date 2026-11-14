"""Stage 2: all experiments of the paper (python -m scripts.evaluate).

Writes results/results.json and figures to results/figs/.
"""
import os, json, pickle, time, glob
import numpy as np
from cspa.env import sum_rate, features
from cspa.optim import wmmse, max_power, random_power, binary_greedy
from cspa.selective import (ensemble_outputs, select_and_features, RiskScorer, FEATURE_SETS,
                            regret, crc_threshold, naive_threshold, hp_threshold,
                            online_recalibration)

OUT = os.environ.get('CSPA_OUT', 'results')
ALPHAS = [0.001, 0.002, 0.003, 0.004, 0.005]
ALPHA_MAIN = 0.003
TAIL_TAU, TAIL_ALPHAS = 0.05, [0.002, 0.005, 0.01]
rng = np.random.default_rng(2026)
R = {}


def load(name):
    return pickle.load(open(f'{OUT}/data_{name}.pkl', 'rb'))


def model(kind, seed=0):
    return pickle.load(open(f'{OUT}/model_{kind}_{seed}.pkl', 'rb'))


ens = [model('iaan', s) for s in range(5)]
gnn, mlp, nob = model('gnn'), model('mlp'), model('iaan_nobias')


def pipeline(d, members, rule='best'):
    P, Rm = ensemble_outputs(members, d['He'])
    p_sel, feats = select_and_features(P, Rm, d['He'], rule)
    loss = regret(d['Ht'], p_sel, d['Pref'])
    return dict(p=p_sel, feats=feats, loss=loss, P=P)


def si_heuristic_score(d, P):
    """Deferral score of Lee et al. (arXiv:2401.13206): sum-rate spread over the
    epistemic interval p +- 1.96 sigma_epi, relative to the predicted sum-rate."""
    mu, sd = P.mean(0), P.std(0)
    lo, up = np.clip(mu - 1.96 * sd, 0, 1), np.clip(mu + 1.96 * sd, 0, 1)
    r = np.stack([sum_rate(d['He'], q) for q in (lo, mu, up)])
    return (r.max(0) - r.min(0)) / (r[1] + 1e-9)


def evaluate_threshold(scores, losses, rates_nn, rates_ref, lam):
    acc = scores <= lam
    risk = float(np.mean(losses * acc))
    rate = np.where(acc, rates_nn, rates_ref)
    return dict(lam=float(lam), risk=risk, defer=float(1 - acc.mean()),
                rate=float(rate.mean()), pct_ref=float(100 * rate.mean() / rates_ref.mean()))


if __name__ == '__main__':
    os.makedirs(f'{OUT}/figs', exist_ok=True)
    sc_d, cal_d, te_d = load('scorer'), load('cal'), load('test')
    t0 = time.time()
    S, C, T = pipeline(sc_d, ens), pipeline(cal_d, ens), pipeline(te_d, ens)
    print(f'ensemble pipelines {time.time() - t0:.1f}s', flush=True)
    Ht, He, Pref = te_d['Ht'], te_d['He'], te_d['Pref']
    r_ref = sum_rate(Ht, Pref)

    # ---------------------------------------------------------- 1. baselines
    Ft = features(He)
    p_w5, _ = wmmse(He, iters=5)
    base = {
        'Max power': max_power(He),
        'Random power': random_power(He, rng),
        'Greedy binary': binary_greedy(He, sum_rate),
        'WMMSE (5 it.)': p_w5,
        'WMMSE (100 it.) [ref]': Pref,
        'MLP L2O [Sun 2018]': mlp.predict(Ft),
        'MPGNN [Shen 2021]': gnn.predict(Ft),
        'IAAN w/o gain bias': nob.predict(Ft),
        'IAAN single (ours, backbone)': ens[0].predict(Ft),
        'IAAN ensemble best-of-5': T['p'],
        'Run both, keep better on G-hat': np.where((sum_rate(He, T['p']) >= sum_rate(He, Pref))[:, None], T['p'], Pref),
    }
    R['baselines'] = {}
    for k, p in base.items():
        r = sum_rate(Ht, p)
        reg = regret(Ht, p, Pref)
        R['baselines'][k] = dict(rate=float(r.mean()), std=float(r.std()),
                                 pct_ref=float(100 * r.mean() / r_ref.mean()),
                                 mean_regret=float(reg.mean()),
                                 p95_regret=float(np.percentile(reg, 95)),
                                 frac_regret_gt_5pct=float((reg > 0.05).mean()))
        print(f'{k:32s} {r.mean():7.3f}  {100 * r.mean() / r_ref.mean():6.2f}%  '
              f'meanReg {reg.mean():.4f} P95 {np.percentile(reg, 95):.4f}', flush=True)

    # ---------------------------------------------------------- 2. CSPA main
    scorer = RiskScorer(FEATURE_SETS['full']).fit(S['feats'], S['loss'])
    s_cal, s_te = scorer.score(C['feats']), scorer.score(T['feats'])
    r_nn = sum_rate(Ht, T['p'])
    R['cspa'] = {}
    for a in ALPHAS:
        R['cspa'][str(a)] = {
            'CRC': evaluate_threshold(s_te, T['loss'], r_nn, r_ref, crc_threshold(s_cal, C['loss'], a)),
            'LTT-HP': evaluate_threshold(s_te, T['loss'], r_nn, r_ref, hp_threshold(s_cal, C['loss'], a, 0.1)),
        }
        print('alpha', a, R['cspa'][str(a)], flush=True)

    # ------------------------------------------- 3. validity over resplits
    pool_s = np.concatenate([s_cal, s_te])
    pool_l = np.concatenate([C['loss'], T['loss']])
    n_cal = len(s_cal)
    val = {m: {str(a): [] for a in ALPHAS} for m in ['CRC', 'LTT-HP', 'Naive']}
    defer = {m: {str(a): [] for a in ALPHAS} for m in ['CRC', 'LTT-HP', 'Naive']}
    for rep in range(300):
        perm = rng.permutation(len(pool_s))
        ci, ti = perm[:n_cal], perm[n_cal:]
        for a in ALPHAS:
            for m, f in [('CRC', crc_threshold), ('Naive', naive_threshold),
                         ('LTT-HP', lambda s, l, a: hp_threshold(s, l, a, 0.1))]:
                lam = f(pool_s[ci], pool_l[ci], a)
                acc = pool_s[ti] <= lam
                val[m][str(a)].append(float(np.mean(pool_l[ti] * acc)))
                defer[m][str(a)].append(float(1 - acc.mean()))
    R['validity'] = {m: {a: dict(mean=float(np.mean(v)), std=float(np.std(v)),
                                 viol=float(np.mean(np.array(v) > float(a))),
                                 defer=float(np.mean(defer[m][a])))
                         for a, v in val[m].items()} for m in val}
    print('validity', json.dumps(R['validity'], indent=0)[:1500], flush=True)

    # ------------------------------------------------ 3b. tail-risk control
    lt_c, lt_t = (C['loss'] > TAIL_TAU).astype(float), (T['loss'] > TAIL_TAU).astype(float)
    pool_t = np.concatenate([lt_c, lt_t])
    R['tail'] = {'base_rate': float(lt_t.mean())}
    for a in TAIL_ALPHAS:
        lam = crc_threshold(s_cal, lt_c, a)
        ev = evaluate_threshold(s_te, lt_t, r_nn, r_ref, lam)
        ev['mean_regret'] = float(np.mean(T['loss'] * (s_te <= lam)))
        vs, ds = [], []
        for rep in range(300):
            perm = rng.permutation(len(pool_s)); ci, ti = perm[:n_cal], perm[n_cal:]
            lm = crc_threshold(pool_s[ci], pool_t[ci], a); acc = pool_s[ti] <= lm
            vs.append(float(np.mean(pool_t[ti] * acc))); ds.append(float(1 - acc.mean()))
        ev['resplit_mean'], ev['resplit_std'] = float(np.mean(vs)), float(np.std(vs))
        ev['resplit_viol'], ev['resplit_defer'] = float(np.mean(np.array(vs) > a)), float(np.mean(ds))
        R['tail'][str(a)] = ev
        print('tail', a, ev, flush=True)

    # ---------------------------------------- 4. risk-deferral curves (scores)
    si_cal, si_te = si_heuristic_score(cal_d, C['P']), si_heuristic_score(te_d, T['P'])
    scores = {
        'CSPA learned scorer (full)': (s_cal, s_te),
        'Ensemble features only': None, 'Physics features only': None,
        'SI-DNN spread heuristic [Lee 2024]': (si_cal, si_te),
        'Ensemble power std': (C['feats']['ens_power_std'], T['feats']['ens_power_std']),
        'Random': (rng.random(len(s_cal)), rng.random(len(s_te))),
        'Oracle': (C['loss'], T['loss']),
    }
    for nm, fs in [('Ensemble features only', 'ensemble_only'), ('Physics features only', 'physics_only')]:
        m = RiskScorer(FEATURE_SETS[fs]).fit(S['feats'], S['loss'])
        scores[nm] = (m.score(C['feats']), m.score(T['feats']))
    grid = np.linspace(0, 0.8, 41)
    R['curves'] = {'defer_grid': grid.tolist()}
    R['score_cmp'] = {}
    for nm, (sc_, st_) in scores.items():
        o = np.argsort(-st_)                         # defer highest score first
        risks = []
        for g in grid:
            k = int(round(g * len(st_)))
            acc = np.ones(len(st_), bool); acc[o[:k]] = False
            risks.append(float(np.mean(T['loss'] * acc)))
        R['curves'][nm] = risks
        res = {}
        for a in [0.002, 0.003]:
            lam = crc_threshold(sc_, C['loss'], a)
            res[str(a)] = evaluate_threshold(st_, T['loss'], r_nn, r_ref, lam)
        res['aurc'] = float(np.trapezoid(risks, grid))
        R['score_cmp'][nm] = res
        print(f'{nm:38s} defer@0.002={res["0.002"]["defer"]:.3f} defer@0.003={res["0.003"]["defer"]:.3f} '
              f'risk@0.003={res["0.003"]["risk"]:.4f} AURC={res["aurc"]:.5f}', flush=True)

    # ------------------------------------------------- 5. ensemble ablations
    R['ablation_ens'] = {}
    for M in [1, 3, 5]:
        for rule in (['best'] if M == 1 else ['best', 'mean']):
            Sm, Cm, Tm = (pipeline(d, ens[:M], rule) for d in (sc_d, cal_d, te_d))
            names = FEATURE_SETS['full']
            sm = RiskScorer(names).fit(Sm['feats'], Sm['loss'])
            lam = crc_threshold(sm.score(Cm['feats']), Cm['loss'], ALPHA_MAIN)
            rn = sum_rate(Ht, Tm['p'])
            ev = evaluate_threshold(sm.score(Tm['feats']), Tm['loss'], rn, r_ref, lam)
            ev['nn_pct_ref'] = float(100 * rn.mean() / r_ref.mean())
            ev['nn_mean_regret'] = float(Tm['loss'].mean())
            R['ablation_ens'][f'M={M}, {rule}'] = ev
            print('ablation', M, rule, ev, flush=True)

    # ------------------------------------------------------- 6. OOD + online
    R['ood'] = {}
    lam0 = crc_threshold(s_cal, C['loss'], ALPHA_MAIN)
    for f in sorted(glob.glob(f'{OUT}/data_ood_*.pkl')):
        name = f.split('data_ood_')[1][:-4]
        d = pickle.load(open(f, 'rb'))
        O = pipeline(d, ens)
        so = scorer.score(O['feats'])
        rr = sum_rate(d['Ht'], d['Pref'])
        rn = sum_rate(d['Ht'], O['p'])
        g = sum_rate(d['Ht'], gnn.predict(features(d['He'])))
        off = evaluate_threshold(so, O['loss'], rn, rr, lam0)
        accs = []
        for rep in range(20):
            perm = rng.permutation(len(so))
            acc, aud, lams = online_recalibration(so[perm], O['loss'][perm], lam0, ALPHA_MAIN,
                                                  eta=0.01, beta=0.1, rng=rng)
            accs.append(dict(risk=float(np.mean(O['loss'][perm] * acc)),
                             risk_last_half=float(np.mean((O['loss'][perm] * acc)[len(so) // 2:])),
                             defer=float(1 - acc.mean()), audit=float(aud.mean())))
        on = {k: float(np.mean([a[k] for a in accs])) for k in accs[0]}
        on['risk_std'] = float(np.std([a['risk'] for a in accs]))
        R['ood'][name] = dict(K=int(d['Ht'].shape[1]), nn_pct_ref=float(100 * rn.mean() / rr.mean()),
                              gnn_pct_ref=float(100 * g.mean() / rr.mean()),
                              nn_mean_regret=float(O['loss'].mean()), offline=off, online=on)
        print('OOD', name, R['ood'][name], flush=True)
    # online on in-distribution stream as a sanity check
    acc, aud, _ = online_recalibration(s_te, T['loss'], lam0, ALPHA_MAIN, 0.01, 0.1, rng)
    R['online_in_dist'] = dict(risk=float(np.mean(T['loss'] * acc)), defer=float(1 - acc.mean()),
                               audit=float(aud.mean()))

    # ------------------------------------------------------------ 7. runtime
    n = 2000
    Hs = He[:n]
    def tm(fn, rep=3):
        best = 1e9
        for _ in range(rep):
            t = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t)
        return 1e3 * best / n  # ms per sample
    rt = {
        'WMMSE (100 it.)': tm(lambda: wmmse(Hs, iters=100, tol=0)),
        'WMMSE (5 it.)': tm(lambda: wmmse(Hs, iters=5)),
        'MLP': tm(lambda: mlp.predict(features(Hs))),
        'MPGNN': tm(lambda: gnn.predict(features(Hs))),
        'IAAN single': tm(lambda: ens[0].predict(features(Hs))),
        'CSPA front end (5 IAAN + features + scorer)': tm(
            lambda: scorer.score(select_and_features(*ensemble_outputs(ens, Hs), Hs)[1]), rep=2),
    }
    R['runtime_ms_per_sample'] = rt
    K, d, H, L, dh = 20, 32, 2, 3, 16
    layer = H * (3 * K * d * dh + 2 * K * K * dh + 2 * K * K) + K * (2 * d + 2 * H) * d + K * d * d
    iaan = 5 * K * d + L * layer + K * d
    wm_it = 3 * K * K + 6 * K
    R['complexity'] = {
        'IAAN MAC/sample': iaan, 'Ensemble(5) MAC/sample': 5 * iaan,
        'Features+probe MAC/sample': 5 * K * K + 2 * K * K + wm_it,
        'WMMSE MAC/iteration': wm_it, 'WMMSE(100) MAC/sample': 100 * wm_it,
        'IAAN sequential depth (matmul stages)': L * 4 + 2,
        'WMMSE(100) sequential depth (update stages)': 100 * 3,
    }
    print('complexity', R['complexity'], flush=True)
    print('runtime', rt, flush=True)
    json.dump(R, open(f'{OUT}/results.json', 'w'), indent=1)
    np.savez(f'{OUT}/test_scores.npz', s_te=s_te, loss=T['loss'], r_nn=r_nn, r_ref=r_ref)
    print('done')
