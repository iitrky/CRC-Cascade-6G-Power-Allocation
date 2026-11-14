"""Risk-controlled optimizer cascade and interference-conditional risk control.

Stages (cheapest first), all driven by the same risk score s:
  S1  network + one WMMSE probe step (free: the probe is computed for the score)
  S2  WMMSE warm-started at the network output, I_w iterations
  S3  cold-start WMMSE, I_ref iterations (the reference: zero regret by definition)
Thresholds follow the one-parameter path tau3 = t, tau2 = t - a (a >= 0):
  s <= t - a -> S1,  t - a < s <= t -> S2,  s > t -> S3.
Monotone envelope losses make the per-sample loss non-decreasing in t, so conformal
risk control applies to the whole cascade with a single scalar.
"""
import numpy as np
from .env import sum_rate
from .optim import wmmse
from .selective import regret, crc_threshold


def stage_outputs(H_est, H_true, p_hat, p_ref, warm_iters=10):
    p0 = np.clip(p_hat, 1e-6, 1)
    p1, _ = wmmse(H_est, iters=1, p0=p0)
    keep = (sum_rate(H_est, p1) >= sum_rate(H_est, p_hat))[:, None]
    p1 = np.where(keep, p1, p_hat)
    p2, _ = wmmse(H_est, iters=warm_iters, p0=np.clip(p1, 1e-6, 1))
    keep = (sum_rate(H_est, p2) >= sum_rate(H_est, p1))[:, None]
    p2 = np.where(keep, p2, p1)
    l1, l2 = regret(H_true, p1, p_ref), regret(H_true, p2, p_ref)
    return dict(p1=p1, p2=p2, l1=l1, l2=l2, env1=np.maximum(l1, l2), env2=l2,
                r1=sum_rate(H_true, p1), r2=sum_rate(H_true, p2), r3=sum_rate(H_true, p_ref))


def assign(s, t, a):
    """0 -> S1, 1 -> S2, 2 -> S3."""
    return np.where(s <= t - a, 0, np.where(s <= t, 1, 2))


def path_losses(st, s, t, a, which='env'):
    k = assign(s, t, a)
    L1 = st['env1'] if which == 'env' else st['l1']
    L2 = st['env2'] if which == 'env' else st['l2']
    return np.where(k == 0, L1, np.where(k == 1, L2, 0.0)), k


def cost(k, warm_iters, ref_iters=100):
    return np.where(k == 0, 0, np.where(k == 1, warm_iters, ref_iters))


def calibrate_t(s, st, a, alpha, method='crc', delta=0.1):
    """Largest t whose envelope risk is certified (scan over candidate t = score values)."""
    # envelope loss is non-decreasing in t; reuse the 1-D calibration by sorting per-sample
    # 'activation' events: a sample's loss as a function of t is a step function with steps
    # at s_i (S3->S2) and s_i + a (S2->S1).
    n = len(s)
    ev_t = np.concatenate([s, s + a])
    ev_d = np.concatenate([st['env2'], st['env1'] - st['env2']])   # loss increments
    o = np.argsort(ev_t, kind='stable')
    t_sorted, inc = ev_t[o], ev_d[o]
    R = np.cumsum(inc) / n                                        # empirical risk at t_sorted
    if method == 'crc':
        ok = (n * R + 1.0) / (n + 1) <= alpha
    else:
        c = np.log(2 / delta)
        # empirical-Bernstein UCB evaluated on a thinned grid for speed
        idx = np.unique(np.linspace(0, len(t_sorted) - 1, 400).astype(int))
        last = -1
        for i in idx:
            L, _ = path_losses(st, s, t_sorted[i], a)
            ucb = L.mean() + np.sqrt(2 * L.var(ddof=1) * c / n) + 7 * c / (3 * (n - 1))
            if ucb > alpha:
                break
            last = i
        return t_sorted[last] if last >= 0 else -np.inf
    if not ok[0]:
        return -np.inf
    i = np.argmax(~ok) - 1 if (~ok).any() else len(ok) - 1
    return t_sorted[i]


def choose_offset(s, st, alpha, warm_iters, grid):
    """Pick the path offset a on the (independent) scorer set to minimise expected cost."""
    best = None
    for a in grid:
        t = calibrate_t(s, st, a, alpha)
        k = assign(s, t, a)
        c = cost(k, warm_iters).mean()
        if best is None or c < best[1]:
            best = (a, c)
    return best[0]


def groupwise_crc(s_cal, l_cal, g_cal, alpha, G):
    return np.array([crc_threshold(s_cal[g_cal == i], l_cal[g_cal == i], alpha) for i in range(G)])
