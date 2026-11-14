"""Uncertainty features, learned risk scorer and risk-controlled deferral for CSPA."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from .env import sum_rate, features
from .optim import wmmse
from .models import node_edge_inputs

LN2 = np.log(2)


def rate_gradient(H, p):
    """d(sum-rate)/dp for batches (bit/s/Hz per unit power fraction)."""
    d = np.diagonal(H, axis1=1, axis2=2)
    Tt = np.einsum('nkj,nj->nk', H, p) + 1.0          # total received power + noise
    I = Tt - d * p                                     # interference + noise
    Hoff = H * (1 - np.eye(H.shape[1]))
    G = H / Tt[:, :, None] - Hoff / I[:, :, None]      # dR_k/dp_j, eq. (gradient)
    return G.sum(1) / LN2


def kkt_violation(H, p, tol=1e-3):
    g = rate_gradient(H, p)
    v = (p < 1 - tol) * np.maximum(g, 0) + (p > tol) * np.maximum(-g, 0)
    return v


def ensemble_outputs(models, H_est):
    F = features(H_est)
    P = np.stack([m.predict(F) for m in models])            # (M,n,K)
    R = np.stack([sum_rate(H_est, p) for p in P])            # (M,n)
    return P, R


def select_and_features(P, R, H_est, rule='best'):
    """Return the deployed power vector and the uncertainty feature matrix."""
    M, n, K = P.shape
    if rule == 'best':
        idx = R.argmax(0)
        p_sel = P[idx, np.arange(n)]
    else:
        p_sel = P.mean(0)
    r_sel = sum_rate(H_est, p_sel)
    p_bar = P.mean(0)
    r_bar = sum_rate(H_est, p_bar)
    p1, _ = wmmse(H_est, iters=1, p0=np.clip(p_sel, 1e-6, 1))
    r1 = sum_rate(H_est, p1)
    v = kkt_violation(H_est, p_sel)
    x0, _, _, _ = node_edge_inputs(features(H_est))
    feats = {
        'rate_per_link': r_sel / K,
        'ens_rate_cv': R.std(0) / (R.mean(0) + 1e-9) if M > 1 else np.zeros(n),
        'ens_power_std': P.std(0).mean(1) if M > 1 else np.zeros(n),
        'ens_mean_gap': (r_sel - r_bar) / (r_sel + 1e-9),
        'probe_gain': (r1 - r_sel) / (r_sel + 1e-9),
        'kkt_l1': np.log1p(v.sum(1)),
        'kkt_max': np.log1p(v.max(1)),
        'frac_off': (p_sel < 0.05).mean(1),
        'max_inr': x0[:, :, 2].max(1),
        'mean_snr': x0[:, :, 0].mean(1),
    }
    return p_sel, feats


FEATURE_SETS = {
    'full': ['rate_per_link', 'ens_rate_cv', 'ens_power_std', 'ens_mean_gap', 'probe_gain',
             'kkt_l1', 'kkt_max', 'frac_off', 'max_inr', 'mean_snr'],
    'ensemble_only': ['ens_rate_cv', 'ens_power_std', 'ens_mean_gap'],
    'physics_only': ['probe_gain', 'kkt_l1', 'kkt_max'],
}


def stack(feats, names):
    return np.stack([feats[k] for k in names], 1)


class RiskScorer:
    def __init__(self, names, seed=0):
        self.names = names
        self.m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                               max_leaf_nodes=31, random_state=seed)

    def fit(self, feats, regret):
        self.m.fit(stack(feats, self.names), regret)
        return self

    def score(self, feats):
        return self.m.predict(stack(feats, self.names))


def regret(H_true, p, p_ref):
    """Normalised sum-rate regret w.r.t. the reference optimiser, clipped to [0,1]."""
    return np.clip(1 - sum_rate(H_true, p) / sum_rate(H_true, p_ref), 0.0, 1.0)


# ------------------------------------------------------------ calibration ---
def _risk_curve(scores, losses):
    o = np.argsort(scores)
    s, l = scores[o], losses[o]
    return s, np.cumsum(l) / len(l)          # risk if accepting the first i+1


def crc_threshold(scores, losses, alpha, B=1.0):
    """Conformal risk control (Angelopoulos et al., ICLR 2024) for the monotone
    loss L_i(lam) = loss_i * 1{score_i <= lam}. Guarantees E[L_test] <= alpha."""
    n = len(scores)
    s, R = _risk_curve(scores, losses)
    ok = (n * R + B) / (n + 1) <= alpha
    if not ok[0]:
        return -np.inf
    # largest prefix such that all smaller prefixes also satisfy (R monotone)
    i = np.argmax(~ok) - 1 if (~ok).any() else n - 1
    return s[i]


def naive_threshold(scores, losses, alpha):
    s, R = _risk_curve(scores, losses)
    ok = R <= alpha
    i = np.argmax(~ok) - 1 if (~ok).any() else len(s) - 1
    return s[i] if i >= 0 else -np.inf


def hp_threshold(scores, losses, alpha, delta=0.1):
    """High-probability version: Learn-then-Test with fixed-sequence testing and
    the empirical-Bernstein bound (Maurer & Pontil 2009). P(risk <= alpha) >= 1-delta."""
    n = len(scores)
    o = np.argsort(scores)
    s, l = scores[o], losses[o]
    c1 = np.cumsum(l)
    c2 = np.cumsum(l ** 2)
    m = c1 / n
    v = (c2 - n * m ** 2) / (n - 1)
    c = np.log(2 / delta)
    ucb = m + np.sqrt(2 * np.maximum(v, 0) * c / n) + 7 * c / (3 * (n - 1))
    bad = ucb > alpha
    if bad[0]:
        return -np.inf
    i = np.argmax(bad) - 1 if bad.any() else n - 1
    return s[i]


def online_recalibration(scores, losses, lam0, alpha, eta, beta, rng):
    """Adaptive risk tracking under distribution shift (in the spirit of
    Gibbs & Candes, NeurIPS 2021). A fraction beta of accepted decisions is
    audited by the reference optimiser to obtain an unbiased loss estimate."""
    lam = lam0
    acc = np.zeros(len(scores), bool)
    aud = np.zeros(len(scores), bool)
    lams = np.empty(len(scores))
    for t in range(len(scores)):
        lams[t] = lam
        acc[t] = scores[t] <= lam
        est = 0.0
        if acc[t] and rng.random() < beta:
            aud[t] = True
            est = losses[t] / beta
        lam = lam + eta * (alpha - est)
    return acc, aud, lams
