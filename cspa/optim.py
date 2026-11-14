"""Model-based power allocation baselines (vectorised over a batch)."""
import numpy as np


def wmmse(H, iters=100, p0=None, tol=1e-6):
    """WMMSE power control for the SISO interference channel (Shi et al., IEEE TSP 2011).

    H: (n,K,K) normalised gains, returns p in [0,1]^K and the iteration count used.
    """
    n, K, _ = H.shape
    h = np.sqrt(H)
    hd = np.diagonal(h, axis1=1, axis2=2)
    v = np.ones((n, K)) if p0 is None else np.sqrt(p0)
    for it in range(iters):
        rx = np.einsum('nkj,nj->nk', H, v ** 2) + 1.0
        u = hd * v / rx
        w = 1.0 / np.maximum(1.0 - u * hd * v, 1e-12)
        den = np.einsum('nj,nj,njk->nk', w, u ** 2, H)
        v_new = np.clip(w * u * hd / np.maximum(den, 1e-30), 0.0, 1.0)
        if np.max(np.abs(v_new - v)) < tol:
            v = v_new
            break
        v = v_new
    return v ** 2, it + 1


def max_power(H):
    return np.ones(H.shape[:2])


def random_power(H, rng):
    return rng.uniform(0, 1, H.shape[:2])


def binary_greedy(H, rate_fn):
    """Greedy on/off scheduling: switch off the link whose removal helps most."""
    n, K, _ = H.shape
    p = np.ones((n, K))
    best = rate_fn(H, p)
    improved = np.ones(n, bool)
    while improved.any():
        improved[:] = False
        cand_best, cand_k = best.copy(), -np.ones(n, int)
        for k in range(K):
            q = p.copy()
            q[:, k] = 0
            r = rate_fn(H, q)
            m = (r > cand_best + 1e-9) & (p[:, k] > 0)
            cand_best[m], cand_k[m] = r[m], k
        idx = np.where(cand_k >= 0)[0]
        p[idx, cand_k[idx]] = 0
        best = cand_best
        improved[idx] = True
    return p
