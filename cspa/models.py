"""Learning-based power allocators built on the NumPy autodiff engine.

IAAN  : proposed interference-aware attention network (permutation equivariant,
        attention logits biased by incoming/outgoing cross-link gains).
MLP   : fully connected 'learning to optimize' baseline (Sun et al., IEEE TSP 2018).
GNN   : message-passing baseline in the spirit of WCGCN/IGCNet (Shen et al., IEEE JSAC 2021).
"""
import numpy as np
from .autodiff import T, param, concat, Adam


# ---------------------------------------------------------------- shared ----
def node_edge_inputs(F):
    """F: (n,K,K) normalised log gains -> node features (n,K,5), E_in, E_out, mask."""
    n, K, _ = F.shape
    eye = np.eye(K, dtype=bool)
    off = ~eye
    Fo = np.where(off, F, np.nan)
    x0 = np.stack([
        np.diagonal(F, axis1=1, axis2=2),
        np.nanmean(Fo, axis=2), np.nanmax(Fo, axis=2),     # incoming interference
        np.nanmean(Fo, axis=1), np.nanmax(Fo, axis=1),     # outgoing (caused) interference
    ], -1)
    E_in = np.where(off, F, 0.0)                     # E_in[k,j]  = gain j -> k
    E_out = np.where(off, np.swapaxes(F, 1, 2), 0.0)  # E_out[k,j] = gain k -> j
    mask = np.where(off, 0.0, -1e9)
    return x0, E_in, E_out, mask


def rate_loss(Hn, p):
    """-(sum-rate)/K as an autodiff tensor. Hn numpy (n,K,K), p tensor (n,K)."""
    n, K, _ = Hn.shape
    d = np.diagonal(Hn, axis1=1, axis2=2)
    pc = p.reshape(n, K, 1)
    rx = (T(Hn) @ pc).reshape(n, K)
    sig = p * d
    sinr = sig / (rx - sig + 1.0)
    r = (sinr + 1.0).log() * (1 / np.log(2))
    return r.sum(axis=1).mean() * (-1.0 / K)


class Model:
    def params(self):
        return [v for v in self.__dict__.values() if isinstance(v, T)] + \
               [p for L in getattr(self, 'layers', []) for p in L.values()]

    def predict(self, F, bs=512):
        out = [self.forward(F[i:i + bs]).value for i in range(0, len(F), bs)]
        return np.concatenate(out)

    def fit(self, F, H, P_ref, iters=3000, bs=128, lr=2e-3, sup_w0=1.0, sup_decay=0.5,
            rng=None, unsup=True, log_every=500, verbose=False):
        """Hybrid training: supervised imitation of WMMSE with weight decaying from
        sup_w0 to 0 over `sup_decay` fraction of iterations, plus unsupervised
        sum-rate maximisation on the estimated channel."""
        rng = rng or np.random.default_rng(0)
        opt = Adam(self.params(), lr=lr)
        n = len(F)
        for t in range(iters):
            idx = rng.integers(0, n, bs)
            p = self.forward(F[idx])
            w = sup_w0 * max(0.0, 1 - t / (sup_decay * iters)) if unsup else 1.0
            loss = T(0.0)
            if w > 0:
                diff = p - P_ref[idx]
                loss = loss + (diff * diff).mean() * (w * 10.0)
            if unsup:
                loss = loss + rate_loss(H[idx], p)
            loss.backward()
            opt.lr = lr * 0.5 * (1 + np.cos(np.pi * t / iters))
            opt.step()
            if verbose and t % log_every == 0:
                print(f'  it {t:5d} loss {float(loss.value):.4f}')
        return self


# ----------------------------------------------------------- proposed IAAN --
class IAAN(Model):
    def __init__(self, d=32, heads=2, layers=3, seed=0, gain_bias=True):
        rng = np.random.default_rng(seed)
        self.d, self.h, self.nl, self.gain_bias = d, heads, layers, gain_bias
        self.W_in = param((5, d), rng)
        self.b_in = T(np.zeros(d), requires_grad=True)
        self.layers = []
        dh = d // heads
        for _ in range(layers):
            L = {}
            for h in range(heads):
                L[f'q{h}'] = param((d, dh), rng, 1 / np.sqrt(d))
                L[f'k{h}'] = param((d, dh), rng, 1 / np.sqrt(d))
                L[f'v{h}'] = param((d, dh), rng, 1 / np.sqrt(d))
                L[f'a{h}'] = T(rng.normal(0, 0.5, 2), requires_grad=True)  # gain-bias weights
            L['wo'] = param((d + d + 2 * heads, d), rng)
            L['bo'] = T(np.zeros(d), requires_grad=True)
            L['w2'] = param((d, d), rng)
            L['b2'] = T(np.zeros(d), requires_grad=True)
            self.layers.append(L)
        self.W_h = param((d, 1), rng, 0.1)
        self.b_h = T(np.zeros(1), requires_grad=True)

    def forward(self, F):
        x0, E_in, E_out, mask = node_edge_inputs(F)
        n, K, _ = F.shape
        x = (T(x0) @ self.W_in + self.b_in).relu()
        Ein, Eout = T(E_in), T(E_out)
        dh = self.d // self.h
        for L in self.layers:
            msgs, stats = [], []
            for h in range(self.h):
                q, k, v = x @ L[f'q{h}'], x @ L[f'k{h}'], x @ L[f'v{h}']
                a = L[f'a{h}']
                logits = (q @ k.swap()) * (1 / np.sqrt(dh))
                # interference-aware bias: a0*E_in + a1*E_out (learned per head)
                if self.gain_bias:
                    logits = logits + _bias(a, Ein, Eout)
                A = (logits + T(mask)).softmax(-1)
                msgs.append(A @ v)
                stats.append((A * Ein).sum(axis=2, keepdims=True))
                stats.append((A * Eout).sum(axis=2, keepdims=True))
            z = concat([x] + msgs + stats, -1)
            u = (z @ L['wo'] + L['bo']).relu()
            x = x + (u @ L['w2'] + L['b2'])
        return (x @ self.W_h + self.b_h).reshape(n, K).sigmoid()


def _bias(a, Ein, Eout):
    """a: tensor (2,), returns a0*E_in + a1*E_out with gradient to a."""
    out = T(a.value[0] * Ein.value + a.value[1] * Eout.value, (a, Ein, Eout))

    def bw(g):
        a._acc(np.array([(g * Ein.value).sum(), (g * Eout.value).sum()]))

    out._backward = bw
    return out


# ------------------------------------------------------------- baselines ----
class MLP(Model):
    def __init__(self, K, hidden=(200, 200, 200), seed=0):
        rng = np.random.default_rng(seed)
        dims = (K * K,) + tuple(hidden) + (K,)
        self.K = K
        self.layers = [{'W': param((a, b), rng), 'b': T(np.zeros(b), requires_grad=True)}
                       for a, b in zip(dims[:-1], dims[1:])]

    def forward(self, F):
        n = len(F)
        x = T(F.reshape(n, -1))
        for i, L in enumerate(self.layers):
            x = x @ L['W'] + L['b']
            x = x.relu() if i < len(self.layers) - 1 else x.sigmoid()
        return x


class GNN(Model):
    """Message passing with edge features; mean aggregation."""

    def __init__(self, d=32, layers=3, seed=0):
        rng = np.random.default_rng(seed)
        self.d = d
        self.W_in = param((5, d), rng)
        self.b_in = T(np.zeros(d), requires_grad=True)
        self.layers = [{'wm': param((d, d), rng), 'we': param((2, d), rng, 0.3),
                        'bm': T(np.zeros(d), requires_grad=True),
                        'wu': param((2 * d, d), rng), 'bu': T(np.zeros(d), requires_grad=True)}
                       for _ in range(layers)]
        self.W_h = param((d, 1), rng, 0.1)
        self.b_h = T(np.zeros(1), requires_grad=True)

    def forward(self, F):
        x0, E_in, E_out, mask = node_edge_inputs(F)
        n, K, _ = F.shape
        x = (T(x0) @ self.W_in + self.b_in).relu()
        E = T(np.stack([E_in, E_out], -1))                     # (n,K,K,2)
        off = T((mask == 0).astype(float)[..., None] / (K - 1))
        for L in self.layers:
            src = (x @ L['wm']).reshape(n, 1, K, self.d)         # sender j
            m = (src + E @ L['we'] + L['bm']).relu() * off       # (n,K,K,d)
            agg = m.sum(axis=2)
            x = x + (concat([x, agg], -1) @ L['wu'] + L['bu']).relu()
        return (x @ self.W_h + self.b_h).reshape(n, K).sigmoid()
