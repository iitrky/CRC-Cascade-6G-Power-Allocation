"""Minimal reverse-mode automatic differentiation on NumPy arrays.

Only the operations needed by CSPA and the learning baselines are implemented.
Every op is verified against finite differences in tests/test_autodiff.py.
"""
import numpy as np


def _unbroadcast(grad, shape):
    """Sum a broadcast gradient back to `shape`."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class T:
    """Tensor node holding a value, a gradient and a backward closure."""

    def __init__(self, value, parents=(), backward=None, requires_grad=False):
        self.value = np.asarray(value, dtype=np.float64)
        self.grad = None
        self.parents = parents
        self._backward = backward
        self.requires_grad = requires_grad or any(p.requires_grad for p in parents)

    @property
    def shape(self):
        return self.value.shape

    # ---- graph traversal -------------------------------------------------
    def backward(self, grad=None):
        topo, seen = [], set()

        def build(v):
            if id(v) in seen:
                return
            seen.add(id(v))
            for p in v.parents:
                build(p)
            topo.append(v)

        build(self)
        self.grad = np.ones_like(self.value) if grad is None else grad
        for v in reversed(topo):
            if v._backward is not None and v.grad is not None:
                v._backward(v.grad)

    def _acc(self, g):
        if not self.requires_grad:
            return
        self.grad = g if self.grad is None else self.grad + g

    # ---- arithmetic -------------------------------------------------------
    def __add__(self, o):
        o = o if isinstance(o, T) else T(o)
        out = T(self.value + o.value, (self, o))
        out._backward = lambda g: (self._acc(_unbroadcast(g, self.shape)),
                                   o._acc(_unbroadcast(g, o.shape)))
        return out

    __radd__ = __add__

    def __sub__(self, o):
        o = o if isinstance(o, T) else T(o)
        return self + (o * -1.0)

    def __rsub__(self, o):
        return T(o) - self

    def __mul__(self, o):
        o = o if isinstance(o, T) else T(o)
        out = T(self.value * o.value, (self, o))
        out._backward = lambda g: (self._acc(_unbroadcast(g * o.value, self.shape)),
                                   o._acc(_unbroadcast(g * self.value, o.shape)))
        return out

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = o if isinstance(o, T) else T(o)
        out = T(self.value / o.value, (self, o))
        out._backward = lambda g: (
            self._acc(_unbroadcast(g / o.value, self.shape)),
            o._acc(_unbroadcast(-g * self.value / o.value ** 2, o.shape)))
        return out

    def __matmul__(self, o):
        o = o if isinstance(o, T) else T(o)
        out = T(np.matmul(self.value, o.value), (self, o))

        def bw(g):
            ga = np.matmul(g, np.swapaxes(o.value, -1, -2))
            gb = np.matmul(np.swapaxes(self.value, -1, -2), g)
            self._acc(_unbroadcast(ga, self.shape))
            o._acc(_unbroadcast(gb, o.shape))

        out._backward = bw
        return out

    # ---- unary ops ----------------------------------------------------------
    def relu(self):
        out = T(np.maximum(self.value, 0.0), (self,))
        out._backward = lambda g: self._acc(g * (self.value > 0))
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.value, -40, 40)))
        out = T(s, (self,))
        out._backward = lambda g: self._acc(g * s * (1 - s))
        return out

    def log(self):
        out = T(np.log(self.value), (self,))
        out._backward = lambda g: self._acc(g / self.value)
        return out

    def softmax(self, axis=-1):
        z = self.value - self.value.max(axis=axis, keepdims=True)
        e = np.exp(z)
        s = e / e.sum(axis=axis, keepdims=True)
        out = T(s, (self,))
        out._backward = lambda g: self._acc(s * (g - (g * s).sum(axis=axis, keepdims=True)))
        return out

    def sum(self, axis=None, keepdims=False):
        out = T(self.value.sum(axis=axis, keepdims=keepdims), (self,))

        def bw(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._acc(np.broadcast_to(g, self.shape).copy())

        out._backward = bw
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.value.size if axis is None else self.value.shape[axis]
        return self.sum(axis, keepdims) * (1.0 / n)

    def swap(self):
        out = T(np.swapaxes(self.value, -1, -2), (self,))
        out._backward = lambda g: self._acc(np.swapaxes(g, -1, -2))
        return out

    def reshape(self, *shape):
        out = T(self.value.reshape(*shape), (self,))
        out._backward = lambda g: self._acc(g.reshape(self.shape))
        return out


def concat(ts, axis=-1):
    vals = [t.value for t in ts]
    out = T(np.concatenate(vals, axis=axis), tuple(ts))
    sizes = np.cumsum([v.shape[axis] for v in vals])[:-1]

    def bw(g):
        for t, gi in zip(ts, np.split(g, sizes, axis=axis)):
            t._acc(gi)

    out._backward = bw
    return out


def param(shape, rng, scale=None):
    fan_in = shape[0] if len(shape) > 1 else 1
    scale = np.sqrt(2.0 / fan_in) if scale is None else scale
    return T(rng.normal(0, scale, size=shape), requires_grad=True)


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, clip=5.0):
        self.p, self.lr, self.b1, self.b2, self.eps, self.clip = params, lr, b1, b2, eps, clip
        self.m = [np.zeros_like(p.value) for p in params]
        self.v = [np.zeros_like(p.value) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        gn = np.sqrt(sum(float((p.grad ** 2).sum()) for p in self.p if p.grad is not None))
        sc = min(1.0, self.clip / (gn + 1e-12))
        for i, p in enumerate(self.p):
            if p.grad is None:
                continue
            g = p.grad * sc
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p.value -= self.lr * mh / (np.sqrt(vh) + self.eps)
            p.grad = None

    def zero(self):
        for p in self.p:
            p.grad = None
