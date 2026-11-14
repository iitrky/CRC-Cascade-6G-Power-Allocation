"""Dense 6G in-X subnetwork (indoor-factory) interference channel generator.

Each of K subnetworks has one access point (AP) serving one device on a shared
sub-band. Large-scale path loss follows the 3GPP TR 38.901 InF formulas (LOS for
the short intra-subnetwork link, InF-DL NLOS for inter-subnetwork links),
log-normal shadowing and Rayleigh small-scale fading. Imperfect CSI is modelled
as a Gauss-Markov error on the small-scale coefficient.

All gains are returned normalised by the noise power and Pmax, so that a
transmit power fraction p in [0, 1] gives SINR_k = H_kk p_k / (sum_{j!=k} H_kj p_j + 1).
H[k, j] is the gain from transmitter j to receiver k.
"""
from dataclasses import dataclass, replace
import numpy as np


@dataclass
class Scenario:
    K: int = 20                 # number of subnetworks (links)
    area: float = 30.0          # side of square factory hall [m]
    r_min: float = 0.5          # min AP-device distance [m]
    r_max: float = 2.0          # subnetwork radius [m]
    min_sep: float = 2.0        # min AP-AP separation [m]
    fc_ghz: float = 10.0        # carrier frequency [GHz]
    bw_hz: float = 100e6        # bandwidth [Hz]
    nf_db: float = 9.0          # receiver noise figure [dB]
    pmax_dbm: float = 10.0      # max transmit power [dBm]
    sigma_los: float = 4.0      # LOS shadowing std [dB]
    sigma_nlos: float = 7.2     # NLOS shadowing std [dB]
    csi_err: float = 0.1        # CSI error coefficient eps in [0, 1)

    def shift(self, **kw):
        return replace(self, **kw)


def _place_aps(K, area, min_sep, rng):
    pts = []
    tries = 0
    while len(pts) < K:
        c = rng.uniform(0, area, 2)
        if all(np.hypot(*(c - q)) >= min_sep for q in pts) or tries > 20000:
            pts.append(c)
        tries += 1
    return np.array(pts)


def generate(sc: Scenario, n: int, rng: np.random.Generator):
    """Return (H_true, H_est) each of shape (n, K, K)."""
    K = sc.K
    noise_dbm = -174 + 10 * np.log10(sc.bw_hz) + sc.nf_db
    H_true = np.empty((n, K, K))
    H_est = np.empty((n, K, K))
    eye = np.eye(K, dtype=bool)
    for b in range(n):
        ap = _place_aps(K, sc.area, sc.min_sep, rng)
        r = np.sqrt(rng.uniform(sc.r_min ** 2, sc.r_max ** 2, K))
        th = rng.uniform(0, 2 * np.pi, K)
        ue = ap + np.stack([r * np.cos(th), r * np.sin(th)], 1)
        d = np.linalg.norm(ue[:, None, :] - ap[None, :, :], axis=-1)   # d[k,j]: rx k, tx j
        d = np.maximum(d, 0.3)
        pl_los = 31.84 + 21.5 * np.log10(d) + 19.0 * np.log10(sc.fc_ghz)
        pl_nlos = 18.6 + 35.7 * np.log10(d) + 20.0 * np.log10(sc.fc_ghz)
        pl_nlos = np.maximum(pl_nlos, pl_los)  # 38.901 convention: NLOS >= LOS
        sh = np.where(eye, rng.normal(0, sc.sigma_los, (K, K)), rng.normal(0, sc.sigma_nlos, (K, K)))
        pl = np.where(eye, pl_los, pl_nlos) + sh
        beta = 10 ** ((sc.pmax_dbm - pl - noise_dbm) / 10)            # large-scale SNR/INR
        h = (rng.normal(size=(K, K)) + 1j * rng.normal(size=(K, K))) / np.sqrt(2)
        e = (rng.normal(size=(K, K)) + 1j * rng.normal(size=(K, K))) / np.sqrt(2)
        h_hat = np.sqrt(1 - sc.csi_err ** 2) * h + sc.csi_err * e
        H_true[b] = beta * np.abs(h) ** 2
        H_est[b] = beta * np.abs(h_hat) ** 2
    return H_true, H_est


def sum_rate(H, p):
    """Sum spectral efficiency [bit/s/Hz] for batches. H: (n,K,K), p: (n,K)."""
    d = np.diagonal(H, axis1=1, axis2=2)
    rx = np.einsum('nkj,nj->nk', H, p)
    sinr = d * p / (rx - d * p + 1.0)
    return np.log2(1 + sinr).sum(-1)


def features(H):
    """Normalised log-domain gain tensor used by all learning models."""
    return (10 * np.log10(H + 1e-30) - 40.0) / 20.0
