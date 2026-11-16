# CRC-Cascade: Conformal Risk Controlled Cascades for Neural Power Allocation in Dense 6G Subnetworks

[![tests](https://github.com/iitrky/CRC-Cascade-6G-Power-Allocation/actions/workflows/tests.yml/badge.svg)](https://github.com/iitrky/CRC-Cascade-6G-Power-Allocation/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24271050.svg)](https://doi.org/10.5281/zenodo.24271050)

This is the official implementation of the paper

> G. Arul Dalton and S. V. J. Mani, *Conformal Risk Controlled Cascades for Neural Power Allocation in Dense 6G Subnetworks*, manuscript submitted to a Springer journal, 2026.

Neural power allocators are fast but give no guarantee on any single decision. **CRC-Cascade** routes each channel realization through a three-stage optimizer cascade:

| Stage | What runs | WMMSE iterations |
| --- | --- | --- |
| S1 | Interference-aware attention network (IAAN) ensemble, plus one WMMSE probe step | 0 (the probe is already paid for by the risk score) |
| S2 | WMMSE warm-started at the network output | 10 |
| S3 | Cold-start WMMSE (the reference optimizer) | 100 |

The routing uses a physics-informed risk score. The thresholds are calibrated by conformal risk control, so that the **expected sum-rate regret relative to WMMSE stays below a user-chosen budget α**. This holds for any distribution, provided calibration and deployment data are exchangeable. Three further tools come with the method:
- a group-wise (Mondrian) variant that keeps the budget inside every interference regime;
- a Learn-then-Test variant that gives a high-probability bound;
- an audit-driven online update that tracks the budget under distribution shift.

Everything is written in NumPy, including a small reverse-mode autodiff engine. No GPU or deep-learning framework is needed. The full study runs in about 40 minutes on a 2-core CPU.

---

## Contents

1. [Repository layout](#1-repository-layout)
2. [Installation](#2-installation)
3. [Quick start](#3-quick-start)
4. [Reproducing every table and figure](#4-reproducing-every-table-and-figure)
5. [Method overview and implementation details](#5-method-overview-and-implementation-details)
6. [Configuration and hyperparameters](#6-configuration-and-hyperparameters)
7. [Results](#7-results)
8. [Using the components in your own code](#8-using-the-components-in-your-own-code)
9. [Testing](#9-testing)
10. [Limitations](#10-limitations)
11. [Citation](#11-citation) · [License](#12-license) · [Contact](#13-contact)

A line-by-line mapping from every numbered equation (1)–(31) of the paper to the code is given in [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md).

---

## 1. Repository layout

```
CRC-Cascade-6G-Power-Allocation/
├── cspa/                      # the library
│   ├── autodiff.py            # reverse-mode autodiff on NumPy (tensor class T, concat, param init, Adam)
│   ├── env.py                 # 6G in-factory simulator: 3GPP TR 38.901 InF path loss, shadowing,
│   │                          #   Rayleigh fading, imperfect CSI; sum-rate; log-gain features
│   ├── optim.py               # WMMSE (reference), max power, random power, greedy binary on/off
│   ├── models.py              # IAAN (proposed), MLP L2O and MPGNN baselines, hybrid training loss
│   ├── selective.py           # rate gradient, KKT residual, WMMSE probe, risk features, risk scorer,
│   │                          #   CRC / Learn-then-Test (empirical Bernstein) / naive calibration,
│   │                          #   audit-driven online recalibration
│   └── cascade.py             # 3-stage cascade, monotone loss envelope, cascade CRC/LTT,
│                              #   path-offset selection, group-wise (Mondrian) CRC
├── scripts/
│   ├── train.py               # stage 1: generate all data sets (fixed seeds) and train all models
│   ├── evaluate.py            # stage 2: single-fallback experiments (Tables 4-9, Figs 3-4)
│   ├── evaluate_cascade.py    # stage 2b: cascade and per-regime experiments (Tables 4, 10, 11)
│   ├── cascade_online.py      # stage 2c: cascade under distribution shift with online recalibration
│   └── make_figures.py        # stage 3: 300-dpi figures
├── tests/
│   ├── test_autodiff.py       # finite-difference gradient check of IAAN, MLP and MPGNN
│   └── test_pipeline.py       # 3-second end-to-end smoke test of the whole pipeline
├── results/
│   ├── results.json           # every number reported in the paper (single-fallback experiments)
│   ├── results_cascade.json   # every number reported in the paper (cascade experiments)
│   └── figs/                  # figures exactly as they appear in the paper
├── docs/
│   ├── IMPLEMENTATION.md      # equation-to-code map, data flow, design decisions
│   └── figures/               # system-model diagrams (Figs 1-2 of the paper)
├── reproduce.sh               # one command to regenerate everything
├── requirements.txt
├── pyproject.toml
├── CITATION.cff
└── LICENSE
```

## 2. Installation

```bash
git clone https://github.com/iitrky/CRC-Cascade-6G-Power-Allocation.git
cd CRC-Cascade-6G-Power-Allocation
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional: install as a package (editable) and add pytest
pip install -e ".[dev]"
```

Requirements are Python ≥ 3.10, `numpy ≥ 1.24`, `scipy ≥ 1.10`, `scikit-learn ≥ 1.3` and `matplotlib ≥ 3.7`. The code was developed and tested on Linux with Python 3.12. It is pure Python, so it also runs on macOS and Windows.

## 3. Quick start

```bash
python -m pytest -q                 # 3 tests, about 5 s
python -m tests.test_pipeline       # tiny end-to-end run of the full method
```

A minimal example of the complete pipeline on a small problem (K = 8) is in [`tests/test_pipeline.py`](tests/test_pipeline.py). It generates channels, trains a 2-member IAAN ensemble, fits the risk scorer, calibrates the cascade with CRC and routes test samples through S1/S2/S3.

## 4. Reproducing every table and figure

```bash
bash reproduce.sh                   # about 40 min on 2 CPU cores, about 0.5 GB of disk
```

or step by step:

| Step | Command | Time (2 cores) | Writes |
| --- | --- | --- | --- |
| 0 | `python -m tests.test_autodiff` | < 1 min | – |
| 1 | `python -m scripts.train` | about 15 min | `results/data_*.pkl`, `results/model_*.pkl` |
| 2 | `python -m scripts.evaluate` | about 10 min | `results/results.json`, `results/test_scores.npz` |
| 3 | `python -m scripts.evaluate_cascade` | about 10 min | `results/results_cascade.json` |
| 4 | `python -m scripts.cascade_online` | about 3 min | adds `cascade_bounded`, `cascade_ood_online` to `results_cascade.json` |
| 5 | `python -m scripts.make_figures` | < 1 min | `results/figs/*.png` |

Set `CSPA_OUT=<folder>` to write somewhere other than `results/`. All random seeds are fixed (Section 6), so a rerun reproduces the stored JSON files up to floating-point and library-version effects.

**Where each paper item comes from**

| Paper item | Content | Script | JSON key |
| --- | --- | --- | --- |
| Table 3 | Simulation parameters | `cspa/env.py` (`Scenario`) | – |
| Table 4 | Sum SE and regret of all allocators | `evaluate.py`, `evaluate_cascade.py`, `cascade_online.py` | `baselines`, `cspa`, `fixed`, `cascade_bounded` |
| Table 5 | Realized regret over 300 calibration/test resplits (CRC, LTT-EB, naive) | `evaluate.py` | `validity` |
| Table 6 | Deferral needed to meet the budget; AURC by risk score | `evaluate.py` | `score_cmp` |
| Table 7 | Ensemble size and selection rule ablation | `evaluate.py` | `ablation_ens` |
| Table 8 | OOD deployment, offline vs online recalibration | `evaluate.py` | `ood`, `online_in_dist` |
| Table 9 | Complexity and runtime per realization | `evaluate.py` | `complexity`, `runtime_ms_per_sample` |
| Table 10 | Single fallback vs certified cascade | `evaluate_cascade.py` | `single`, `single_probe`, `cascade`, `cascade_validity` |
| Table 11 | Per-regime regret: marginal vs group-wise CRC | `evaluate_cascade.py` | `conditional` |
| Fig. 3 | Risk–deferral curves | `make_figures.py` | `curves` |
| Fig. 4 | OOD regret, offline vs online | `make_figures.py` | `ood` |
| Fig. 5 | Cascade cost vs α | `make_figures.py` | `cascade`, `single` |
| Fig. S1 | Validity over resplits (supplementary) | `make_figures.py` | `validity` |
| Sec. 7 text | Cascade under shift, offline and online | `evaluate_cascade.py`, `cascade_online.py` | `cascade_ood`, `cascade_ood_online` |
| Figs 1–2 | System-model diagrams | drawn by the authors | `docs/figures/` |

## 5. Method overview and implementation details

### 5.1 System model (`cspa/env.py`)
- **K subnetworks** are placed in a square hall, with a minimum AP–AP separation of 2 m. Each AP serves one device at distance 0.5–2 m.
- **Path loss** follows 3GPP TR 38.901 InF: LOS for the serving links and NLOS for the interfering links, with NLOS ≥ LOS. Log-normal shadowing is σ = 4 dB (LOS) and 7.2 dB (NLOS). Small-scale fading is Rayleigh.
- **Channel gains** are normalized by the noise power, so G_kj is the received SNR or INR at full power. The power fraction is p_k ∈ [0, 1].
- **Imperfect CSI:** ĥ = √(1−ε²) h + ε e with ε = 0.1. Every decision uses the estimated gains Ĝ. All rates and regrets are evaluated on the true gains G.
- **Learning features:** `features(H) = (10 log10 H − 40) / 20`, the normalized log gains.

### 5.2 Reference optimizer (`cspa/optim.py`)
`wmmse(H, iters=100, p0=None)` is the batched SISO WMMSE of Shi et al. (2011), with a convergence tolerance of 1e-6 on √p. Passing `p0` warm-starts it, which is used by cascade stage S2.

### 5.3 Proposed allocator: IAAN (`cspa/models.py`)
- **Node token** x_k⁽⁰⁾ (5 features): the direct gain, plus the mean and max of incoming and outgoing interference (`node_edge_inputs`).
- **Edge features** E^in_kj = F_kj and E^out_kj = F_jk. The diagonal is masked out of attention.
- **L = 3 layers, H = 2 heads, width d = 32.** The attention logits get a learned interference bias θ_h1 E^in + θ_h2 E^out. Each head also returns attention-weighted interference statistics. A residual MLP update follows (paper Eqs. 13–15).
- **Output:** p_k = sigmoid(w·x_k⁽ᴸ⁾ + b). The model is permutation-equivariant and independent of K.
- **Hybrid loss** (Eq. 16): negative sum rate on Ĝ, plus a WMMSE-imitation term whose weight decays linearly to zero at mid-training. Training uses Adam with a cosine learning-rate schedule, gradient clipping at 5, batch size 128 and 4,000 iterations.
- **Ensemble:** M = 5 members trained on bootstrap resamples with seeds 0–4. The deployed output is the member with the highest estimated sum rate (Eq. 17).
- **Baselines:** an MLP L2O (3 × 200 hidden units, supervised, Sun et al. 2018), an MPGNN (3 layers, edge-feature messages, mean aggregation), and IAAN without the interference bias (ablation).

### 5.4 Physics-informed risk score (`cspa/selective.py`)
`select_and_features` builds the 10-dimensional feature vector φ(Ĝ):

| Group | Features |
| --- | --- |
| Ensemble disagreement | coefficient of variation of member rates, mean per-link power std, relative gain over the ensemble mean |
| KKT certificate | log(1 + Σ_j v_j), log(1 + max_j v_j), with v from the closed-form rate gradient (Eqs. 18–19) |
| Optimizer probe | relative gain ρ of one warm-started WMMSE step (Eq. 20) |
| Context | rate per link, fraction of links switched off, max INR, mean SNR |

`RiskScorer` is a `HistGradientBoostingRegressor` (300 iterations, learning rate 0.05, 31 leaves) fitted to the observed regret on a separate scorer split D_sc (Eq. 21). The splits are disjoint, so the conformal guarantee is unaffected.

### 5.5 Risk-controlled selective prediction (`cspa/selective.py`)
- **Loss:** ℓ = clip(1 − R(p̂; G) / R(p_ref; G), 0, 1), the sum-rate regret relative to WMMSE (Eq. 9).
- **`crc_threshold`:** conformal risk control (Eq. 22). It finds the largest λ with (n R̂_n(λ′) + B)/(n + 1) ≤ α for all λ′ ≤ λ, where B = 1. This guarantees E[ℓ·1{s ≤ λ̂}] ≤ α (Eq. 23).
- **`hp_threshold`:** Learn-then-Test with fixed-sequence testing and the empirical-Bernstein bound of Maurer & Pontil (Eq. 24). This guarantees Pr[risk ≤ α] ≥ 1 − δ, with δ = 0.1.
- **`naive_threshold`:** the empirical plug-in rule, included as an uncalibrated baseline.
- **`online_recalibration`:** λ_{t+1} = λ_t + η(α − L̃_t). Here L̃_t is an inverse-propensity estimate built from audits: each accepted decision is checked against WMMSE with probability β_a = 0.1 (Eq. 25). The long-run regret bound is Proposition 1 (Eq. 26). Defaults are η = 0.01 and β_a = 0.1.

### 5.6 CRC-Cascade (`cspa/cascade.py`)
- **`stage_outputs`** computes the S1 output (network plus one probe step, kept only if it improves the rate) and the S2 output (10-iteration warm WMMSE, kept only if it improves). It also returns their true-channel regrets l1 and l2.
- **Routing** (`assign`, Eq. 27): a one-parameter path with τ₃ = t and τ₂ = t − a. A sample goes to S1 if s ≤ t − a, to S2 if t − a < s ≤ t, and to S3 otherwise.
- **Monotone envelope** (Eq. 28): ℓ̄₁ = max(l1, l2), ℓ̄₂ = l2, ℓ̄₃ = 0. This makes each sample's loss non-decreasing in t, so a single-scalar CRC certifies the whole cascade (Theorem 2). `calibrate_t` implements Eq. 29 exactly, by sweeping the 2n breakpoints {s_i, s_i + a}.
- **`choose_offset`** (Eq. 30) picks the offset a from a grid on the independent scorer split, minimizing the expected WMMSE iterations. This keeps the calibration split untouched.
- **`groupwise_crc`** (Eq. 31, Corollary 1) calibrates one threshold per interference regime. The regimes are terciles of the estimated rate per link, so the budget holds within every regime.

The full derivations and proofs are in Sections 4–5 of the paper. [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) gives the equation-by-equation code map.

## 6. Configuration and hyperparameters

| Item | Value | Where |
| --- | --- | --- |
| Subnetworks K, hall size | 20, 30 m × 30 m | `env.Scenario` |
| Carrier, bandwidth, noise figure, P_max | 10 GHz, 100 MHz, 9 dB, 10 dBm | `env.Scenario` |
| Shadowing σ_LOS / σ_NLOS | 4 / 7.2 dB | `env.Scenario` |
| CSI error ε | 0.1 | `env.Scenario` |
| Train / scorer / calibration / test sizes | 20,000 / 5,000 / 5,000 / 10,000 | `scripts/train.py` |
| Data seeds | 1 / 2 / 3 / 4; OOD sets 50–54 | `scripts/train.py` |
| OOD shifts (6,000 samples each) | K = 30, K = 10, 20 m hall, ε = 0.3, heavy shadowing (6 / 10 dB) | `scripts/train.py` |
| IAAN | L = 3, H = 2, d = 32, 4,000 iterations, batch 128, lr 2e-3 (cosine), M = 5 | `models.IAAN`, `scripts/train.py` |
| MPGNN / MLP | 3 layers, d = 32, 3,000 iterations, lr 2e-3 / 3 × 200 hidden, 8,000 iterations, lr 1e-3 (supervised) | `scripts/train.py` |
| WMMSE reference / warm stage | 100 / 10 iterations | `optim.wmmse`, `evaluate_cascade.WARM` |
| Risk budgets α | 0.001–0.005 (main: 0.003) | `scripts/evaluate*.py` |
| LTT confidence δ | 0.1 | `selective.hp_threshold` |
| Online step η, audit rate β_a | 0.01, 0.1 | `scripts/evaluate.py`, `scripts/cascade_online.py` |
| Calibration/test resplits | 300 | `scripts/evaluate*.py` |
| Offset grid for a | {0, 0.001, …, 0.05, 1.0} | `evaluate_cascade.GRID` |

To run a different scenario, edit `Scenario` fields or call `Scenario().shift(K=40, csi_err=0.2)`.

## 7. Results

Main results on the in-distribution test set (10,000 realizations, K = 20, α = 0.3%). These are the numbers stored in `results/*.json`:

| Allocator | Sum SE [bit/s/Hz] | % of WMMSE | Mean regret | WMMSE effort |
| --- | --- | --- | --- | --- |
| WMMSE (100 it., reference) | 55.15 | 100.0 | 0 | 100 it./cycle |
| IAAN ensemble (no fallback) | 55.87 | 101.3 | 0.54% | 0 |
| CSPA single fallback, CRC | 55.87 | 101.3 | 0.28% | 19.8% of cycles deferred |
| **CRC-Cascade, CRC** | **56.21** | **101.9** | **0.27%** | **3.9 it./cycle** |

Further findings reported in the paper:
- **Cost savings:** the cascade needs 28–100% fewer WMMSE iterations than the single fallback over α = 0.1–0.5%. Its realized risk stays below α over 300 resplits.
- **Better risk score:** the physics-informed score defers 19.8% of cycles, against 30.6% for an epistemic-interval heuristic.
- **Per-regime guarantee:** group-wise CRC cuts the violation rate in the interference-limited regime from 98% to 1.7%.
- **Distribution shift:** online recalibration keeps the realized risk close to α under all five shifts.

The network can exceed 100% of WMMSE because WMMSE only reaches a stationary point of the non-convex problem.

## 8. Using the components in your own code

```python
import numpy as np
from cspa.env import Scenario, generate, features, sum_rate
from cspa.optim import wmmse
from cspa.models import IAAN
from cspa.selective import ensemble_outputs, select_and_features, RiskScorer, FEATURE_SETS, regret
from cspa.cascade import stage_outputs, calibrate_t, assign, cost

sc = Scenario(K=20)
Ht, He = generate(sc, 2000, np.random.default_rng(1))         # true and estimated gains
Pref, _ = wmmse(He)                                              # reference allocations
ens = [IAAN(seed=s).fit(features(He), He, Pref, iters=500, rng=np.random.default_rng(s)) for s in range(3)]

def split(seed, n=1000):
    ht, he = generate(sc, n, np.random.default_rng(seed)); pr, _ = wmmse(he)
    P, R = ensemble_outputs(ens, he); p, f = select_and_features(P, R, he)
    return dict(ht=ht, he=he, pr=pr, p=p, f=f, loss=regret(ht, p, pr))

S, C, T = split(2), split(3), split(4)
scorer = RiskScorer(FEATURE_SETS['full']).fit(S['f'], S['loss'])
stC = stage_outputs(C['he'], C['ht'], C['p'], C['pr'], warm_iters=10)
t = calibrate_t(scorer.score(C['f']), stC, a=0.01, alpha=0.003)  # certified threshold
k = assign(scorer.score(T['f']), t, a=0.01)                      # 0 = S1, 1 = S2, 2 = S3
print('stage shares', np.bincount(k, minlength=3) / len(k), 'mean WMMSE it.', cost(k, 10).mean())
```

## 9. Testing

```bash
python -m pytest -q
```
- `test_autodiff.py` compares analytic and numerical gradients for IAAN, MLP and MPGNN. The relative error must be below 1e-3.
- `test_pipeline.py` runs the entire method on a tiny problem. It checks loss ranges, that LTT is at least as conservative as CRC, that the envelope loss is monotone in t (the condition behind Theorem 2), and the shapes of the group-wise and online outputs.

GitHub Actions runs both tests on every push (`.github/workflows/tests.yml`).

## 10. Limitations

- All results come from simulation. There is no over-the-air or hardware test bed.
- The guarantee is **relative to WMMSE**, not to the global optimum of the non-convex sum-rate problem.
- The guarantee is **marginal**, in expectation over the calibration draw. Per-regime validity needs the group-wise variant. Validity under shift is long-run only (Proposition 1), and requires audit feedback.
- Runtimes were measured on a 2-core CPU and are for relative comparison only. The ensemble uses more multiply–accumulate operations than WMMSE (reported honestly in Table 9). The benefit is lower sequential depth and fewer WMMSE iterations.

## 11. Citation

If you use this code, please cite the paper and the archived software (Zenodo DOI [10.5281/zenodo.24271050](https://doi.org/10.5281/zenodo.24271050)). The entry will be updated with volume, pages and DOI after publication.

```bibtex
@article{dalton2026crccascade,
  author  = {Dalton, G. Arul and Mani, S. V. J.},
  title   = {Conformal Risk Controlled Cascades for Neural Power Allocation in Dense {6G} Subnetworks},
  journal = {Manuscript submitted for publication},
  year    = {2026}
}

@software{dalton2026crccascade_code,
  author    = {Dalton, G. Arul and Mani, S. V. J.},
  title     = {{CRC-Cascade}: Conformal Risk Controlled Cascades for Neural Power Allocation in Dense {6G} Subnetworks},
  version   = {1.0.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24271050},
  url       = {https://doi.org/10.5281/zenodo.24271050}
}
```

GitHub also reads [`CITATION.cff`](CITATION.cff) ("Cite this repository" button).

## 12. License

Released under the [MIT License](LICENSE).

## 13. Contact

- G. Arul Dalton (corresponding author), Department of Computer Science and Engineering, Saveetha Engineering College, Chennai, India. aruldalton@saveetha.ac.in. ORCID [0000-0002-5981-7176](https://orcid.org/0000-0002-5981-7176)
- S. V. J. Mani, Department of Computer Science Engineering and Cyber Security, Saveetha Engineering College, Chennai, India. manisvj@saveetha.ac.in. ORCID [0009-0008-0957-6301](https://orcid.org/0009-0008-0957-6301)

Please open a GitHub issue for bugs or questions about the code.
