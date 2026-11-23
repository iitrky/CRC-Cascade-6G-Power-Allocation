# Implementation notes

This document links every numbered equation, algorithm and result of the paper to the code. It also records the design decisions a reviewer or re-implementer is likely to ask about.

## 1. Data flow

```
                 scripts/train.py                                  scripts/evaluate*.py
 Scenario ──► generate() ──► (G, Ĝ) ──► wmmse(Ĝ) = p_ref           ensemble_outputs()  ─► P (M×n×K), R
   (env)        seeds 1-4, 50-54        (100 it.)                  select_and_features() ─► p̂, φ(Ĝ)
                    │                                              regret(G, p̂, p_ref)  ─► ℓ
                    └──► IAAN ×5 (bootstrap), IAAN-no-bias,         RiskScorer.fit(φ_sc, ℓ_sc) ─► s(·)
                         MPGNN, MLP   (models.fit, hybrid loss)     ┌ single fallback: crc/hp/naive_threshold
                                                                   └ cascade: stage_outputs ─► calibrate_t ─► assign
                                                                     online:  online_recalibration
```

Data splits are disjoint and independently generated:

| Split | Size | Seed | Used for |
| --- | --- | --- | --- |
| train | 20,000 | 1 | fitting allocators only |
| scorer (D_sc) | 5,000 | 2 | fitting the risk scorer and choosing the cascade offset a |
| calibration (D_cal) | 5,000 | 3 | conformal calibration only |
| test | 10,000 | 4 | reporting only |
| OOD × 5 | 6,000 each | 50–54 | shift experiments |

The resplit experiments (Tables 7, 8 and S1) pool calibration and test data, then redraw a random calibration/test partition 300 times. This is the exchangeable setting assumed by the guarantees.

## 2. Equation-to-code map

| Eq. | Meaning | Code |
| --- | --- | --- |
| (1) | Received signal model | implicit in `env.generate` (gains only) |
| (2) | Noise-normalized gain G_kj | `env.generate`: `beta * abs(h)**2` |
| (3)–(4) | 3GPP TR 38.901 InF LOS / NLOS path loss, NLOS ≥ LOS | `env.generate`: `pl_los`, `pl_nlos` |
| (5) | Imperfect CSI ĥ = √(1−ε²)h + εe | `env.generate`: `h_hat` |
| (6)–(7) | SINR and sum spectral efficiency | `env.sum_rate`; differentiable version `models.rate_loss` |
| (8) | Problem P1 (max sum rate on Ĝ) | solved by `optim.wmmse` (reference) |
| (9) | Regret loss ℓ ∈ [0, 1] | `selective.regret` |
| (10) | Selective policy π_λ | `scripts/evaluate.py: evaluate_threshold` |
| (11)–(12) | Problem P2 and expected-cost model | calibration routines + `cascade.cost` |
| (13) | Node token x_k⁽⁰⁾ and edge features E^in, E^out | `models.node_edge_inputs` |
| (14) | Interference-biased attention logits and softmax over j ≠ k | `models.IAAN.forward` (`logits`, `_bias`, `mask`) |
| (15) | Stacked messages + statistics z_k⁽ℓ⁾ and residual update | `models.IAAN.forward` (`concat`, `wo`, `w2`) |
| (16) | Hybrid loss (sum rate + decaying WMMSE imitation) | `models.Model.fit` (`w`, `rate_loss`) |
| (17) | Best-of-M ensemble selection | `selective.select_and_features(rule='best')` |
| (18) | Closed-form rate gradient g_j | `selective.rate_gradient` |
| (19) | KKT violation v_j | `selective.kkt_violation` |
| (20) | One-step WMMSE probe gain ρ | `selective.select_and_features` (`probe_gain`) |
| (21) | Risk scorer fitted on D_sc | `selective.RiskScorer` |
| (22)–(23) | Conformal risk control threshold and guarantee | `selective.crc_threshold` |
| (24) | Learn-then-Test with empirical Bernstein | `selective.hp_threshold`; cascade: `cascade.calibrate_t(method='ltt')` |
| (25)–(26) | Audit-driven online update; Proposition 1 | `selective.online_recalibration`; cascade version in `scripts/cascade_online.py` |
| (27) | Cascade routing k(s; t, a) | `cascade.assign` |
| (28) | Monotone loss envelope ℓ̄ | `cascade.stage_outputs` (`env1`, `env2`), `cascade.path_losses` |
| (29) | Cascade CRC threshold t̂ (Theorem 2) | `cascade.calibrate_t(method='crc')` |
| (30) | Offset selection a⋆ on D_sc | `cascade.choose_offset` |
| (31) | Group-wise (Mondrian) CRC, Corollary 1 | `cascade.groupwise_crc` |

**Algorithm 1 (CRC-Cascade)** corresponds to lines 1–5 (offline: `RiskScorer.fit`, `stage_outputs`, `choose_offset`, `calibrate_t`) and lines 6–14 (online: `select_and_features` → `scorer.score` → `assign` → stage execution, with optional `online_recalibration` on audited samples).

## 3. Design decisions

1. **NumPy autodiff instead of PyTorch.** This keeps the artifact dependency-light and fully inspectable. `cspa/autodiff.py` implements the few operations needed (add, mul, matmul with broadcasting, relu, sigmoid, log, softmax, sum, concat, reshape) and Adam with gradient clipping. `tests/test_autodiff.py` checks every model against finite differences.
2. **Why envelope losses (Eq. 28).** The actual stage regrets l1 and l2 are not ordered: warm WMMSE can occasionally do worse than the probe output on the true channel. Without ordering, a sample's loss is not monotone in t, and single-scalar CRC would not apply. Taking ℓ̄₁ = max(l1, l2) makes it monotone, and costs very little because l2 ≤ l1 for most samples. Reported risks use the *actual* losses (`which='actual'`), which are bounded by the envelope.
3. **Exact CRC sweep for the cascade.** As t increases, a sample's envelope loss changes only at s_i (S3→S2) and at s_i + a (S2→S1). `calibrate_t` sorts these 2n events and accumulates loss increments, which gives the exact risk curve in O(n log n).
4. **Separate scorer split.** The scorer and the offset a are fitted on D_sc, never on D_cal, so the calibration scores stay exchangeable with the test scores.
5. **Stage outputs are improvement-only.** S1 keeps the probe step only if it raises the estimated rate, and S2 likewise. This makes stage rates non-decreasing on Ĝ.
6. **Regret is clipped to [0, 1].** Negative regret (beating WMMSE) counts as zero loss. This gives the bound B = 1 used in Eq. 22 and is conservative.
7. **Interference regimes for Corollary 1** are terciles of the estimated rate per link. The tercile edges are computed on the scorer split D_sc, so the grouping does not depend on calibration data, and are then applied to calibration and test samples.

## 4. Reproducibility checklist

- [x] All random seeds fixed: data (1–4, 50–54), models (0–4), bootstrap (100 + seed), evaluation (2026, 7, 11).
- [x] All hyperparameters listed in `README.md` § 6 and set as module constants in `scripts/`.
- [x] Every number in the paper is stored in `results/results.json` or `results/results_cascade.json`.
- [x] Figures are regenerated from the JSON files by `scripts/make_figures.py`.
- [x] Unit and end-to-end tests run in CI.
- [ ] Large generated data and trained weights (about 460 MB) are not stored in git. Regenerate them with `scripts/train.py`, or download them from the archive DOI once deposited.
