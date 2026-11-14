"""Stage 3: paper figures from results/results.json (python -m scripts.make_figures)."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.environ.get('CSPA_OUT', 'results')
R = json.load(open(f'{OUT}/results.json'))
os.makedirs(f'{OUT}/figs', exist_ok=True)
plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'axes.spines.top': False,
                     'axes.spines.right': False})

# Fig. 3: risk-deferral curves
g = [100 * x for x in R['curves']['defer_grid']]
fig, ax = plt.subplots(figsize=(3.5, 2.6))
style = {'CSPA learned scorer (full)': ('C0', '-', 2.0, 'CSPA (proposed)'),
         'Physics features only': ('0.45', '-', 1.0, 'Physics only'),
         'SI-DNN spread heuristic [Lee 2024]': ('0.55', '--', 1.0, 'SI-DNN heuristic [19]'),
         'Ensemble features only': ('0.65', '-.', 1.0, 'Ensemble only'),
         'Random': ('0.75', ':', 1.0, 'Random'),
         'Oracle': ('k', ':', 1.0, 'Oracle')}
for k, (c, ls, lw, lab) in style.items():
    ax.plot(g, [100 * v for v in R['curves'][k]], color=c, ls=ls, lw=lw, label=lab)
ax.axhline(0.3, color='0.3', lw=0.8, ls='--')
ax.text(78, 0.31, r'$\alpha$ = 0.3%', ha='right', va='bottom', fontsize=8)
ax.set_xlabel('Cycles deferred to WMMSE [%]')
ax.set_ylabel('Expected regret [%]')
ax.set_xlim(0, 80); ax.set_ylim(0, 0.56)
ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(f'{OUT}/figs/fig3_risk_deferral.png', dpi=300); plt.close(fig)

# Fig. 4: OOD offline vs online
lab = {'csi_err_0.3': 'CSI error 0.3', 'sparser_K10': 'K = 10', 'heavy_shadow': 'Shadowing 6/10 dB',
       'small_hall': '20 m hall', 'denser_K30': 'K = 30'}
fig, ax = plt.subplots(figsize=(3.5, 2.4))
for i, (k, n) in enumerate(lab.items()):
    o = R['ood'][k]
    off, on = 1e3 * o['offline']['risk'], 1e3 * o['online']['risk']
    ax.annotate('', xy=(on, i), xytext=(off, i), arrowprops=dict(arrowstyle='->', color='C0', lw=1))
    ax.plot(off, i, 'o', color='0.6', ms=6); ax.plot(on, i, 'o', color='C0', ms=6)
ax.axvline(3.0, color='0.3', ls='--', lw=0.8)
ax.set_yticks(range(len(lab))); ax.set_yticklabels(list(lab.values())); ax.invert_yaxis()
ax.set_xlabel(r'Expected regret [$\times 10^{-3}$]'); ax.set_xlim(0, 5.5)
fig.tight_layout(); fig.savefig(f'{OUT}/figs/fig4_ood_online.png', dpi=300); plt.close(fig)

# Fig. 5: validity of calibration
al = sorted(R['validity']['CRC'], key=float)
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for m, c, mk in [('CRC', 'C0', 'o'), ('LTT-HP', 'C2', 's'), ('Naive', '0.5', '^')]:
    ax.errorbar([1e3 * float(a) for a in al], [1e3 * R['validity'][m][a]['mean'] for a in al],
                yerr=[1e3 * R['validity'][m][a]['std'] for a in al], fmt=mk, color=c, ms=4,
                capsize=2, label=m)
ax.plot([0.5, 5.5], [0.5, 5.5], 'k--', lw=0.8, label=r'$y=\alpha$')
ax.set_xlabel(r'Budget $\alpha$ [$\times 10^{-3}$]'); ax.set_ylabel(r'Realized regret [$\times 10^{-3}$]')
ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(f'{OUT}/figs/figS1_validity.png', dpi=300); plt.close(fig)
print('figures written to', f'{OUT}/figs')

# Fig. 6: certified cascade vs single fallback (WMMSE iterations per cycle)
if os.path.exists(f'{OUT}/results_cascade.json'):
    C = json.load(open(f'{OUT}/results_cascade.json'))
    al = sorted(C['single'], key=float)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot([100 * float(a) for a in al], [C['single'][a]['iters'] for a in al], 'o-', color='0.6', label='Single fallback (CSPA)')
    ax.plot([100 * float(a) for a in al], [C['cascade'][a]['crc']['iters'] for a in al], 's-', color='C0', label='Certified cascade')
    ax.axhline(C['warm_iters'], color='0.3', ls='--', lw=0.8, label='Always warm start (no certificate)')
    ax.set_xlabel(r'Regret budget $\alpha$ [%]'); ax.set_ylabel('WMMSE iterations per cycle')
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(f'{OUT}/figs/fig5_cascade_cost.png', dpi=300); plt.close(fig)
    print('cascade figure written')
