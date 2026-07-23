"""Supplementary Figure S7: best-strategy identity as a function of WS
rewiring and SBM mixing, at beta = 0.15 (saturated) and beta = 0.02
(near-threshold). Rank 1 = lowest mean final size (strongest suppression).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / 'data'
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)

plt.rcParams.update({'font.size': 9, 'figure.dpi': 120})

STRATEGIES = ['random', 'degree', 'betweenness', 'pagerank', 'eigenvector', 'kcore',
              'collective_influence', 'community_bridge']
LABELS = ['Random', 'Degree', 'Betweenness', 'PageRank', 'Eigenvector', 'k-core',
          'Collective\ninfluence', 'Community\nbridge']

ws = pd.read_csv(DATA / 'structural_strategy_ws_rewire.csv')
sbm = pd.read_csv(DATA / 'structural_strategy_sbm_mixing.csv')


def rank_matrix(df, values):
    mat = np.zeros((len(STRATEGIES), len(values)))
    for j, v in enumerate(values):
        means = df[df.value == v].groupby('strategy')['final_size'].mean()
        ranks = means.rank(method='min')
        for i, s in enumerate(STRATEGIES):
            mat[i, j] = ranks.get(s, np.nan)
    return mat


fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.5))
panels = [
    (ws, 'WS rewiring probability', [.01, .10, .50], .15, axes[0, 0]),
    (sbm, 'SBM mixing parameter', [.02, .05, .15, .30], .15, axes[0, 1]),
    (ws, 'WS rewiring probability', [.01, .10, .50], .02, axes[1, 0]),
    (sbm, 'SBM mixing parameter', [.02, .05, .15, .30], .02, axes[1, 1]),
]
im = None
for data, xlabel, values, beta, ax in panels:
    sub = data[np.isclose(data.beta, beta)]
    mat = rank_matrix(sub, values)
    im = ax.imshow(mat, cmap='RdYlGn_r', vmin=1, vmax=8, aspect='auto')
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([str(v) for v in values])
    ax.set_yticks(range(len(STRATEGIES)))
    ax.set_yticklabels(LABELS, fontsize=7.5)
    ax.set_xlabel(xlabel)
    regime = 'saturated, β=0.15' if np.isclose(beta, .15) else 'near-threshold, β=0.02'
    ax.set_title(f'{xlabel}\n({regime})', fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, int(mat[i, j]), ha='center', va='center', fontsize=7,
                        color='white' if mat[i, j] <= 2 or mat[i, j] >= 7 else 'black')

cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
cbar.set_label('Rank by mean final size (1 = strongest suppression, WS coverage 5%)')
fig.suptitle('Supplementary Figure S7. Strategy rank as a function of clustering (WS rewiring) '
             'and modularity (SBM mixing)', y=0.99, fontweight='bold', fontsize=10)
fig.savefig(FIG / 'figure_structural_strategy_ranks_s7.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('Wrote', FIG / 'figure_structural_strategy_ranks_s7.png')
