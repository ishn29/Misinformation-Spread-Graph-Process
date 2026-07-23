"""Phase 4 (Major Issue #5 / A1c): replace the borrowed beta/gamma
justification with a threshold-relative one.

For each topology at the primary structural settings (N=1000, mean degree 6,
WS rewiring 0.10, BA attachment m=3), computes the largest adjacency
eigenvalue Lambda_max over 20 matched realizations (same construction as the
primary/robustness graphs) and the quenched mean-field epidemic threshold
lambda_c = 1 / Lambda_max (Chakrabarti et al. / Wang et al. quenched
mean-field approximation for SIS/SIR-like spread on networks; also the
standard reference threshold in Pastor-Satorras et al., Rev. Mod. Phys. 2015,
already cited as ref. 34). Reports the effective ratio (beta/gamma)/lambda_c
for beta = 0.15 (saturated regime) and beta = 0.02 (near-threshold regime),
gamma = 0.05.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import networkx as nx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from misinformation_simulation import SimulationConfig, build_graph

TOPOLOGIES = ['ER', 'WS', 'BA']
TRIALS = 20
GAMMA = 0.05
BETAS = [0.15, 0.02]


def stable_seed(*parts):
    x = 2166136261
    for p in parts:
        for b in str(p).encode():
            x = ((x ^ b) * 16777619) & 0xffffffff
    return int(x)


def largest_eigenvalue(g: nx.Graph) -> float:
    A = nx.to_scipy_sparse_array(g, dtype=float)
    from scipy.sparse.linalg import eigsh
    if A.shape[0] < 3:
        return float(np.linalg.eigvalsh(A.toarray()).max())
    val = eigsh(A, k=1, which='LA', return_eigenvectors=False, maxiter=5000)
    return float(val[0])


def main():
    kwargs = dict(n=1000, mean_degree=6, beta=.15, gamma=GAMMA, initial_spreaders=5,
                  max_steps=500, ws_rewire=.10, ba_m=3)
    cfg = SimulationConfig(**kwargs)
    rows = []
    for topo in TOPOLOGIES:
        for trial in range(TRIALS):
            g = build_graph(topo, cfg, stable_seed('threshold-graph', topo, trial))
            lam_max = largest_eigenvalue(g)
            lambda_c = 1.0 / lam_max
            row = {'topology': topo, 'trial': trial, 'lambda_max': lam_max, 'lambda_c': lambda_c}
            for beta in BETAS:
                row[f'ratio_beta{beta}'] = (beta / GAMMA) / lambda_c
            rows.append(row)
    df = pd.DataFrame(rows)
    summary = df.groupby('topology')[['lambda_max', 'lambda_c'] + [f'ratio_beta{b}' for b in BETAS]].agg(['mean', 'std'])
    print(summary.to_string())
    out_dir = HERE.parent / 'analysis_outputs'
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / 'epidemic_threshold_raw.csv', index=False)
    flat = summary.copy()
    flat.columns = ['_'.join(c) for c in flat.columns]
    flat.reset_index().to_csv(out_dir / 'epidemic_threshold_summary.csv', index=False)
    print('\nWrote', out_dir / 'epidemic_threshold_raw.csv', 'and epidemic_threshold_summary.csv')


if __name__ == '__main__':
    main()
