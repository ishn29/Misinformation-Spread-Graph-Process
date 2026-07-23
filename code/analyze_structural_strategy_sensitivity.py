"""Phase 2 analysis (Major Issue #5): build Supplementary Table S16 and the
pre-registered stability determination from the new structural/seed-count x
strategy sweep (run_structural_strategy_sensitivity.py), plus the final
Supplementary Table S17 (seed-count sensitivity, baseline diffusion, now
crossed with a near-threshold beta).

Pre-registered stability criterion (see Methods): within a family x topology
x beta group, a strategy is a "stable winner" if it is one of the two lowest
mean-final-size strategies AND is significantly better than random (paired
t-test across matched graph realizations, Holm-adjusted p < .05 across the
seven non-random strategies) at every tested value of the varied parameter.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / 'data'
OUT = ROOT / 'analysis_outputs'
OUT.mkdir(exist_ok=True)

STRATEGIES = ['random', 'degree', 'betweenness', 'pagerank', 'eigenvector', 'kcore',
              'collective_influence', 'community_bridge']
NON_RANDOM = [s for s in STRATEGIES if s != 'random']

FAMILY_LABEL = {
    'ws_rewire': 'WS rewiring probability',
    'sbm_mixing': 'SBM mixing parameter',
    'seed_count': 'Initial spreader count',
    'n': 'Network size (N)',
}


def per_cell_test(sub: pd.DataFrame):
    """Paired targeted-vs-random test for one (family,value,topology,beta) cell."""
    pivot = sub.pivot(index='trial', columns='strategy', values='final_size')
    pvals, diffs = {}, {}
    for s in NON_RANDOM:
        if s not in pivot.columns or 'random' not in pivot.columns:
            continue
        t, p = ttest_rel(pivot[s], pivot['random'])
        pvals[s] = p
        diffs[s] = float(pivot[s].mean() - pivot['random'].mean())
    if not pvals:
        return {}, {}, {}
    strategies = list(pvals.keys())
    rej, p_adj, _, _ = multipletests(list(pvals.values()), alpha=.05, method='holm')
    sig = {strategies[i]: bool(rej[i] and diffs[strategies[i]] < 0) for i in range(len(strategies))}
    padj = {strategies[i]: float(p_adj[i]) for i in range(len(strategies))}
    return sig, padj, diffs


def analyze_family(df: pd.DataFrame, family: str, group_keys):
    """Returns (rows for S16, stability summary dict keyed by group)."""
    rows = []
    stability_summary = []
    for gkey, gdf in df.groupby(group_keys):
        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
        values = sorted(gdf['value'].unique())
        top2_by_value, sig_by_value, means_by_value, diffs_by_value = {}, {}, {}, {}
        for value in values:
            sub = gdf[gdf['value'] == value]
            means = sub.groupby('strategy')['final_size'].mean()
            means_by_value[value] = means
            non_random_means = means.drop('random', errors='ignore').sort_values()
            top2_by_value[value] = set(non_random_means.index[:2])
            sig, padj, diffs = per_cell_test(sub)
            sig_by_value[value] = {s for s, v in sig.items() if v}
            diffs_by_value[value] = diffs
            rank1_strategy, rank2_strategy = non_random_means.index[0], non_random_means.index[1]
            rank1_mean, rank2_mean = float(non_random_means.iloc[0]), float(non_random_means.iloc[1])
            random_mean = float(means.get('random', np.nan))
            rel_change = (rank1_mean - random_mean) / random_mean * 100 if random_mean else np.nan
            rows.append({
                'family': family, 'family_label': FAMILY_LABEL[family], 'value': value,
                **dict(zip(group_keys, gkey)), 'n_realizations': sub['trial'].nunique(),
                'rank1_strategy': rank1_strategy, 'rank1_mean_final_size': round(rank1_mean, 4),
                'rank1_holm_p': round(padj.get(rank1_strategy, float('nan')), 4),
                'rank1_significant_vs_random': bool(sig_by_value[value] & {rank1_strategy}),
                'rank2_strategy': rank2_strategy, 'rank2_mean_final_size': round(rank2_mean, 4),
                'rank2_holm_p': round(padj.get(rank2_strategy, float('nan')), 4),
                'rank2_significant_vs_random': bool(sig_by_value[value] & {rank2_strategy}),
                'random_mean_final_size': round(random_mean, 4),
                'relative_change_pct': round(rel_change, 2),
                # kept for backward compatibility with earlier drafts
                'best_strategy': rank1_strategy, 'best_mean_final_size': round(rank1_mean, 4),
                'best_significant_vs_random_holm': bool(sig_by_value[value] & {rank1_strategy}),
            })
        if len(values) >= 2:
            candidates = set.intersection(*[top2_by_value[v] & sig_by_value[v] for v in values])
            top2_sets = [frozenset(top2_by_value[v]) for v in values]
            top2_identical = len(set(top2_sets)) == 1
        else:
            candidates, top2_identical = set(), False
        stability_summary.append({
            'family': family, **dict(zip(group_keys, gkey)),
            'values_tested': values, 'top2_identical_across_values': top2_identical,
            'stable_winner_strategies': sorted(candidates),
            'stable': len(candidates) > 0,
        })
    for row in rows:
        match = next(s for s in stability_summary if s['family'] == row['family']
                     and all(s.get(k) == row.get(k) for k in group_keys))
        row['family_stable'] = match['stable']
        row['top2_identical_across_family_range'] = match['top2_identical_across_values']
        row['stable_winner_strategies'] = ','.join(match['stable_winner_strategies']) or '-'
    return pd.DataFrame(rows), pd.DataFrame(stability_summary)


def main():
    ws = pd.read_csv(DATA / 'structural_strategy_ws_rewire.csv')
    sbm = pd.read_csv(DATA / 'structural_strategy_sbm_mixing.csv')
    seed = pd.read_csv(DATA / 'seed_count_strategy.csv')
    density = pd.read_csv(DATA / 'structural_strategy_density.csv')

    ws_rows, ws_stab = analyze_family(ws, 'ws_rewire', ['topology', 'beta'])
    sbm_rows, sbm_stab = analyze_family(sbm, 'sbm_mixing', ['topology', 'beta'])
    seed_rows, seed_stab = analyze_family(seed, 'seed_count', ['topology', 'beta'])
    density_rows, density_stab = analyze_family(density, 'n', ['topology', 'beta'])

    s16 = pd.concat([ws_rows, sbm_rows, seed_rows, density_rows], ignore_index=True)
    s16.to_csv(OUT / 'S16_structural_seed_strategy_stability.csv', index=False)
    stab_all = pd.concat([ws_stab, sbm_stab, seed_stab, density_stab], ignore_index=True)
    stab_all.to_csv(OUT / 'S16_stability_family_summary.csv', index=False)

    print('=== S16 rows ===')
    print(s16.to_string(index=False))
    print('\n=== Family-level stability summary ===')
    print(stab_all.to_string(index=False))

    # S17: seed-count baseline sensitivity (no intervention), both beta regimes
    seed_base = pd.read_csv(DATA / 'seed_count_baseline_extended.csv')
    outcomes = ['t10', 't50', 'peak_infected', 'final_size', 'unique_exposed_fraction', 'reached_50']
    s17_rows = []
    for (seeds, topo, beta), g in seed_base.groupby(['initial_spreaders', 'topology', 'beta']):
        row = {'initial_spreaders': seeds, 'topology': topo, 'beta': beta, 'n_realizations': len(g)}
        for col in outcomes:
            row[f'{col}_mean'] = round(g[col].mean(), 4)
            row[f'{col}_sd'] = round(g[col].std(ddof=1), 4)
        s17_rows.append(row)
    s17 = pd.DataFrame(s17_rows).sort_values(['beta', 'topology', 'initial_spreaders'])
    s17.to_csv(OUT / 'S17_seed_count_sensitivity_full.csv', index=False)
    print('\n=== S17 (seed-count sensitivity, both beta regimes) ===')
    print(s17.to_string(index=False))


if __name__ == '__main__':
    main()
