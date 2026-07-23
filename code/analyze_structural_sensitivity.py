"""Phase 1 (Major Issue #5): aggregate existing baseline structural-sensitivity
and seed-count-sensitivity records into Supplementary Table S15 / S17 inputs.

Reads data/structural_sensitivity.csv (540 baseline-diffusion records, no
strategy dimension) and data/robustness_baseline.csv (family=seed_count,
780-record family) and writes tidy per-setting summary CSVs to
analysis_outputs/. This does not run new simulations; it only reports what
already exists on disk (baseline diffusion only).
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / 'data'
OUT = ROOT / 'analysis_outputs'
OUT.mkdir(exist_ok=True)

OUTCOMES = ['t10', 't50', 'peak_infected', 'final_size', 'unique_exposed_fraction', 'reached_50']

FAMILY_ORDER = {'n': 0, 'mean_degree': 1, 'ws_rewire': 2, 'ba_m': 3, 'sbm_mixing': 4}
FAMILY_LABEL = {
    'n': 'Network size (N)',
    'mean_degree': 'Mean degree',
    'ws_rewire': 'WS rewiring probability',
    'ba_m': 'BA attachment parameter (m)',
    'sbm_mixing': 'SBM mixing parameter',
}


def summarize(df, group_cols):
    rows = []
    for key, g in df.groupby(group_cols):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row['n_realizations'] = len(g)
        for col in OUTCOMES:
            row[f'{col}_mean'] = g[col].mean()
            row[f'{col}_sd'] = g[col].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ss = pd.read_csv(DATA / 'structural_sensitivity.csv')
    s15 = summarize(ss, ['family', 'value', 'topology'])
    s15['family_order'] = s15['family'].map(FAMILY_ORDER)
    s15 = s15.sort_values(['family_order', 'value', 'topology']).drop(columns='family_order')
    s15['family_label'] = s15['family'].map(FAMILY_LABEL)
    s15.to_csv(OUT / 'S15_structural_sensitivity_baseline.csv', index=False)
    print('S15 rows (setting x topology cells):', len(s15))
    print(s15[['family', 'value', 'topology', 'n_realizations', 't10_mean', 't50_mean',
               'final_size_mean', 'reached_50_mean']].to_string(index=False))

    rb = pd.read_csv(DATA / 'robustness_baseline.csv')
    seedcount_existing = rb[rb.family == 'seed_count'].copy()
    print('\nExisting seed_count family beta values:', sorted(seedcount_existing.beta.unique()))
    s17_existing = summarize(seedcount_existing, ['initial_spreaders', 'topology', 'beta'])
    s17_existing.to_csv(OUT / 'S17_seed_count_existing_beta015.csv', index=False)
    print('\nS17 (existing, beta=0.15 only):')
    print(s17_existing[['initial_spreaders', 'topology', 'beta', 'n_realizations', 't10_mean',
                         't50_mean', 'final_size_mean', 'reached_50_mean']].to_string(index=False))


if __name__ == '__main__':
    main()
