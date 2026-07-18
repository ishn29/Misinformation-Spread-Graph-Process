from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import f_oneway
ROOT=Path('/mnt/data/revision_work'); DATA=ROOT/'data'; OUT=ROOT/'analysis_outputs'; OUT.mkdir(exist_ok=True)
rb=pd.read_csv(DATA/'robustness_baseline.csv')
ri=pd.read_csv(DATA/'robustness_intervention.csv')
he=pd.read_csv(DATA/'heterogeneity_results.csv')
sp=pd.read_csv(DATA/'seed_placement_results.csv')
ss=pd.read_csv(DATA/'structural_sensitivity.csv')
sc=pd.read_csv(DATA/'scalability_runtimes.csv')

rb.groupby(['family','beta','gamma','initial_spreaders','topology'],dropna=False)[['t10','t50','reached_50','final_size','active_spreader_auc','unique_exposed_fraction','attempted_exposures']].agg(['mean','std']).reset_index().to_csv(OUT/'robustness_baseline_summary.csv',index=False)
ri.groupby(['beta','topology','strategy'])[['t10','t50','reached_50','final_size','active_spreader_auc','unique_exposed_fraction','attempted_exposures']].agg(['mean','std']).reset_index().to_csv(OUT/'robustness_intervention_summary.csv',index=False)
he.groupby(['behavior','topology','strategy'])[['t10','t50','reached_50','final_size','active_spreader_auc','unique_exposed_fraction','attempted_exposures']].agg(['mean','std']).reset_index().to_csv(OUT/'heterogeneity_summary.csv',index=False)
sp.groupby(['seed_mode','topology'])[['t10','t50','reached_50','final_size','active_spreader_auc','unique_exposed_fraction']].agg(['mean','std']).reset_index().to_csv(OUT/'seed_placement_summary.csv',index=False)
ss.groupby(['family','value','topology'])[['mean_degree','clustering','lcc_fraction','t10','t50','reached_50','final_size','active_spreader_auc','unique_exposed_fraction']].agg(['mean','std']).reset_index().to_csv(OUT/'structural_sensitivity_summary.csv',index=False)
sc.groupby(['topology','n','strategy'])['ranking_seconds'].agg(['mean','std','median']).reset_index().to_csv(OUT/'scalability_summary.csv',index=False)

# winners
wins=[]
for (beta,topo),g in ri.groupby(['beta','topology']):
    means=g.groupby('strategy')[['final_size','active_spreader_auc','unique_exposed_fraction','attempted_exposures','reached_50']].mean()
    for metric in means.columns:
        s=means[metric].idxmin(); wins.append({'beta':beta,'topology':topo,'metric':metric,'best_strategy':s,'best_mean':means.loc[s,metric],'random_mean':means.loc['random',metric]})
pd.DataFrame(wins).to_csv(OUT/'robustness_intervention_winners.csv',index=False)

hwins=[]
for (behavior,topo),g in he.groupby(['behavior','topology']):
    means=g.groupby('strategy')[['final_size','active_spreader_auc','unique_exposed_fraction','attempted_exposures','reached_50']].mean()
    for metric in means.columns:
        s=means[metric].idxmin(); hwins.append({'behavior':behavior,'topology':topo,'metric':metric,'best_strategy':s,'best_mean':means.loc[s,metric],'random_mean':means.loc['random',metric]})
pd.DataFrame(hwins).to_csv(OUT/'heterogeneity_winners.csv',index=False)

print('NEAR THRESHOLD BASELINE')
print(rb[rb.family=='beta_gamma'].groupby(['beta','gamma','topology'])[['t10','reached_50','final_size','unique_exposed_fraction']].mean().round(3).to_string())
print('\nINTERVENTION FINAL SIZE')
print(ri.groupby(['beta','topology','strategy'])['final_size'].mean().round(3).to_string())
print('\nINTERVENTION REACHED50')
print(ri.groupby(['beta','topology','strategy'])['reached_50'].mean().round(2).to_string())
print('\nHETEROGENEITY WINNERS final_size')
print(pd.DataFrame(hwins).query("metric=='final_size'").to_string(index=False))
print('\nSEED PLACEMENT')
print(sp.groupby(['topology','seed_mode'])[['t10','t50','reached_50','final_size']].mean().round(3).to_string())
print('\nSCALABILITY MEDIAN')
print(sc.groupby(['n','strategy'])['ranking_seconds'].median().unstack().round(4).to_string())
