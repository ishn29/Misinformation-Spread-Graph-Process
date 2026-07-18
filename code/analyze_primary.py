from pathlib import Path
import json, math
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal, levene, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.anova import AnovaRM

ROOT=Path('')
DATA=ROOT/'data'; OUT=ROOT/'analysis_outputs'; OUT.mkdir(exist_ok=True)
base=pd.read_csv(DATA/'baseline_results.csv')
inter=pd.read_csv(DATA/'intervention_results.csv')
graph=pd.read_csv(DATA/'graph_metrics.csv')
struct=pd.read_csv(DATA/'structure_metrics.csv')
runtime=pd.read_csv(DATA/'ranking_runtimes.csv')

metrics=['t10','t50','peak_infected','final_size','active_spreader_auc','unique_exposed_fraction','attempted_exposures','repeat_exposures']
# baseline summaries
sumrows=[]
for topo,g in base.groupby('topology'):
    for metric in metrics:
        x=g[metric].dropna()
        sumrows.append({'topology':topo,'metric':metric,'n':len(x),'mean':x.mean(),'sd':x.std(ddof=1),'median':x.median(),'q025':x.quantile(.025),'q975':x.quantile(.975)})
pd.DataFrame(sumrows).to_csv(OUT/'baseline_summary.csv',index=False)

# baseline inferential
rows=[]
pvals=[]
for metric in metrics:
    groups=[base.loc[base.topology==t,metric].dropna().values for t in ['BA','ER','WS']]
    F,p=f_oneway(*groups)
    df1=2; df2=sum(len(x) for x in groups)-3
    eta=(F*df1)/(F*df1+df2)
    kw,pk=kruskal(*groups)
    welch=anova_oneway(groups,use_var='unequal')
    lev,pl=levene(*groups,center='median')
    rows.append({'metric':metric,'F':F,'df1':df1,'df2':df2,'p':p,'eta2':eta,'welch_F':welch.statistic,'welch_p':welch.pvalue,'kruskal_H':kw,'kruskal_p':pk,'levene_W':lev,'levene_p':pl})
    pvals.append(p)
adj=multipletests(pvals,method='holm')[1]
for r,a in zip(rows,adj): r['holm_p']=a
pd.DataFrame(rows).to_csv(OUT/'baseline_tests.csv',index=False)

# intervention summaries
imetrics=['t10','t50','reached_50','peak_infected','final_size','active_spreader_auc','unique_exposed_fraction','attempted_exposures','repeat_exposures']
summary=inter.groupby(['topology','strategy','fraction'])[imetrics].agg(['mean','std','median']).reset_index()
summary.to_csv(OUT/'intervention_summary.csv',index=False)

# 5% repeated-measures ANOVA and pairwise comparisons vs random
rmrows=[]; pairrows=[]
five=inter[np.isclose(inter.fraction,.05)].copy()
def rm_oneway(wide):
    arr=wide.to_numpy(dtype=float)
    n,k=arr.shape
    grand=arr.mean(); cond=arr.mean(axis=0); subj=arr.mean(axis=1)
    ss_total=((arr-grand)**2).sum()
    ss_cond=n*((cond-grand)**2).sum()
    ss_subj=k*((subj-grand)**2).sum()
    ss_error=ss_total-ss_cond-ss_subj
    df1=k-1; df2=(n-1)*(k-1)
    F=(ss_cond/df1)/(ss_error/df2) if ss_error>0 else np.inf
    from scipy.stats import f as fdist
    p=float(fdist.sf(F,df1,df2))
    pe=ss_cond/(ss_cond+ss_error) if (ss_cond+ss_error)>0 else np.nan
    return F,df1,df2,p,pe
for topo,gt in five.groupby('topology'):
    for metric in ['active_spreader_auc','final_size','unique_exposed_fraction','attempted_exposures','peak_infected']:
        wide=gt.pivot(index='trial',columns='strategy',values=metric).dropna()
        F,df1,df2,p,pe=rm_oneway(wide)
        rmrows.append({'topology':topo,'metric':metric,'F':F,'df1':df1,'df2':df2,'p':p,'partial_eta2':pe})
        for strat in [s for s in wide.columns if s!='random']:
            d=(wide[strat]-wide['random']).dropna()
            t,pv=ttest_rel(wide.loc[d.index,strat],wide.loc[d.index,'random'])
            dz=d.mean()/d.std(ddof=1) if d.std(ddof=1)>0 else np.nan
            try: w,pw=wilcoxon(d)
            except Exception: w,pw=np.nan,np.nan
            pairrows.append({'topology':topo,'metric':metric,'strategy':strat,'mean_difference':d.mean(),'percent_difference':100*d.mean()/wide.loc[d.index,'random'].mean(),'t':t,'p':pv,'cohen_dz':dz,'wilcoxon_W':w,'wilcoxon_p':pw})
pdf=pd.DataFrame(pairrows); pdf['holm_p']=np.nan; pdf['holm_wilcoxon_p']=np.nan
for keys,idx in pdf.groupby(['topology','metric']).groups.items():
    ids=list(idx)
    pdf.loc[ids,'holm_p']=multipletests(pdf.loc[ids,'p'],method='holm')[1]
    pdf.loc[ids,'holm_wilcoxon_p']=multipletests(pdf.loc[ids,'wilcoxon_p'],method='holm')[1]
pd.DataFrame(rmrows).to_csv(OUT/'intervention_rm_anova_5pct.csv',index=False)
pdf.to_csv(OUT/'intervention_pairwise_5pct.csv',index=False)

# strategy winners per topology/coverage
winner_rows=[]
means=inter.groupby(['topology','fraction','strategy'])[imetrics].mean().reset_index()
for (topo,f),g in means.groupby(['topology','fraction']):
    for metric in ['active_spreader_auc','final_size','unique_exposed_fraction','attempted_exposures','peak_infected','reached_50']:
        z=g.sort_values(metric,ascending=True).iloc[0]
        winner_rows.append({'topology':topo,'fraction':f,'metric':metric,'best_strategy':z.strategy,'best_mean':z[metric]})
pd.DataFrame(winner_rows).to_csv(OUT/'strategy_winners.csv',index=False)

# Mechanism: merge structure and outcomes; correlations and regression-friendly file
merged=inter.merge(struct,on=['topology','trial','strategy','fraction'],how='left')
merged.to_csv(OUT/'intervention_with_structure.csv',index=False)
corrows=[]
for topo,g in merged.groupby('topology'):
    for x in ['lcc_fraction_original_n','components_after','incident_edge_fraction','cross_community_edges_blocked','protected_mean_degree','approx_mean_path_after']:
        for y in ['final_size','active_spreader_auc','unique_exposed_fraction','attempted_exposures']:
            pair=g[[x,y]].dropna()
            corrows.append({'topology':topo,'x':x,'y':y,'pearson_r':pair[x].corr(pair[y]),'spearman_r':pair[x].corr(pair[y],method='spearman'),'n':len(pair)})
pd.DataFrame(corrows).to_csv(OUT/'mechanism_correlations.csv',index=False)

# Ranking runtime summaries
runtime.groupby(['strategy'])['ranking_seconds'].agg(['mean','std','median','max']).reset_index().to_csv(OUT/'ranking_runtime_summary.csv',index=False)

print('BASELINE MEANS')
print(base.groupby('topology')[metrics].mean().round(4))
print('\nBASELINE TESTS')
print(pd.DataFrame(rows)[['metric','F','p','eta2']].round(5))
print('\n5% MEANS selected')
print(five.groupby(['topology','strategy'])[['final_size','active_spreader_auc','unique_exposed_fraction','attempted_exposures','reached_50']].mean().round(4).to_string())
print('\nWINNERS')
print(pd.DataFrame(winner_rows).query("metric in ['final_size','unique_exposed_fraction','attempted_exposures']").to_string(index=False))
