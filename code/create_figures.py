from pathlib import Path
import sys, math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
ROOT=Path('/mnt/data/revision_work'); DATA=ROOT/'data'; FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT))
from misinformation_simulation import *

plt.rcParams.update({'font.size':9,'figure.dpi':120})
base=pd.read_csv(DATA/'baseline_results.csv')
inter=pd.read_csv(DATA/'intervention_results.csv')
struct=pd.read_csv(DATA/'structure_metrics.csv')
rb=pd.read_csv(DATA/'robustness_baseline.csv')
ri=pd.read_csv(DATA/'robustness_intervention.csv')
he=pd.read_csv(DATA/'heterogeneity_results.csv')
sp=pd.read_csv(DATA/'seed_placement_results.csv')
sc=pd.read_csv(DATA/'scalability_runtimes.csv')
curves=pd.read_csv(DATA/'baseline_curves.csv')
icurves=pd.read_csv(DATA/'intervention_curves_5pct.csv')

# Baseline outcomes
spec=[('t10','Time to 10% ever infected','steps'),('t50','Time to 50% ever infected','steps'),('peak_infected','Peak active spreaders','nodes'),('final_size','Final outbreak size','fraction'),('active_spreader_auc','Active-spreader AUC','node-steps'),('unique_exposed_fraction','Unique directly exposed nodes','fraction')]
fig,axes=plt.subplots(2,3,figsize=(10,6.2)); order=['BA','ER','WS']
for ax,(m,title,ylabel) in zip(axes.ravel(),spec):
    means=base.groupby('topology')[m].mean().reindex(order); sds=base.groupby('topology')[m].std().reindex(order)
    ax.bar(order,means,yerr=sds,capsize=3); ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel('Topology')
fig.suptitle('Baseline diffusion outcomes across 100 graph realizations per topology',y=1.01,fontweight='bold'); fig.tight_layout(); fig.savefig(FIG/'figure_baseline_outcomes.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# Baseline curves
avg=curves.groupby(['topology','time'])['active_spreaders'].mean().reset_index()
fig,ax=plt.subplots(figsize=(7.5,4.5))
for t in order:
    g=avg[avg.topology==t]; ax.plot(g.time,g.active_spreaders,label=t)
ax.set_xlim(0,80); ax.set_xlabel('Simulation step'); ax.set_ylabel('Mean active spreaders'); ax.set_title('Baseline active-spreader curves'); ax.legend(title='Topology'); fig.tight_layout(); fig.savefig(FIG/'figure_baseline_curves.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# Intervention heatmaps at 5%
strategies=['random','degree','betweenness','pagerank','eigenvector','kcore','collective_influence','community_bridge']
labels=['Random','Degree','Betweenness','PageRank','Eigenvector','k-core','Collective influence','Community bridge']
five=inter[np.isclose(inter.fraction,.05)]
fig,axes=plt.subplots(1,3,figsize=(12,4.2))
for ax,(metric,title) in zip(axes,[('final_size','Final outbreak size'),('unique_exposed_fraction','Unique directly exposed'),('attempted_exposures','Exposure attempts')]):
    mat=five.groupby(['topology','strategy'])[metric].mean().unstack().reindex(index=order,columns=strategies)
    im=ax.imshow(mat.values,aspect='auto')
    ax.set_xticks(range(len(strategies)),labels,rotation=55,ha='right'); ax.set_yticks(range(3),order); ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v=mat.iloc[i,j]; txt=f'{v:.3f}' if metric!='attempted_exposures' else f'{v:.0f}'
            ax.text(j,i,txt,ha='center',va='center',fontsize=6)
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
fig.suptitle('Five-percent structural immunization: lower values indicate stronger suppression',y=1.03,fontweight='bold'); fig.tight_layout(); fig.savefig(FIG/'figure_intervention_5pct.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# Coverage response, selected scalable/top performers
sel=['random','degree','pagerank','kcore','collective_influence','community_bridge']
fig,axes=plt.subplots(2,3,figsize=(11,6.5),sharex=True)
for col,t in enumerate(order):
    for row,(metric,ylabel) in enumerate([('final_size','Final size'),('unique_exposed_fraction','Unique exposed fraction')]):
        ax=axes[row,col]
        for s in sel:
            g=inter[(inter.topology==t)&(inter.strategy==s)].groupby('fraction')[metric].mean(); ax.plot(g.index*100,g.values,marker='o',label=s)
        ax.set_title(t); ax.set_xlabel('Protected nodes (%)'); ax.set_ylabel(ylabel)
axes[0,2].legend(bbox_to_anchor=(1.03,1),loc='upper left',fontsize=7); fig.suptitle('Coverage-response patterns by topology and selection strategy',y=1.01,fontweight='bold'); fig.tight_layout(); fig.savefig(FIG/'figure_coverage_response.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# Near-threshold sensitivity
bg=rb[rb.family=='beta_gamma'].groupby(['beta','gamma','topology'])[['final_size','reached_50','t10']].mean().reset_index()
fig,axes=plt.subplots(2,3,figsize=(11,6.5))
for col,t in enumerate(order):
    ax=axes[0,col]
    for gam in [.05,.10]:
        g=bg[(bg.topology==t)&(bg.gamma==gam)]; ax.plot(g.beta,g.final_size,marker='o',label=f'gamma={gam}')
    ax.set_xscale('log'); ax.set_ylim(0,1.03); ax.set_title(f'{t}: baseline final size'); ax.set_xlabel('Transmission probability beta'); ax.set_ylabel('Final size'); ax.legend(fontsize=7)
    ax=axes[1,col]
    g=ri[(ri.topology==t)&np.isclose(ri.beta,.02)].groupby('strategy')['final_size'].mean().reindex(['random','degree','betweenness','kcore','collective_influence','community_bridge'])
    ax.bar(range(len(g)),g.values); ax.set_xticks(range(len(g)),['Random','Degree','Between.','k-core','CI','Bridge'],rotation=45,ha='right'); ax.set_title(f'{t}: 5% protection at beta=0.02'); ax.set_ylabel('Final size')
fig.suptitle('Near-threshold regimes reveal conditional topology and intervention effects',y=1.01,fontweight='bold'); fig.tight_layout(); fig.savefig(FIG/'figure_near_threshold.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# Mechanism analysis
m=inter.merge(struct,on=['topology','trial','strategy','fraction'])
fig,axes=plt.subplots(1,2,figsize=(10,4.2))
for t in order:
    g=m[m.topology==t]; axes[0].scatter(g.lcc_fraction_original_n,g.final_size,s=8,alpha=.25,label=t)
axes[0].set_xlabel('Largest remaining component / original N'); axes[0].set_ylabel('Final outbreak size'); axes[0].set_title('Fragmentation and outbreak size'); axes[0].legend()
q=m[np.isclose(m.fraction,.10)].groupby(['topology','strategy'])[['lcc_fraction_original_n','incident_edge_fraction']].mean().reset_index()
for i,t in enumerate(order):
    g=q[q.topology==t].set_index('strategy').reindex(strategies); axes[1].plot(range(len(strategies)),g.lcc_fraction_original_n,marker='o',label=t)
axes[1].set_xticks(range(len(strategies)),labels,rotation=55,ha='right'); axes[1].set_ylabel('Largest remaining component / original N'); axes[1].set_title('Structural disruption at 10% coverage'); axes[1].legend()
fig.tight_layout(); fig.savefig(FIG/'figure_mechanisms.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# Heterogeneity
fig,axes=plt.subplots(1,3,figsize=(11,4),sharey=True)
for ax,t in zip(axes,order):
    mat=he[he.topology==t].groupby(['behavior','strategy'])['final_size'].mean().unstack().reindex(index=['homogeneous','moderate','strong','degree_correlated'],columns=['random','degree','betweenness','kcore','collective_influence','community_bridge'])
    im=ax.imshow(mat.values,aspect='auto',vmin=0.75,vmax=1.0); ax.set_title(t); ax.set_xticks(range(mat.shape[1]),['Random','Degree','Between.','k-core','CI','Bridge'],rotation=50,ha='right'); ax.set_yticks(range(mat.shape[0]),['Homogeneous','Moderate','Strong','Degree-correlated'])
fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.02,pad=.02,label='Final outbreak size'); fig.suptitle('Strategy performance under heterogeneous sharing behavior',fontweight='bold'); fig.subplots_adjust(left=.12,right=.92,bottom=.25,top=.82,wspace=.18); fig.savefig(FIG/'figure_heterogeneity.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# Seed placement
fig,axes=plt.subplots(1,3,figsize=(10.5,3.8),sharey=True)
for ax,t in zip(axes,order):
    g=sp[sp.topology==t].groupby('seed_mode')['t10'].mean().reindex(['high_degree','bridge','random','peripheral']); ax.bar(['High-degree','Bridge','Random','Peripheral'],g.values); ax.set_title(t); ax.tick_params(axis='x',rotation=35); ax.set_ylabel('Time to 10% ever infected')
fig.suptitle('Initial-spreader placement changes early diffusion speed',fontweight='bold'); fig.tight_layout(); fig.savefig(FIG/'figure_seed_placement.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# Scalability
fig,ax=plt.subplots(figsize=(7.5,4.5))
for s,g in sc.groupby('strategy'):
    med=g.groupby('n')['ranking_seconds'].median(); ax.plot(med.index,med.values,marker='o',label=s)
ax.set_yscale('log'); ax.set_xlabel('Network size N'); ax.set_ylabel('Median ranking time (seconds, log scale)'); ax.set_title('Empirical ranking-time scalability'); ax.legend(fontsize=7,ncol=2); fig.tight_layout(); fig.savefig(FIG/'figure_scalability.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# Matched network visualizations
cfg=SimulationConfig(n=140,mean_degree=6,beta=.05,gamma=.05,initial_spreaders=3,max_steps=250)
g=build_graph('BA',cfg,2026); rankings,_,membership=compute_rankings(g,2026,betweenness_k=60); pos=nx.spring_layout(g,seed=7)
rng=np.random.default_rng(44); rand=list(map(int,rng.permutation(np.arange(cfg.n))))
vis=[('random',rand),('degree',rankings['degree']),('collective_influence',rankings['collective_influence']),('community_bridge',rankings['community_bridge'])]
fig,axes=plt.subplots(2,2,figsize=(9,8))
for ax,(strategy,order_nodes) in zip(axes.ravel(),vis):
    protected=order_nodes[:14]
    metrics,curve,details=simulate_diffusion(g,cfg,91,protected=protected,membership=membership,return_curve=True,return_node_details=True)
    inf=set(details['ever_infected_nodes']); prot=set(protected); init=set(details['initial_nodes'])
    colors=['black' if n in prot else ('red' if n in init else ('orange' if n in inf else 'lightgray')) for n in g.nodes()]
    nx.draw_networkx_edges(g,pos,ax=ax,width=.35,alpha=.25); nx.draw_networkx_nodes(g,pos,ax=ax,node_size=20,node_color=colors,linewidths=0)
    ax.set_title(f"{strategy.replace('_',' ').title()}\nfinal size={metrics['final_size']:.2f}, LCC after protection={removal_structure_metrics(g,protected,membership)['lcc_fraction_original_n']:.2f}"); ax.axis('off')
fig.suptitle('Matched BA network: protected nodes (black), seeds (red), reached nodes (orange)',fontweight='bold'); fig.tight_layout(); fig.savefig(FIG/'figure_network_mechanisms.png',dpi=300,bbox_inches='tight'); plt.close(fig)

print('created',len(list(FIG.glob('*.png'))),'figures')
for p in sorted(FIG.glob('*.png')): print(p.name,p.stat().st_size)
