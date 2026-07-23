from __future__ import annotations
import argparse, os, sys, json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from misinformation_simulation import *
from datetime import datetime

# Extends the robustness sensitivity families (structural: WS rewiring / SBM
# mixing / network size; and initial-spreader count) with the intervention
# strategy dimension that data/structural_sensitivity.csv and
# data/seed_placement_results.csv never carried (Major Issue #5). Uses the
# full 8-strategy primary set so results are directly comparable to
# Supplementary Tables S7/S8/S11, at fixed 5% coverage, crossed with both the
# saturated (beta=0.15) and near-threshold (beta=0.02) transmission regimes.
STRATEGIES=['random','degree','betweenness','pagerank','eigenvector','kcore','collective_influence','community_bridge']
BETAS=[.15,.02]
COVERAGE=.05

def stable_seed(*parts):
    x=2166136261
    for p in parts:
        for b in str(p).encode(): x=((x^b)*16777619)&0xffffffff
    return int(x)

def _rankings_and_random_order(g,seed,betweenness_k):
    rankings,runtimes,membership=compute_rankings(g,seed,betweenness_k=betweenness_k)
    rng=np.random.default_rng(stable_seed('random-order',seed))
    random_order=list(map(int,rng.permutation(np.array(list(g.nodes()),dtype=int))))
    return rankings,membership,random_order

def _strategy_rows(g,cfg,rankings,membership,random_order,tag,beta,family,value,topology,trial):
    n=g.number_of_nodes()
    count=int(round(COVERAGE*n))
    rows=[]
    for strategy in STRATEGIES:
        order=random_order if strategy=='random' else rankings[strategy]
        protected=order[:count]
        m,_,_=simulate_diffusion(g,cfg,stable_seed(tag,family,value,topology,trial,beta,strategy),
                                  protected=protected,membership=membership)
        rows.append({'family':family,'value':value,'topology':topology,'trial':trial,'beta':beta,
                     'gamma':cfg.gamma,'fraction':COVERAGE,'strategy':strategy,**m})
    return rows

def ws_job(p,trial):
    kwargs=dict(n=1000,mean_degree=6,beta=.15,gamma=.05,initial_spreaders=5,max_steps=500,ws_rewire=p,ba_m=3)
    cfg0=SimulationConfig(**kwargs)
    g=build_graph('WS',cfg0,stable_seed('2a-graph','ws_rewire',p,trial))
    rankings,membership,random_order=_rankings_and_random_order(g,stable_seed('2a-rank','ws_rewire',p,trial),betweenness_k=min(100,g.number_of_nodes()))
    rows=[]
    for beta in BETAS:
        cfg=SimulationConfig(**{**kwargs,'beta':beta})
        rows+=_strategy_rows(g,cfg,rankings,membership,random_order,'2a',beta,'ws_rewire',p,'WS',trial)
    return rows

def sbm_job(mu,trial):
    kwargs=dict(n=1000,mean_degree=6,beta=.15,gamma=.05,initial_spreaders=5,max_steps=500,ws_rewire=.10,ba_m=3,sbm_communities=4,sbm_mixing=mu)
    cfg0=SimulationConfig(**kwargs)
    g=build_graph('SBM',cfg0,stable_seed('2b-graph','sbm_mixing',mu,trial))
    rankings,membership,random_order=_rankings_and_random_order(g,stable_seed('2b-rank','sbm_mixing',mu,trial),betweenness_k=min(100,g.number_of_nodes()))
    rows=[]
    for beta in BETAS:
        cfg=SimulationConfig(**{**kwargs,'beta':beta})
        rows+=_strategy_rows(g,cfg,rankings,membership,random_order,'2b',beta,'sbm_mixing',mu,'SBM',trial)
    return rows

def seedcount_job(topology,trial):
    kwargs=dict(n=1000,mean_degree=6,beta=.15,gamma=.05,initial_spreaders=5,max_steps=500,ws_rewire=.10,ba_m=3)
    cfg0=SimulationConfig(**kwargs)
    g=build_graph(topology,cfg0,stable_seed('2c-graph',topology,trial))
    rankings,membership,random_order=_rankings_and_random_order(g,stable_seed('2c-rank',topology,trial),betweenness_k=min(100,g.number_of_nodes()))
    strategy_rows=[]
    baseline_rows=[]
    for seeds in [1,5,10]:
        for beta in BETAS:
            cfg=SimulationConfig(**{**kwargs,'beta':beta,'initial_spreaders':seeds})
            strategy_rows+=_strategy_rows(g,cfg,rankings,membership,random_order,'2c',beta,'seed_count',seeds,topology,trial)
            mb,_,_=simulate_diffusion(g,cfg,stable_seed('2c-base',topology,trial,seeds,beta),protected=[],membership=membership)
            baseline_rows.append({'family':'seed_count','value':seeds,'topology':topology,'trial':trial,
                                   'beta':beta,'gamma':cfg.gamma,'initial_spreaders':seeds,**mb})
    return strategy_rows,baseline_rows

def density_job(n_value,topology,trial):
    kwargs=dict(n=n_value,mean_degree=6,beta=.02,gamma=.05,initial_spreaders=5,max_steps=500,ws_rewire=.10,ba_m=3)
    cfg=SimulationConfig(**kwargs)
    g=build_graph(topology,cfg,stable_seed('2d-graph',topology,n_value,trial))
    rankings,membership,random_order=_rankings_and_random_order(g,stable_seed('2d-rank',topology,n_value,trial),betweenness_k=min(100,g.number_of_nodes()))
    return _strategy_rows(g,cfg,rankings,membership,random_order,'2d',.02,'n',n_value,topology,trial)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--trials',type=int,default=20); ap.add_argument('--workers',type=int,default=max(1,min(8,os.cpu_count() or 1))); ap.add_argument('--out',type=Path,default=HERE.parent/'data')
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)

    ws_jobs=[(p,i) for p in [.01,.10,.50] for i in range(args.trials)]
    sbm_jobs=[(mu,i) for mu in [.02,.05,.15,.30] for i in range(args.trials)]
    seed_jobs=[(t,i) for t in ['ER','WS','BA'] for i in range(args.trials)]
    density_jobs=[(n,t,i) for n in [500,2000] for t in ['ER','WS','BA'] for i in range(args.trials)]

    ws_rows=[]; sbm_rows=[]; seed_strategy_rows=[]; seed_baseline_rows=[]; density_rows=[]

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fs={ex.submit(ws_job,*j):('ws',j) for j in ws_jobs}
        fs.update({ex.submit(sbm_job,*j):('sbm',j) for j in sbm_jobs})
        fs.update({ex.submit(seedcount_job,*j):('seed',j) for j in seed_jobs})
        fs.update({ex.submit(density_job,*j):('density',j) for j in density_jobs})
        done=0; total=len(fs)
        for f in as_completed(fs):
            kind,job=fs[f]; res=f.result()
            if kind=='ws': ws_rows+=res
            elif kind=='sbm': sbm_rows+=res
            elif kind=='seed':
                sr,br=res; seed_strategy_rows+=sr; seed_baseline_rows+=br
            elif kind=='density': density_rows+=res
            done+=1
            if done%20==0 or done==total: print(f'{done}/{total} jobs done',flush=True)

    out=args.out
    pd.DataFrame(ws_rows).to_csv(out/'structural_strategy_ws_rewire.csv',index=False); print('ws rows',len(ws_rows))
    pd.DataFrame(sbm_rows).to_csv(out/'structural_strategy_sbm_mixing.csv',index=False); print('sbm rows',len(sbm_rows))
    pd.DataFrame(seed_strategy_rows).to_csv(out/'seed_count_strategy.csv',index=False); print('seed strategy rows',len(seed_strategy_rows))
    pd.DataFrame(seed_baseline_rows).to_csv(out/'seed_count_baseline_extended.csv',index=False); print('seed baseline rows',len(seed_baseline_rows))
    pd.DataFrame(density_rows).to_csv(out/'structural_strategy_density.csv',index=False); print('density rows',len(density_rows))
    (out/'structural_strategy_metadata.json').write_text(json.dumps({'trials':args.trials,'strategies':STRATEGIES,'betas':BETAS,'coverage':COVERAGE},indent=2))

if __name__=='__main__':
    start_time=datetime.now()
    main()
    print(datetime.now()-start_time)
