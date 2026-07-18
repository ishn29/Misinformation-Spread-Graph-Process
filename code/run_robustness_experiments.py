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

TOPOLOGIES=['ER','WS','BA']
ROBUST_STRATEGIES=['random','degree','betweenness','kcore','collective_influence','community_bridge']

def stable_seed(*parts):
    x=2166136261
    for p in parts:
        for b in str(p).encode(): x=((x^b)*16777619)&0xffffffff
    return int(x)

def core_graph_job(topology,trial,n=1000):
    base=SimulationConfig(n=n,mean_degree=6,beta=.15,gamma=.05,initial_spreaders=5,max_steps=500,ws_rewire=.10,ba_m=3)
    g=build_graph(topology,base,stable_seed('robgraph',topology,trial,n))
    rankings,runtimes,membership=compute_rankings(g,stable_seed('robrank',topology,trial,n),betweenness_k=min(75,n))
    rng=np.random.default_rng(stable_seed('rob-random-order',topology,trial,n))
    random_order=list(map(int,rng.permutation(np.array(list(g.nodes()),dtype=int))))
    out_baseline=[]; out_intervention=[]; out_hetero=[]; out_seed=[]
    # Diffusion parameter sweep including near-threshold settings.
    for beta in [.005,.01,.02,.05,.15]:
        for gamma in [.05,.10]:
            cfg=SimulationConfig(**{**base.__dict__,'beta':beta,'gamma':gamma})
            m,_,_=simulate_diffusion(g,cfg,stable_seed('basepar',topology,trial,beta,gamma))
            out_baseline.append({'topology':topology,'trial':trial,'family':'beta_gamma','beta':beta,'gamma':gamma,'initial_spreaders':5,**m})
    # Initial spreader count sweep.
    for seeds in [1,5,10]:
        cfg=SimulationConfig(**{**base.__dict__,'initial_spreaders':seeds})
        m,_,_=simulate_diffusion(g,cfg,stable_seed('seedcount',topology,trial,seeds))
        out_baseline.append({'topology':topology,'trial':trial,'family':'seed_count','beta':cfg.beta,'gamma':cfg.gamma,'initial_spreaders':seeds,**m})
    # Seed placement sensitivity.
    for seed_mode in ['random','high_degree','bridge','peripheral']:
        m,_,_=simulate_diffusion(g,base,stable_seed('seedplace',topology,trial,seed_mode),seed_mode=seed_mode,membership=membership)
        out_seed.append({'topology':topology,'trial':trial,'seed_mode':seed_mode,**m})
    # Intervention rankings across selected transmission regimes, fixed 5% coverage.
    count=int(round(.05*n))
    for beta in [.01,.02,.05,.15]:
        cfg=SimulationConfig(**{**base.__dict__,'beta':beta})
        for strategy in ROBUST_STRATEGIES:
            protected=(random_order if strategy=='random' else rankings[strategy])[:count]
            m,_,_=simulate_diffusion(g,cfg,stable_seed('robint',topology,trial,beta,strategy),protected=protected,membership=membership)
            out_intervention.append({'topology':topology,'trial':trial,'beta':beta,'gamma':cfg.gamma,'fraction':.05,'strategy':strategy,**m})
    # Behavioral heterogeneity at 5% coverage.
    for behavior in ['homogeneous','moderate','strong','degree_correlated']:
        for strategy in ROBUST_STRATEGIES:
            protected=(random_order if strategy=='random' else rankings[strategy])[:count]
            m,_,_=simulate_diffusion(g,base,stable_seed('hetero',topology,trial,behavior,strategy),protected=protected,behavior_mode=behavior,membership=membership)
            out_hetero.append({'topology':topology,'trial':trial,'behavior':behavior,'fraction':.05,'strategy':strategy,**m})
    return out_baseline,out_intervention,out_hetero,out_seed

def structural_job(kind,value,topology,trial):
    kwargs=dict(n=1000,mean_degree=6,beta=.15,gamma=.05,initial_spreaders=5,max_steps=500,ws_rewire=.10,ba_m=3,sbm_communities=4,sbm_mixing=.10)
    if kind=='n': kwargs['n']=int(value)
    elif kind=='mean_degree':
        kwargs['mean_degree']=int(value); kwargs['ba_m']=max(1,int(value)//2)
    elif kind=='ws_rewire': kwargs['ws_rewire']=float(value)
    elif kind=='ba_m': kwargs['ba_m']=int(value); kwargs['mean_degree']=2*int(value)
    elif kind=='sbm_mixing': kwargs['sbm_mixing']=float(value)
    cfg=SimulationConfig(**kwargs)
    g=build_graph(topology,cfg,stable_seed('struct',kind,value,topology,trial))
    m,_,_=simulate_diffusion(g,cfg,stable_seed('structdiff',kind,value,topology,trial))
    return {'family':kind,'value':value,'topology':topology,'trial':trial,**graph_metrics(g),**m}

def scalability_job(topology,n,repeat):
    cfg=SimulationConfig(n=n,mean_degree=6,beta=.15,gamma=.05,initial_spreaders=5,max_steps=500)
    seed=stable_seed('scale',topology,n,repeat)
    import time
    t0=time.perf_counter(); g=build_graph(topology,cfg,seed); graph_seconds=time.perf_counter()-t0
    rankings,runtimes,_=compute_rankings(g,seed,betweenness_k=min(100,n))
    rows=[]
    for strategy,secs in runtimes.items():
        rows.append({'topology':topology,'n':n,'edges':g.number_of_edges(),'repeat':repeat,'strategy':strategy,'graph_seconds':graph_seconds,'ranking_seconds':secs})
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--trials',type=int,default=20); ap.add_argument('--workers',type=int,default=max(1,min(8,os.cpu_count() or 1))); ap.add_argument('--out',type=Path,default=HERE/'data')
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    jobs=[(t,i) for t in TOPOLOGIES for i in range(args.trials)]
    buckets=[[],[],[],[]]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fs={ex.submit(core_graph_job,*j):j for j in jobs}
        for k,f in enumerate(as_completed(fs),1):
            res=f.result()
            for i,x in enumerate(res): buckets[i].extend(x)
            if k%10==0 or k==len(fs): print('core',k,'/',len(fs),flush=True)
    for name,rows in zip(['robustness_baseline.csv','robustness_intervention.csv','heterogeneity_results.csv','seed_placement_results.csv'],buckets):
        pd.DataFrame(rows).to_csv(args.out/name,index=False); print(name,len(rows))
    struct_jobs=[]
    for v in [500,1000,2000]:
        for t in TOPOLOGIES:
            for i in range(args.trials): struct_jobs.append(('n',v,t,i))
    for v in [4,6,10]:
        for t in TOPOLOGIES:
            for i in range(args.trials): struct_jobs.append(('mean_degree',v,t,i))
    for v in [.01,.10,.50]:
        for i in range(args.trials): struct_jobs.append(('ws_rewire',v,'WS',i))
    for v in [2,3,5]:
        for i in range(args.trials): struct_jobs.append(('ba_m',v,'BA',i))
    for v in [.02,.10,.30]:
        for i in range(args.trials): struct_jobs.append(('sbm_mixing',v,'SBM',i))
    srows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fs={ex.submit(structural_job,*j):j for j in struct_jobs}
        for k,f in enumerate(as_completed(fs),1):
            srows.append(f.result())
            if k%50==0 or k==len(fs): print('struct',k,'/',len(fs),flush=True)
    pd.DataFrame(srows).to_csv(args.out/'structural_sensitivity.csv',index=False); print('structural_sensitivity.csv',len(srows))
    scale_jobs=[(t,n,r) for t in ['BA','WS'] for n in [500,1000,2000] for r in range(2)]
    scrows=[]
    with ProcessPoolExecutor(max_workers=min(args.workers,4)) as ex:
        fs={ex.submit(scalability_job,*j):j for j in scale_jobs}
        for k,f in enumerate(as_completed(fs),1):
            scrows.extend(f.result()); print('scale',k,'/',len(fs),flush=True)
    pd.DataFrame(scrows).to_csv(args.out/'scalability_runtimes.csv',index=False); print('scalability_runtimes.csv',len(scrows))
    (args.out/'robustness_metadata.json').write_text(json.dumps({'trials':args.trials,'strategies':ROBUST_STRATEGIES},indent=2))
if __name__=='__main__': 
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(end_time-start_time)
