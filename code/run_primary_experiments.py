from __future__ import annotations
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from misinformation_simulation import (
    SimulationConfig, build_graph, graph_metrics, compute_rankings,
    simulate_diffusion, removal_structure_metrics
)

TOPOLOGIES=['ER','WS','BA']
STRATEGIES=['random','degree','betweenness','pagerank','eigenvector','kcore','collective_influence','community_bridge']
COVERAGES=[0.01,0.05,0.10]

def stable_seed(*parts):
    x=2166136261
    for p in parts:
        for b in str(p).encode('utf-8'):
            x ^= b
            x=(x*16777619)&0xffffffff
    return int(x)

def one_graph(topology, trial, n=1000):
    cfg=SimulationConfig(n=n,mean_degree=6,beta=0.15,gamma=0.05,initial_spreaders=5,max_steps=500,ws_rewire=0.10,ba_m=3)
    graph_seed=stable_seed('graph',topology,trial)
    g=build_graph(topology,cfg,graph_seed)
    gm={'topology':topology,'trial':trial,'graph_seed':graph_seed,**graph_metrics(g)}
    rankings,runtimes,membership=compute_rankings(g,seed=stable_seed('rank',topology,trial),betweenness_k=100)
    runtime_rows=[{'topology':topology,'trial':trial,'strategy':k,'ranking_seconds':v,'n':n,'edges':g.number_of_edges()} for k,v in runtimes.items()]
    # Fixed random ordering gives nested random protection sets across coverage levels.
    rng=np.random.default_rng(stable_seed('random-order',topology,trial))
    random_order=list(map(int,rng.permutation(np.array(list(g.nodes()),dtype=int))))
    baseline, curve, _=simulate_diffusion(
        g,cfg,seed=stable_seed('diffusion',topology,trial,'none'),
        protected=[],seed_mode='random',behavior_mode='homogeneous',return_curve=True
    )
    baseline_row={'topology':topology,'trial':trial,**baseline}
    curve_rows=[{'topology':topology,'trial':trial,'time':i,'active_spreaders':v} for i,v in enumerate(curve or [])]
    intervention_rows=[]
    structure_rows=[]
    selected_curve_rows=[]
    for strategy in STRATEGIES:
        order=random_order if strategy=='random' else rankings[strategy]
        for fraction in COVERAGES:
            count=int(round(n*fraction))
            protected=order[:count]
            metrics, icurve, _=simulate_diffusion(
                g,cfg,seed=stable_seed('diffusion',topology,trial,strategy,fraction),
                protected=protected,seed_mode='random',behavior_mode='homogeneous',
                membership=membership,return_curve=(fraction==0.05 and strategy in ['random','degree','betweenness','kcore','collective_influence','community_bridge'])
            )
            intervention_rows.append({'topology':topology,'trial':trial,'strategy':strategy,'fraction':fraction,**metrics})
            structure_rows.append({'topology':topology,'trial':trial,'strategy':strategy,'fraction':fraction,**removal_structure_metrics(g,protected,membership)})
            if icurve is not None:
                selected_curve_rows.extend({'topology':topology,'trial':trial,'strategy':strategy,'fraction':fraction,'time':i,'active_spreaders':v} for i,v in enumerate(icurve))
    return gm, runtime_rows, baseline_row, curve_rows, intervention_rows, structure_rows, selected_curve_rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trials',type=int,default=100)
    ap.add_argument('--workers',type=int,default=max(1,min(8,os.cpu_count() or 1)))
    ap.add_argument('--out',type=Path,default=HERE/'data')
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    jobs=[(t,i) for t in TOPOLOGIES for i in range(args.trials)]
    buckets=[[] for _ in range(7)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures={ex.submit(one_graph,t,i): (t,i) for t,i in jobs}
        done=0
        for fut in as_completed(futures):
            result=fut.result()
            for j,item in enumerate(result):
                if isinstance(item,list): buckets[j].extend(item)
                else: buckets[j].append(item)
            done+=1
            if done%10==0 or done==len(jobs): print(f'completed {done}/{len(jobs)}',flush=True)
    names=['graph_metrics.csv','ranking_runtimes.csv','baseline_results.csv','baseline_curves.csv','intervention_results.csv','structure_metrics.csv','intervention_curves_5pct.csv']
    for name,rows in zip(names,buckets):
        df=pd.DataFrame(rows)
        sort_cols=[c for c in ['topology','trial','strategy','fraction','time'] if c in df.columns]
        if sort_cols: df=df.sort_values(sort_cols).reset_index(drop=True)
        df.to_csv(args.out/name,index=False)
        print(name,len(df))
    metadata={'trials':args.trials,'topologies':TOPOLOGIES,'strategies':STRATEGIES,'coverages':COVERAGES,'config':SimulationConfig().__dict__,'workers':args.workers}
    (args.out/'primary_metadata.json').write_text(json.dumps(metadata,indent=2))
if __name__=='__main__':
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(end_time-start_time)