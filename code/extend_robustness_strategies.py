import time
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from misinformation_simulation import (
    SimulationConfig, build_graph, compute_rankings, simulate_diffusion)

DATA = HERE / "data"
SCRATCH = Path("/sessions/clever-festive-curie/mnt/outputs")
PART_I = SCRATCH / "_b4_partial_intervention.csv"
PART_H = SCRATCH / "_b4_partial_heterogeneity.csv"

TOPOLOGIES = ["ER", "WS", "BA"]
NEW_STRATEGIES = ["pagerank", "eigenvector"]
CHECK_STRATEGY = "degree"
TRIALS = 20
N = 1000
BETAS = [.01, .02, .05, .15]
BEHAVIORS = ["homogeneous", "moderate", "strong", "degree_correlated"]
BUDGET_S = 38.0


def stable_seed(*parts):
    x = 2166136261
    for p in parts:
        for b in str(p).encode():
            x = ((x ^ b) * 16777619) & 0xffffffff
    return int(x)


def graph_job(topology, trial, strategies):
    """Near-threshold intervention + heterogeneity arms for one (topology, trial),
    identical to run_robustness_experiments.core_graph_job. Rankings computed once."""
    inter, hetero = [], []
    base = SimulationConfig(n=N, mean_degree=6, beta=.15, gamma=.05,
                            initial_spreaders=5, max_steps=500, ws_rewire=.10, ba_m=3)
    count = int(round(.05 * N))
    g = build_graph(topology, base, stable_seed("robgraph", topology, trial, N))
    rankings, _, membership = compute_rankings(
        g, stable_seed("robrank", topology, trial, N), betweenness_k=min(75, N))
    for beta in BETAS:
        cfg = SimulationConfig(**{**base.__dict__, "beta": beta})
        for strategy in strategies:
            protected = rankings[strategy][:count]
            m, _, _ = simulate_diffusion(
                g, cfg, stable_seed("robint", topology, trial, beta, strategy),
                protected=protected, membership=membership)
            inter.append({"topology": topology, "trial": trial, "beta": beta,
                          "gamma": cfg.gamma, "fraction": .05, "strategy": strategy, **m})
    for behavior in BEHAVIORS:
        for strategy in strategies:
            protected = rankings[strategy][:count]
            m, _, _ = simulate_diffusion(
                g, base, stable_seed("hetero", topology, trial, behavior, strategy),
                protected=protected, behavior_mode=behavior, membership=membership)
            hetero.append({"topology": topology, "trial": trial, "behavior": behavior,
                           "fraction": .05, "strategy": strategy, **m})
    return pd.DataFrame(inter), pd.DataFrame(hetero)


def done_jobs():
    if not PART_I.exists():
        return set()
    d = pd.read_csv(PART_I)
    return set(map(tuple, d[["topology", "trial"]].drop_duplicates().values))


def append_partial(df_i, df_h):
    df_i.to_csv(PART_I, mode="a", header=not PART_I.exists(), index=False)
    df_h.to_csv(PART_H, mode="a", header=not PART_H.exists(), index=False)


def finalize():
    part_i = pd.read_csv(PART_I)
    part_h = pd.read_csv(PART_H)

    # ---- determinism guard on a sample: recomputed 'degree' must match published ----
    inter_old = pd.read_csv(DATA / "robustness_intervention.csv")
    hetero_old = pd.read_csv(DATA / "heterogeneity_results.csv")
    sample = [("ER", 0), ("WS", 3), ("BA", 7)]
    num = ["final_size", "t50", "reached_50", "active_spreader_auc",
           "unique_exposed_fraction", "attempted_exposures"]
    for topo, tr in sample:
        gi, gh = graph_job(topo, tr, [CHECK_STRATEGY])
        for label, gen, pub, key in [
            ("intervention", gi, inter_old, ["topology", "trial", "beta", "strategy"]),
            ("heterogeneity", gh, hetero_old, ["topology", "trial", "behavior", "strategy"])]:
            ref = pub[(pub.topology == topo) & (pub.trial == tr) &
                      (pub.strategy == CHECK_STRATEGY)].merge(gen, on=key, suffixes=("_pub", "_new"))
            assert len(ref) > 0, f"{label} {topo}/{tr}: no overlap"
            for c in num:
                dmax = np.abs(ref[f"{c}_pub"] - ref[f"{c}_new"]).max()
                assert dmax < 1e-9, f"{label} {topo}/{tr}: '{c}' mismatch {dmax}"
    print("Determinism guard PASSED on sample", sample)

    new_i = part_i[part_i.strategy.isin(NEW_STRATEGIES)][inter_old.columns]
    new_h = part_h[part_h.strategy.isin(NEW_STRATEGIES)][hetero_old.columns]
    exp_i = len(TOPOLOGIES) * TRIALS * len(BETAS) * len(NEW_STRATEGIES)
    exp_h = len(TOPOLOGIES) * TRIALS * len(BEHAVIORS) * len(NEW_STRATEGIES)
    assert len(new_i) == exp_i, f"intervention rows {len(new_i)} != {exp_i}"
    assert len(new_h) == exp_h, f"heterogeneity rows {len(new_h)} != {exp_h}"

    pd.concat([inter_old, new_i], ignore_index=True).to_csv(
        DATA / "robustness_intervention.csv", index=False)
    pd.concat([hetero_old, new_h], ignore_index=True).to_csv(
        DATA / "heterogeneity_results.csv", index=False)
    print(f"Appended {len(new_i)} intervention + {len(new_h)} heterogeneity rows.")
    print("ALL DONE. Delete scratch partials:", PART_I.name, PART_H.name)


def main():
    all_jobs = [(t, i) for t in TOPOLOGIES for i in range(TRIALS)]
    done = done_jobs()
    remaining = [j for j in all_jobs if j not in done]
    if not remaining:
        finalize()
        return
    t0 = time.time()
    processed = 0
    for topo, tr in remaining:
        if time.time() - t0 > BUDGET_S and processed > 0:
            break
        di, dh = graph_job(topo, tr, [CHECK_STRATEGY] + NEW_STRATEGIES)
        append_partial(di, dh)
        processed += 1
        print(f"  job {topo}/{tr} done ({len(done) + processed}/{len(all_jobs)})", flush=True)
    total_done = len(done) + processed
    if total_done >= len(all_jobs):
        finalize()
    else:
        print(f"PROGRESS {total_done}/{len(all_jobs)} — re-run to continue", flush=True)


if __name__ == "__main__":
    main()
