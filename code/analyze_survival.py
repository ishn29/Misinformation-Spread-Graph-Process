from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.duration.survfunc import survdiff, SurvfuncRight
from statsmodels.stats.multitest import multipletests

ROOT = Path("")
DATA = ROOT / "data"
OUT = ROOT / "analysis_outputs"; OUT.mkdir(exist_ok=True)
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

COVERAGE = 0.05
TARGETS_ORDER = ["betweenness", "collective_influence", "community_bridge",
                 "degree", "eigenvector", "kcore", "pagerank"]


def event_time(df):
    """Return (time, status) arrays: t50 if reached, else censored at duration."""
    reached = df["reached_50"].to_numpy(dtype=float)
    t50 = df["t50"].to_numpy(dtype=float)
    dur = df["duration"].to_numpy(dtype=float)
    time = np.where(reached == 1.0, t50, dur)
    status = (reached == 1.0).astype(int)
    return time, status


def logrank_two_group(target, random_grp):
    """Two-group log-rank via statsmodels survdiff. Returns (chi2, p)."""
    t_t, s_t = event_time(target)
    t_r, s_r = event_time(random_grp)
    time = np.concatenate([t_t, t_r])
    status = np.concatenate([s_t, s_r])
    group = np.concatenate([np.ones(len(t_t)), np.zeros(len(t_r))])
    # Undefined if neither group ever has an event (e.g. WS near threshold).
    if status.sum() == 0:
        return np.nan, np.nan
    chi2, p = survdiff(time, status, group)
    return float(chi2), float(p)


def run_dataset(name, df, targets):
    rows = []
    for topo in ["BA", "ER", "WS"]:
        sub = df[df["topology"] == topo]
        rnd = sub[sub["strategy"] == "random"]
        recs = []
        for strat in targets:
            tgt = sub[sub["strategy"] == strat]
            if len(tgt) == 0:
                continue
            chi2, p = logrank_two_group(tgt, rnd)
            recs.append(dict(dataset=name, topology=topo, strategy=strat,
                             chi2=chi2, p=p,
                             target_reached50=round(tgt["reached_50"].mean(), 4),
                             random_reached50=round(rnd["reached_50"].mean(), 4)))
        # Holm within this (dataset, topology) family, over defined p values.
        pv = [r["p"] for r in recs if not np.isnan(r["p"])]
        if pv:
            adj = multipletests(pv, method="holm")[1]
            it = iter(adj)
            for r in recs:
                r["holm_p"] = float(next(it)) if not np.isnan(r["p"]) else np.nan
        else:
            for r in recs:
                r["holm_p"] = np.nan
        rows.extend(recs)
    return rows


def survival_figure(df_nt, targets):
    """KM survival curves (P[not yet reached 50%]) by strategy, near-threshold.

    Groups with no events (a strategy under which no run ever reaches 50%) are
    drawn as a flat line pinned at 1.0 so they read as 'never reached 50%'
    rather than silently vanishing.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, topo in zip(axes, ["BA", "ER", "WS"]):
        sub = df_nt[df_nt["topology"] == topo]
        # Panel x-extent from the longest observed/censored time in the panel.
        all_t = event_time(sub)[0] if len(sub) else np.array([1.0])
        xmax = float(np.nanmax(all_t)) if len(all_t) else 1.0
        any_event = False
        for strat in ["random"] + targets:
            g = sub[sub["strategy"] == strat]
            if len(g) == 0:
                continue
            time, status = event_time(g)
            lw = 1.8 if strat == "random" else 1.0
            if status.sum() == 0:
                ax.plot([0, xmax], [1.0, 1.0], lw=lw, label=f"{strat} (never)")
            else:
                any_event = True
                sf = SurvfuncRight(time, status)
                ax.step(sf.surv_times, sf.surv_prob, where="post",
                        label=strat, lw=lw)
        ax.set_title(f"{topo}  (beta = 0.02, 5% coverage)")
        ax.set_xlabel("time step")
        ax.set_ylim(-0.03, 1.05)
        if not any_event:
            ax.text(0.5, 0.5, "no run reached 50%", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="0.4")
        ax.legend(fontsize=6.5, loc="lower left")
    axes[0].set_ylabel("P(not yet reached 50% infected)")
    fig.tight_layout()
    fig.savefig(FIG / "survival_curves_near_threshold.png", dpi=200)
    plt.close(fig)


def main():
    inter = pd.read_csv(DATA / "intervention_results.csv")
    rob = pd.read_csv(DATA / "robustness_intervention.csv")

    primary = inter[np.isclose(inter["fraction"], COVERAGE)]
    nt = rob[np.isclose(rob["fraction"], COVERAGE) & np.isclose(rob["beta"], 0.02)]
    nt_targets = [s for s in TARGETS_ORDER if s in set(nt["strategy"])]

    rows = []
    rows += run_dataset("primary_5pct", primary, TARGETS_ORDER)
    rows += run_dataset("near_threshold_beta_0.02", nt, nt_targets)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "survival_tests.csv", index=False)
    survival_figure(nt, nt_targets)

    print(out.to_string(index=False))
    print("\nWrote", OUT / "survival_tests.csv")
    print("Wrote", FIG / "survival_curves_near_threshold.png")


if __name__ == "__main__":
    main()
