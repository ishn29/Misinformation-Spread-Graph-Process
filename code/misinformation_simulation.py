"""Reproducible network-diffusion and structural-immunization utilities.

The model is an idealized SIR-like graph process. Protected nodes are permanently
non-susceptible. The code records active-spreader burden and direct simulated
exposure measures separately.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math
import random

import networkx as nx
import numpy as np

SUSCEPTIBLE = np.int8(0)
INFECTED = np.int8(1)
RECOVERED = np.int8(2)


@dataclass(frozen=True)
class SimulationConfig:
    n: int = 1000
    mean_degree: int = 6
    beta: float = 0.15
    gamma: float = 0.05
    initial_spreaders: int = 5
    max_steps: int = 500
    ws_rewire: float = 0.10
    ba_m: int = 3
    sbm_communities: int = 4
    sbm_mixing: float = 0.10


def build_graph(topology: str, config: SimulationConfig, seed: int) -> nx.Graph:
    """Generate a simple undirected graph with approximately matched density."""
    topology = topology.upper()
    n = config.n
    if topology == "ER":
        p = config.mean_degree / max(n - 1, 1)
        g = nx.fast_gnp_random_graph(n, p, seed=seed)
    elif topology == "WS":
        k = int(config.mean_degree)
        if k % 2:
            k += 1
        k = min(k, n - 1 if (n - 1) % 2 == 0 else n - 2)
        g = nx.watts_strogatz_graph(n, k, config.ws_rewire, seed=seed)
    elif topology == "BA":
        m = max(1, min(int(config.ba_m), n - 1))
        g = nx.barabasi_albert_graph(n, m, seed=seed)
    elif topology == "SBM":
        q = max(2, int(config.sbm_communities))
        sizes = [n // q] * q
        for i in range(n % q):
            sizes[i] += 1
        sbar = float(np.mean(sizes))
        mu = float(np.clip(config.sbm_mixing, 0.0, 0.95))
        # Fraction mu of expected degree is external, distributed over other blocks.
        p_in = min(1.0, (1.0 - mu) * config.mean_degree / max(sbar - 1.0, 1.0))
        p_out = min(1.0, mu * config.mean_degree / max(n - sbar, 1.0))
        probs = [[p_in if i == j else p_out for j in range(q)] for i in range(q)]
        g = nx.stochastic_block_model(sizes, probs, seed=seed)
    else:
        raise ValueError(f"Unknown topology: {topology}")
    g.remove_edges_from(nx.selfloop_edges(g))
    # Relabel ensures compact integer indices for vectorized state arrays.
    if set(g.nodes()) != set(range(n)):
        g = nx.convert_node_labels_to_integers(g, ordering="sorted")
    return g


def graph_metrics(g: nx.Graph) -> Dict[str, float]:
    n = g.number_of_nodes()
    e = g.number_of_edges()
    comps = list(nx.connected_components(g)) if n else []
    lcc = max(comps, key=len) if comps else set()
    clustering = nx.average_clustering(g) if n else np.nan
    transitivity = nx.transitivity(g) if n else np.nan
    assortativity = nx.degree_assortativity_coefficient(g) if e else np.nan
    return {
        "nodes": n,
        "edges": e,
        "mean_degree": (2.0 * e / n) if n else np.nan,
        "density": nx.density(g) if n > 1 else 0.0,
        "clustering": clustering,
        "transitivity": transitivity,
        "assortativity": assortativity,
        "components": len(comps),
        "lcc_fraction": len(lcc) / n if n else np.nan,
    }


def _distance_l_boundary(g: nx.Graph, source: int, radius: int = 2) -> Iterable[int]:
    lengths = nx.single_source_shortest_path_length(g, source, cutoff=radius)
    return (node for node, dist in lengths.items() if dist == radius)


def static_collective_influence_scores(g: nx.Graph, radius: int = 2) -> Dict[int, float]:
    deg = dict(g.degree())
    scores: Dict[int, float] = {}
    for node in g.nodes():
        boundary_sum = sum(max(deg[j] - 1, 0) for j in _distance_l_boundary(g, node, radius))
        scores[node] = max(deg[node] - 1, 0) * boundary_sum
    return scores


def community_bridge_scores(g: nx.Graph) -> Tuple[Dict[int, float], Dict[int, int]]:
    if g.number_of_edges() == 0:
        return {n: 0.0 for n in g}, {n: i for i, n in enumerate(g)}
    communities = list(nx.algorithms.community.greedy_modularity_communities(g))
    membership: Dict[int, int] = {}
    for ci, community in enumerate(communities):
        for node in community:
            membership[node] = ci
    scores: Dict[int, float] = {}
    for node in g:
        external = sum(1 for nbr in g.neighbors(node) if membership.get(nbr) != membership.get(node))
        total = g.degree(node)
        # Prioritize cross-community reach, then ordinary degree for deterministic ties.
        scores[node] = external * (1.0 + math.log1p(total)) + 1e-6 * total
    return scores, membership


def compute_rankings(
    g: nx.Graph,
    seed: int,
    betweenness_k: int = 100,
    include: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, List[int]], Dict[str, float], Dict[int, int]]:
    """Compute nested node rankings and per-method runtime.

    Betweenness is approximated by sampled source nodes for scalability.
    Collective influence uses the published radius-2 score on the intact graph;
    it is labeled static collective influence in outputs.
    """
    methods = list(include or [
        "degree", "betweenness", "pagerank", "eigenvector", "kcore",
        "collective_influence", "community_bridge"
    ])
    rankings: Dict[str, List[int]] = {}
    runtimes: Dict[str, float] = {}
    membership: Dict[int, int] = {n: 0 for n in g}

    for method in methods:
        start = perf_counter()
        if method == "degree":
            score = dict(g.degree())
        elif method == "betweenness":
            k = min(max(10, betweenness_k), g.number_of_nodes())
            score = nx.betweenness_centrality(g, k=k, normalized=True, seed=seed)
        elif method == "pagerank":
            score = nx.pagerank(g, alpha=0.85, max_iter=200, tol=1e-8)
        elif method == "eigenvector":
            try:
                score = nx.eigenvector_centrality(g, max_iter=500, tol=1e-7)
            except nx.PowerIterationFailedConvergence:
                score = nx.eigenvector_centrality_numpy(g)
        elif method == "kcore":
            score = nx.core_number(g) if g.number_of_edges() else {n: 0 for n in g}
        elif method == "collective_influence":
            score = static_collective_influence_scores(g, radius=2)
        elif method == "community_bridge":
            score, membership = community_bridge_scores(g)
        else:
            raise ValueError(method)
        rankings[method] = sorted(g.nodes(), key=lambda n: (-float(score[n]), -g.degree(n), int(n)))
        runtimes[method] = perf_counter() - start
    return rankings, runtimes, membership


def select_nodes(
    g: nx.Graph,
    strategy: str,
    count: int,
    rng: np.random.Generator,
    rankings: Optional[Mapping[str, Sequence[int]]] = None,
) -> List[int]:
    count = int(max(0, min(count, g.number_of_nodes())))
    if count == 0 or strategy == "none":
        return []
    if strategy == "random":
        return list(map(int, rng.choice(np.array(list(g.nodes()), dtype=int), size=count, replace=False)))
    if rankings is None or strategy not in rankings:
        raise ValueError(f"Ranking unavailable for strategy {strategy}")
    return list(map(int, rankings[strategy][:count]))


def make_behavior_multipliers(
    g: nx.Graph,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    n = g.number_of_nodes()
    if mode == "homogeneous":
        mult = np.ones(n, dtype=float)
    elif mode == "moderate":
        mult = rng.lognormal(mean=0.0, sigma=0.45, size=n)
    elif mode == "strong":
        mult = rng.lognormal(mean=0.0, sigma=0.90, size=n)
    elif mode == "degree_correlated":
        deg = np.array([g.degree(i) for i in range(n)], dtype=float)
        if deg.std() > 0:
            z = (deg - deg.mean()) / deg.std()
        else:
            z = np.zeros(n)
        mult = np.exp(0.45 * z + rng.normal(0.0, 0.25, size=n))
    else:
        raise ValueError(mode)
    mult /= max(mult.mean(), 1e-12)
    return mult


def choose_seeds(
    g: nx.Graph,
    count: int,
    protected: Sequence[int],
    mode: str,
    rng: np.random.Generator,
    membership: Optional[Mapping[int, int]] = None,
) -> List[int]:
    protected_set = set(protected)
    candidates = [n for n in g.nodes() if n not in protected_set]
    count = min(count, len(candidates))
    if mode == "random":
        return list(map(int, rng.choice(np.array(candidates, dtype=int), size=count, replace=False)))
    if mode == "high_degree":
        return sorted(candidates, key=lambda n: (-g.degree(n), n))[:count]
    if mode == "peripheral":
        return sorted(candidates, key=lambda n: (g.degree(n), n))[:count]
    if mode == "bridge":
        scores, _ = community_bridge_scores(g)
        return sorted(candidates, key=lambda n: (-scores[n], -g.degree(n), n))[:count]
    raise ValueError(mode)


def simulate_diffusion(
    g: nx.Graph,
    config: SimulationConfig,
    seed: int,
    protected: Sequence[int] = (),
    seed_mode: str = "random",
    behavior_mode: str = "homogeneous",
    initial_nodes: Optional[Sequence[int]] = None,
    membership: Optional[Mapping[int, int]] = None,
    return_curve: bool = False,
    return_node_details: bool = False,
) -> Tuple[Dict[str, float], Optional[List[int]], Optional[Dict[str, object]]]:
    """Run a synchronous SIR-like process and return outcome metrics."""
    rng = np.random.default_rng(seed)
    n = g.number_of_nodes()
    state = np.full(n, SUSCEPTIBLE, dtype=np.int8)
    protected_arr = np.array(sorted(set(map(int, protected))), dtype=int)
    if len(protected_arr):
        state[protected_arr] = RECOVERED
    if initial_nodes is None:
        initial_nodes = choose_seeds(
            g, config.initial_spreaders, protected_arr.tolist(), seed_mode, rng, membership
        )
    initial_nodes = list(map(int, initial_nodes))
    state[initial_nodes] = INFECTED
    infected = set(initial_nodes)
    ever_infected = set(initial_nodes)
    exposed_nodes: set[int] = set()
    attempted_exposures = 0
    successful_transmissions = 0
    curve: List[int] = [len(infected)]
    t10 = 0 if len(ever_infected) >= 0.10 * n else np.nan
    t50 = 0 if len(ever_infected) >= 0.50 * n else np.nan
    multipliers = make_behavior_multipliers(g, behavior_mode, rng)
    infection_time = {node: 0 for node in initial_nodes}
    parent: Dict[int, int] = {}

    for step in range(1, config.max_steps + 1):
        if not infected:
            break
        snapshot_susceptible = state == SUSCEPTIBLE
        newly_infected: set[int] = set()
        newly_parent: Dict[int, int] = {}
        # Each infected-susceptible edge at the start of a step is one direct exposure attempt.
        for source in tuple(infected):
            p = float(np.clip(config.beta * multipliers[source], 0.0, 1.0))
            for target in g.neighbors(source):
                if snapshot_susceptible[target]:
                    attempted_exposures += 1
                    exposed_nodes.add(int(target))
                    if target not in newly_infected and rng.random() < p:
                        newly_infected.add(int(target))
                        newly_parent[int(target)] = int(source)
        recovering = {node for node in infected if rng.random() < config.gamma}
        if recovering:
            state[list(recovering)] = RECOVERED
        if newly_infected:
            state[list(newly_infected)] = INFECTED
            successful_transmissions += len(newly_infected)
            for node in newly_infected:
                infection_time[node] = step
                if node in newly_parent:
                    parent[node] = newly_parent[node]
        infected.difference_update(recovering)
        infected.update(newly_infected)
        ever_infected.update(newly_infected)
        curve.append(len(infected))
        if np.isnan(t10) and len(ever_infected) >= 0.10 * n:
            t10 = float(step)
        if np.isnan(t50) and len(ever_infected) >= 0.50 * n:
            t50 = float(step)

    duration = len(curve) - 1
    auc = float(np.sum(curve))
    unique_exposed = len(exposed_nodes)
    repeat_exposures = max(0, attempted_exposures - unique_exposed)
    metrics: Dict[str, float] = {
        "t10": float(t10) if not np.isnan(t10) else np.nan,
        "t50": float(t50) if not np.isnan(t50) else np.nan,
        "reached_10": float(not np.isnan(t10)),
        "reached_50": float(not np.isnan(t50)),
        "peak_infected": float(max(curve) if curve else 0),
        "final_size": float(len(ever_infected) / n),
        "active_spreader_auc": auc,
        "unique_exposed_fraction": float(unique_exposed / n),
        "attempted_exposures": float(attempted_exposures),
        "repeat_exposures": float(repeat_exposures),
        "successful_transmissions": float(successful_transmissions),
        "duration": float(duration),
        "protected_fraction": float(len(protected_arr) / n),
    }
    details: Optional[Dict[str, object]] = None
    if return_node_details:
        details = {
            "initial_nodes": initial_nodes,
            "protected_nodes": protected_arr.tolist(),
            "ever_infected_nodes": sorted(ever_infected),
            "exposed_nodes": sorted(exposed_nodes),
            "infection_time": infection_time,
            "parent": parent,
        }
    return metrics, curve if return_curve else None, details


def removal_structure_metrics(
    g: nx.Graph,
    protected: Sequence[int],
    membership: Optional[Mapping[int, int]] = None,
) -> Dict[str, float]:
    n = g.number_of_nodes()
    protected_set = set(protected)
    h = g.copy()
    incident_edges = sum(1 for u, v in g.edges() if u in protected_set or v in protected_set)
    cross_incident = 0
    if membership:
        cross_incident = sum(
            1 for u, v in g.edges()
            if (u in protected_set or v in protected_set) and membership.get(u) != membership.get(v)
        )
    h.remove_nodes_from(protected_set)
    comps = list(nx.connected_components(h)) if h.number_of_nodes() else []
    lcc = max(comps, key=len) if comps else set()
    # Approximate mean shortest path by sampling at most 100 nodes in LCC.
    asp = np.nan
    if len(lcc) > 1:
        nodes = sorted(lcc)
        sample = nodes if len(nodes) <= 20 else nodes[:20]
        vals = []
        for s in sample:
            lengths = nx.single_source_shortest_path_length(h, s)
            vals.extend(lengths[t] for t in sample if t != s and t in lengths)
        if vals:
            asp = float(np.mean(vals))
    return {
        "remaining_nodes": h.number_of_nodes(),
        "remaining_edges": h.number_of_edges(),
        "components_after": len(comps),
        "lcc_fraction_original_n": len(lcc) / n if n else np.nan,
        "incident_edges_blocked": incident_edges,
        "incident_edge_fraction": incident_edges / max(g.number_of_edges(), 1),
        "cross_community_edges_blocked": cross_incident,
        "approx_mean_path_after": asp,
        "protected_mean_degree": float(np.mean([g.degree(x) for x in protected])) if protected else 0.0,
    }


def config_dict(config: SimulationConfig) -> Dict[str, object]:
    return asdict(config)
