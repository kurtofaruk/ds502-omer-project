"""
Genetic Algorithm for the Clustered TSP (CTSP).

Chromosome: indexed_route list  [indexed_node_for_cluster_1, ..., indexed_node_for_cluster_C]
  position i (0-based) holds the within-cluster indexed_node (1-based) for cluster i+1.

Tour ordering + cost: solved by Gurobi (via call_mp_gurobi_route_level).
Fitness evaluation is parallelised and cached to skip redundant Gurobi calls.

Adapted from gtsp-with-walking-customers-2026 reference (ga_tsp_model_v5.py,
genetic_operators_v2.py):
  - generate_random_route_n_coords      → population initialisation
  - call_mp_gurobi_route_level          → parallel Gurobi batch evaluation (K=1)
  - roulette_wheel_parent_selection     → minimisation roulette wheel selection
  - roulette_wheel_veh_parent_selection → variant for converged-route populations
  - crossover                           → two-point / single-point / uniform
  - mutate                              → point mutation + swap mutation
  - validate_route                      → repair invalid node assignments after operators
"""

import random
import time
import pickle
import sys
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "gurobi_model"))
from gurobi_functions import _prepare_instance
from gurobi_solver_route_parallel import call_mp_gurobi_route_level


# =============================================================================
# FITNESS CACHE  (adapted from ga_tsp_model_v5.py)
# =============================================================================

_fitness_cache: dict = {}
_cache_stats:   dict = {"hits": 0, "misses": 0}
_gurobi_stats:  dict = {"total_calls": 0, "total_gurobi_time": 0.0}


def _cache_key(indexed_route: list) -> tuple:
    return tuple(indexed_route)


def clear_fitness_cache():
    global _fitness_cache, _cache_stats
    _fitness_cache.clear()
    _cache_stats["hits"] = 0
    _cache_stats["misses"] = 0


def clear_gurobi_stats():
    global _gurobi_stats
    _gurobi_stats = {"total_calls": 0, "total_gurobi_time": 0.0}


def get_gurobi_stats() -> dict:
    total = _cache_stats["hits"] + _cache_stats["misses"]
    return {
        "total_gurobi_calls":   _gurobi_stats["total_calls"],
        "total_gurobi_time_s":  round(_gurobi_stats["total_gurobi_time"], 3),
        "avg_time_per_call_ms": round(
            _gurobi_stats["total_gurobi_time"] / max(1, _gurobi_stats["total_calls"]) * 1000, 2),
        "cache_hits":           _cache_stats["hits"],
        "cache_misses":         _cache_stats["misses"],
        "cache_hit_rate_pct":   round(_cache_stats["hits"] / max(1, total) * 100, 1),
    }


def print_gurobi_stats():
    s = get_gurobi_stats()
    print("=" * 55)
    print("  GUROBI / CACHE STATISTICS")
    print("=" * 55)
    print(f"  Total Gurobi calls     : {s['total_gurobi_calls']}")
    print(f"  Total Gurobi time (s)  : {s['total_gurobi_time_s']:.2f}")
    print(f"  Avg time per call (ms) : {s['avg_time_per_call_ms']:.1f}")
    print(f"  Cache hits             : {s['cache_hits']}")
    print(f"  Cache misses           : {s['cache_misses']}")
    print(f"  Cache hit rate         : {s['cache_hit_rate_pct']:.1f}%")
    print("=" * 55)


# =============================================================================
# DATA HELPER
# =============================================================================

def build_clusters_to_coords(all_dict):
    """{ cluster_id: { indexed_node: (x, y) } }  —  used by node optimisation."""
    ctc = {}
    for _, info in all_dict.items():
        c = info["cluster"]
        n = info["indexed_node"]
        x, y = info["coordinates"]
        ctc.setdefault(c, {})[n] = (x, y)
    return ctc


# =============================================================================
# POPULATION INITIALISATION  (adapted from generate_random_route_n_coords)
# =============================================================================

def generate_random_route_n_coords(clusters_to_nodes, ctc, pop_size):
    """
    Generate initial population of indexed_route lists and their coordinates.
    Adapted from data_utils.generate_random_route_n_coords (K=1, no depot).

    Args:
        clusters_to_nodes: {cluster_id: {original_node: indexed_node}}
        ctc:               {cluster_id: {indexed_node: (x, y)}}
        pop_size:          number of individuals

    Returns:
        indexed_routes:  list of [indexed_node_cluster_1, ..., indexed_node_cluster_C]
        route_coords:    list of [[coord_1, ..., coord_C]]  (K=1 wrapping for Gurobi)
    """
    indexed_routes = []
    route_coords   = []
    C = max(k for k in clusters_to_nodes if k != 0)

    for _ in range(pop_size):
        indexed_route = []
        coords        = []
        for cluster_id in range(1, C + 1):
            nodes = list(clusters_to_nodes[cluster_id].values())   # indexed_node values
            node  = random.choice(nodes)
            indexed_route.append(node)
            coords.append(ctc[cluster_id][node])
        indexed_routes.append(indexed_route)
        route_coords.append([coords])          # [[c1, c2, ...]] for K=1
    return indexed_routes, route_coords


def build_route_coords(population, clusters_to_nodes, ctc):
    """
    Rebuild route_coords from a population of indexed_routes.
    Adapted from genetic_operators_v2.build_route_coords (K=1, no depot).
    """
    C = max(k for k in clusters_to_nodes if k != 0)
    route_coords = []
    for indexed_route in population:
        coords = [ctc[i + 1][indexed_route[i]] for i in range(C)]
        route_coords.append([coords])          # [[c1, c2, ...]] for K=1
    return route_coords


# =============================================================================
# BATCH FITNESS EVALUATION  (uses call_mp_gurobi_route_level)
# =============================================================================

def get_cached_fitness(population, ctc, clusters_to_nodes, time_limit, verbose_rate=False):
    """
    Evaluate population fitness using call_mp_gurobi_route_level with caching.
    Adapted from ga_tsp_model_v5.get_cached_fitness.

    Returns:
        fitnesses:  list of tour costs
        seqs:       list of optimal cluster sequences (1-based) from Gurobi
        num_new:    number of new Gurobi evaluations
    """
    C = max(k for k in clusters_to_nodes if k != 0)
    results     = [None] * len(population)
    uncached    = []   # (original_idx, route_coords)

    for i, route in enumerate(population):
        key = _cache_key(route)
        if key in _fitness_cache:
            _cache_stats["hits"] += 1
            results[i] = _fitness_cache[key]
        else:
            _cache_stats["misses"] += 1
            coords = [ctc[j + 1][route[j]] for j in range(C)]
            uncached.append((i, [coords]))     # [coords] = one vehicle route

    if verbose_rate and population:
        hit_rate = (len(population) - len(uncached)) / len(population) * 100
        print(f"  [Cache] hits={len(population)-len(uncached)}, "
              f"misses={len(uncached)}, hit_rate={hit_rate:.1f}%")

    if uncached:
        t0           = time.time()
        batch_coords = [u[1] for u in uncached]   # [[coords_i], ...]
        gurobi_res   = call_mp_gurobi_route_level(
            batch_coords, show_gurobi_progress=False,
            n_threads=1, time_limit=time_limit,
        )
        _gurobi_stats["total_calls"]       += len(uncached)
        _gurobi_stats["total_gurobi_time"] += time.time() - t0

        for (orig_i, _), (obj_Z, obj_routes) in zip(uncached, gurobi_res):
            route_order        = obj_routes[0]                         # K=1
            optimal_cluster_seq = [j + 1 for j in route_order]        # 1-based
            entry = (obj_Z, optimal_cluster_seq)
            _fitness_cache[_cache_key(population[orig_i])] = entry
            results[orig_i] = entry

    fitnesses = [r[0] for r in results]
    seqs      = [r[1] for r in results]
    return fitnesses, seqs, len(uncached)


# =============================================================================
# CHROMOSOME VALIDATION  (adapted from genetic_operators_v2.validate_route)
# =============================================================================

def validate_route(route, clusters_to_nodes):
    """
    Ensure each position's indexed_node is valid for its cluster.
    Adapted from genetic_operators_v2.validate_route.
    """
    validated = list(route)
    for i, node in enumerate(validated):
        cluster_id  = i + 1
        valid_nodes = list(clusters_to_nodes[cluster_id].values())
        if node not in valid_nodes:
            validated[i] = random.choice(valid_nodes)
    return validated


# =============================================================================
# CROSSOVER  (adapted from genetic_operators_v2.crossover, K=1 / route only)
# =============================================================================

def crossover(parent1, parent2, clusters_to_nodes):
    """
    Two-point / single-point / uniform crossover on route genes.
    Adapted from genetic_operators_v2.crossover (vehicle genes removed for K=1).

    Returns two child routes (both validated).
    """
    length = len(parent1)
    method = random.choices(
        ['two_point', 'single_point', 'uniform'], weights=[0.3, 0.3, 0.4]
    )[0]

    if method == 'two_point':
        p1, p2     = sorted(random.sample(range(1, length), 2))
        child1     = parent1[:p1] + parent2[p1:p2] + parent1[p2:]
        child2     = parent2[:p1] + parent1[p1:p2] + parent2[p2:]

    elif method == 'single_point':
        p          = random.randint(1, length - 1)
        child1     = parent1[:p] + parent2[p:]
        child2     = parent2[:p] + parent1[p:]

    else:   # uniform
        child1, child2 = [], []
        for i in range(length):
            if random.random() < 0.5:
                child1.append(parent1[i]); child2.append(parent2[i])
            else:
                child1.append(parent2[i]); child2.append(parent1[i])

    return (validate_route(child1, clusters_to_nodes),
            validate_route(child2, clusters_to_nodes))


# =============================================================================
# MUTATION  (adapted from genetic_operators_v2.mutate, K=1 / route only)
# =============================================================================

def mutate(individual, clusters_to_nodes):
    """
    Point mutation (reassign one cluster's node) or swap mutation (exchange two).
    Adapted from genetic_operators_v2.mutate (vehicle mutations removed for K=1).
    """
    ind         = list(individual)
    node_choice = random.choices(['point', 'swap'], weights=[0.6, 0.4])[0]

    if node_choice == 'point':
        i           = random.randint(0, len(ind) - 1)
        cluster_id  = i + 1
        valid_nodes = list(clusters_to_nodes[cluster_id].values())
        alternatives = [n for n in valid_nodes if n != ind[i]]
        if alternatives:
            ind[i] = random.choice(alternatives)
        elif len(ind) >= 2:
            i, j       = random.sample(range(len(ind)), 2)
            ind[i], ind[j] = ind[j], ind[i]

    elif node_choice == 'swap' and len(ind) >= 2:
        i, j       = random.sample(range(len(ind)), 2)
        ind[i], ind[j] = ind[j], ind[i]

    return validate_route(ind, clusters_to_nodes)


# =============================================================================
# PARENT SELECTION  (adapted from genetic_operators_v2 roulette wheel, minimisation)
# =============================================================================

def _roulette_pick(population, fitness_scores):
    """Roulette wheel pick for minimisation — inverts fitness scores."""
    max_f    = max(fitness_scores)
    inverted = [max_f - f + 1e-10 for f in fitness_scores]
    total    = sum(inverted)
    probs    = [v / total for v in inverted]
    cumprob  = [sum(probs[:i + 1]) for i in range(len(probs))]
    r = random.random()
    for idx, cp in enumerate(cumprob):
        if r <= cp:
            return idx
    return len(population) - 1


def roulette_wheel_parent_selection(population, fitness_scores):
    """
    Select two parents with distinct routes via roulette wheel.
    Adapted from genetic_operators_v2.roulette_wheel_parent_selection (minimisation).
    """
    parents = []
    seen    = set()
    attempts = 0
    unique_routes = len({_cache_key(p) for p in population})

    while len(parents) < 2 and attempts < len(population) * 10:
        idx = _roulette_pick(population, fitness_scores)
        if unique_routes <= 1 or idx not in seen:
            parents.append(list(population[idx]))
            seen.add(idx)
        attempts += 1

    # fallback: random
    while len(parents) < 2:
        idx = random.randrange(len(population))
        parents.append(list(population[idx]))

    return parents


def roulette_wheel_veh_parent_selection(population, fitness_scores):
    """
    Variant used when routes have converged but diversity remains in fitness.
    For K=1 this behaves identically to roulette_wheel_parent_selection.
    Adapted from genetic_operators_v2.roulette_wheel_veh_parent_selection.
    """
    return roulette_wheel_parent_selection(population, fitness_scores)


# =============================================================================
# MAIN GA
# =============================================================================

def run_ga(
    ctc,
    clusters_to_nodes,
    pop_size=100,
    n_generations=100,
    ga_counter=None,
    cx_mut_count=None,
    mut_count=None,
    elite_size=5,
    max_stall_generations=20,
    time_limit_eval=20.0,
    total_time_limit=120.0,
    seed=42,
    verbose=True,
    verbose_rate=False,
):
    """
    GA for CTSP — adapted from ga_tsp_model_v5.run_genetic_algorithm.

    Chromosome: indexed_route list  (position i = indexed_node for cluster i+1).
    Fitness:    Gurobi C-node TSP cost, evaluated in parallel batches with caching.

    Args:
        ctc:                  {cluster_id: {indexed_node: (x, y)}}
        clusters_to_nodes:    {cluster_id: {original_node: indexed_node}}
        pop_size:             population size
        n_generations:        max generations
        ga_counter:           crossover pairs per generation (default: pop_size)
        cx_mut_count:         of those, also apply mutation (default: ga_counter // 2)
        mut_count:            parent-only mutation offspring (default: ga_counter // 3)
        elite_size:           all-time-best elite history size
        max_stall_generations: early-stop after no improvement
        time_limit_eval:      Gurobi time limit per route solve (s)
        total_time_limit:     wall-clock budget (s)
        seed:                 random seed
        verbose:              print progress every 10 gens
        verbose_rate:         print cache hit rate per evaluation

    Returns:
        (best_cost, best_tour, obj_list, gurobi_stats)
    """
    random.seed(seed)
    np.random.seed(seed)
    clear_fitness_cache()
    clear_gurobi_stats()

    if ga_counter  is None: ga_counter  = max(pop_size, 30)
    if cx_mut_count is None: cx_mut_count = ga_counter // 2
    if mut_count    is None: mut_count    = ga_counter // 3

    start_time = datetime.now()

    # ── 1. Initialise population ──────────────────────────────────────────────
    population, _ = generate_random_route_n_coords(clusters_to_nodes, ctc, pop_size)

    # ── 2. Initial fitness evaluation ─────────────────────────────────────────
    fitnesses, seqs, _ = get_cached_fitness(
        population, ctc, clusters_to_nodes, time_limit_eval, verbose_rate)

    best_idx  = int(np.argmin(fitnesses))
    best_cost = fitnesses[best_idx]
    best_route = deepcopy(population[best_idx])
    best_seq   = list(seqs[best_idx])

    elite_solutions = [deepcopy(best_route)]
    stall_count = 0
    obj_list    = {0: best_cost}
    total_unique_keys = pop_size

    if verbose:
        print(f"  Gen 0: best = {best_cost:,.0f}  (pop={pop_size}, C={len(ctc)})")

    # ── 3. Evolution loop ─────────────────────────────────────────────────────
    for gen in range(1, n_generations + 1):

        # Time limit
        if (datetime.now() - start_time) >= timedelta(seconds=total_time_limit):
            if verbose:
                print(f"  Early stop: time limit ({total_time_limit}s) at gen {gen}")
            break

        # Stall limit
        if stall_count >= max_stall_generations:
            if verbose:
                print(f"  Early stop: no improvement for {stall_count} gens")
            break

        # Convergence: all individuals identical
        unique_pop_count = len({_cache_key(p) for p in population})
        if unique_pop_count <= 1:
            if verbose:
                print(f"  Early stop: population converged at gen {gen}")
            break

        # ── Generate offspring ────────────────────────────────────────────────
        offspring = []

        for counter in range(ga_counter):
            # Parent selection
            if unique_pop_count == 1:
                p1, p2 = deepcopy(population[0]), deepcopy(population[0])
            else:
                p1, p2 = roulette_wheel_parent_selection(population, fitnesses)

            # Crossover → 2 children
            c1, c2 = crossover(p1, p2, clusters_to_nodes)
            offspring.extend([c1, c2])

            # Crossover + mutation
            if counter < cx_mut_count:
                m1 = mutate(c1, clusters_to_nodes)
                m2 = mutate(c2, clusters_to_nodes)
                offspring.extend([m1, m2])

            # Mutation-only on parents
            if counter < mut_count:
                m3 = mutate(list(p1), clusters_to_nodes)
                m4 = mutate(list(p2), clusters_to_nodes)
                offspring.extend([m3, m4])

        # ── Survivor selection ────────────────────────────────────────────────
        combined = population + offspring
        c_fit, c_seqs, num_new = get_cached_fitness(
            combined, ctc, clusters_to_nodes, time_limit_eval, verbose_rate)
        total_unique_keys += num_new

        sorted_idx = np.argsort(c_fit)[:pop_size]
        population = [combined[i] for i in sorted_idx]
        fitnesses  = [c_fit[i]    for i in sorted_idx]
        seqs       = [c_seqs[i]   for i in sorted_idx]

        obj_list[gen] = fitnesses[0]

        if fitnesses[0] < best_cost:
            best_cost  = fitnesses[0]
            best_route = deepcopy(population[0])
            best_seq   = list(seqs[0])
            stall_count = 0
            elite_solutions.insert(0, deepcopy(best_route))
            elite_solutions = elite_solutions[:elite_size]
        else:
            stall_count += 1

        if verbose and gen % 10 == 0:
            hit_rate = get_gurobi_stats()["cache_hit_rate_pct"]
            print(f"  Gen {gen:3d}: best = {best_cost:,.0f}  stall={stall_count}  "
                  f"cache={hit_rate:.0f}%  unique={total_unique_keys}")

    if verbose:
        print_gurobi_stats()

    C = len(ctc)
    best_tour = ([(best_seq[j], best_route[best_seq[j] - 1]) for j in range(C)]
                 + [(best_seq[0], best_route[best_seq[0] - 1])])
    return round(best_cost, 2), best_tour, obj_list, get_gurobi_stats()


# =============================================================================
# PER-INSTANCE RUNNER
# =============================================================================

def run_ga_instance(sample, pop_size=100, n_generations=100,
                    time_limit_eval=20.0, total_time_limit=120.0,
                    max_stall_generations=20, seed=42, verbose=False):
    N, C, _, clusters_to_nodes, all_dict = _prepare_instance(sample)
    ctc = build_clusters_to_coords(all_dict)
    t0 = time.time()
    ga_cost, ga_tour, obj_list, gstats = run_ga(
        ctc, clusters_to_nodes,
        pop_size=pop_size, n_generations=n_generations,
        time_limit_eval=time_limit_eval, total_time_limit=total_time_limit,
        max_stall_generations=max_stall_generations,
        seed=seed, verbose=verbose,
    )
    return {
        "key":            sample["key"],
        "N":              N,
        "C":              C,
        "ga_cost":        ga_cost,
        "ga_time":        round(time.time() - t0, 4),
        "generations":    max(obj_list.keys()) if obj_list else 0,
        "gurobi_calls":   gstats["total_gurobi_calls"],
        "cache_hit_rate": gstats["cache_hit_rate_pct"],
        "ga_tour":        ga_tour,
        "obj_list":       obj_list,
    }


# =============================================================================
# PLOTTING
# =============================================================================

def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.5, alpha=0.4, zorder=0)


def plot_convergence(obj_list, key, out_dir):
    out_dir = Path(out_dir)
    gens = sorted(obj_list.keys())
    vals = [obj_list[g] for g in gens]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gens, vals, color="#E9C46A", linewidth=1.5)
    ax.set_title(f"GA convergence — {key}", fontsize=10)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best tour distance")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _style(ax)
    plt.tight_layout()
    plt.savefig(out_dir / f"ga_convergence_{key}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_ga_comparison(ga_df, milp_df, mdp_df, out_dir):
    out_dir = Path(out_dir)
    merged = ga_df.merge(milp_df[["key", "obj"]], on="key", how="left")
    merged = merged.merge(mdp_df[["key", "greedy_cost"]], on="key", how="left")
    merged = merged.dropna(subset=["obj"])
    merged["ga_gap_pct"]     = ((merged["ga_cost"]     - merged["obj"]) / merged["obj"] * 100).round(2)
    merged["greedy_gap_pct"] = ((merged["greedy_cost"] - merged["obj"]) / merged["obj"] * 100).round(2)

    x      = np.arange(len(merged))
    x_keys = merged["key"].tolist()

    fig, ax = plt.subplots(figsize=(16, 6))
    w = 0.25
    ax.bar(x - w, merged["obj"],         width=w, label="MILP optimal",      color="#378ADD", zorder=2)
    ax.bar(x,     merged["ga_cost"],      width=w, label="Genetic Algorithm", color="#E9C46A", zorder=2)
    ax.bar(x + w, merged["greedy_cost"], width=w, label="Greedy (MDP)",      color="#F4A261", zorder=2)
    ax.set_title("Objective: MILP vs GA vs Greedy", fontsize=11)
    ax.set_ylabel("Tour distance")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xticks(x);  ax.set_xticklabels(x_keys, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=9);  _style(ax)
    plt.tight_layout()
    plt.savefig(out_dir / "ga_vs_milp_objective.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - 0.2, merged["ga_gap_pct"],     width=0.38, label="GA gap %",     color="#E9C46A", zorder=2)
    ax.bar(x + 0.2, merged["greedy_gap_pct"], width=0.38, label="Greedy gap %", color="#F4A261", zorder=2)
    ax.set_title("Gap % above MILP optimal", fontsize=11)
    ax.set_ylabel("(heuristic − MILP) / MILP × 100 %")
    ax.set_xticks(x);  ax.set_xticklabels(x_keys, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=9);  _style(ax)
    plt.tight_layout()
    plt.savefig(out_dir / "ga_gap_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nGA gap vs MILP     — mean: {merged['ga_gap_pct'].mean():.1f}%  "
          f"max: {merged['ga_gap_pct'].max():.1f}%  "
          f"min: {merged['ga_gap_pct'].min():.1f}%")
    print(f"Greedy gap vs MILP — mean: {merged['greedy_gap_pct'].mean():.1f}%  "
          f"max: {merged['greedy_gap_pct'].max():.1f}%  "
          f"min: {merged['greedy_gap_pct'].min():.1f}%")
    return merged


# =============================================================================
# MAIN
# =============================================================================

def main():
    base_dir = Path(__file__).resolve().parent

    print("── Loading instances ─────────────────────────────────────────────")
    instances = pickle.load(open(base_dir / "data/tsplib_instances.pkl", "rb"))
    print(f"Loaded {len(instances)} instances.\n")

    print("── Running Genetic Algorithm ─────────────────────────────────────")
    results = []
    for sample in tqdm(instances):
        if sample["n_clusters"] > 21:
            print(f"  Skipping {sample['key']} (C={sample['n_clusters']} > 21)")
            continue
        r = run_ga_instance(
            sample, pop_size=100, n_generations=100,
            time_limit_eval=20.0, total_time_limit=120.0,
            max_stall_generations=20, verbose=True,
        )
        print(f"  {r['key']:12s}  N={r['N']:4d}  C={r['C']:3d}  "
              f"ga_cost={r['ga_cost']:,.0f}  time={r['ga_time']:.1f}s  "
              f"gens={r['generations']}  cache={r['cache_hit_rate']:.0f}%")
        #plt.plot(r['obj_list'].keys(), r['obj_list'].values(), color="#E9C46A", linewidth=1.5)
        results.append(r)

    ga_df = pd.DataFrame([{k: v for k, v in r.items()
                            if k not in ("ga_tour", "obj_list")} for r in results])
    ga_df.to_excel(base_dir / "reports/results_ga.xlsx", index=False)
    ga_df = pd.read_excel(base_dir / "reports/results_ga.xlsx", index_col=0)
    print(f"\nSaved GA results → reports/results_ga.xlsx")

    milp_path = base_dir / "reports/results_extension_TSP.xlsx"
    mdp_path  = base_dir / "reports/results_mdp.xlsx"
    if milp_path.exists() and mdp_path.exists():
        milp_df = pd.read_excel(milp_path, index_col=0)
        mdp_df  = pd.read_excel(mdp_path)
        merged  = plot_ga_comparison(ga_df, milp_df, mdp_df, base_dir / "figures")
        merged.to_excel(base_dir / "reports/results_ga_comparison.xlsx", index=False)
        print("Saved → reports/results_ga_comparison.xlsx")

    print("\nDone.")


if __name__ == "__main__":
    main()
