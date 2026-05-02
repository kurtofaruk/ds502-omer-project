"""
MDP / DP solution for the Clustered TSP (CTSP).

Sequential decision structure:
  Stage t: choose next unvisited cluster k and node l within it.
  State:  s_t = ( (i, j), V )  — current (cluster, node) + frozenset of visited clusters
  Action: a   = (k, l)         — next (cluster, node), k not in V
  Cost:   Euclidean distance from (i,j) to (k,l)
  Terminal: return to start node after all C clusters visited.

Bellman equation:
  V_t((i,j), V) = min_{(k,l): k not in V} { d((i,j),(k,l)) + V_{t+1}((k,l), V|{k}) }
  V_C((i,j), M) = d((i,j), start_node)

Methods:
  - Exact DP (Held-Karp style): optimal, feasible only for small C (<= 14).
  - Greedy heuristic:           always picks nearest unvisited cluster-node. Runs on all instances.
"""

import math
import pickle
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "gurobi_model"))
from gurobi_functions import _prepare_instance

DP_THRESHOLD = 15   # run exact DP only when C <= this value (C=14 takes ~100s in Python)


# ── Core distance ──────────────────────────────────────────────────────────────

def _dist(coord_a, coord_b):
    return math.sqrt((coord_a[0] - coord_b[0]) ** 2 + (coord_a[1] - coord_b[1]) ** 2)


# ── Data conversion ────────────────────────────────────────────────────────────

def build_clusters_to_coords(all_dict):
    """
    Convert all_dict (from _prepare_instance) to the MDP state-space format:
        { cluster_id: { indexed_node: (x, y) } }
    """
    ctc = {}
    for _, info in all_dict.items():
        c = info['cluster']
        n = info['indexed_node']
        x, y = info['coordinates']
        ctc.setdefault(c, {})[n] = (x, y)
    return ctc


# ── Greedy heuristic (MDP greedy policy) ──────────────────────────────────────

def greedy_from_start(ctc, start_cl, start_nd):
    """
    Greedy policy: at each stage select the nearest unvisited (cluster, node).
    ctc: { cluster_id: { node_id: (x, y) } }
    Returns (total_cost, tour) where tour is list of (cluster, node) pairs.
    """
    all_cls = set(ctc.keys())
    visited = {start_cl}
    cur = (start_cl, start_nd)
    tour = [cur]
    total = 0.0

    while len(visited) < len(all_cls):
        cur_coord = ctc[cur[0]][cur[1]]
        best_d, best_next = math.inf, None
        for k in all_cls - visited:
            for l, coord in ctc[k].items():
                d = _dist(cur_coord, coord)
                if d < best_d:
                    best_d, best_next = d, (k, l)
        total += best_d
        cur = best_next
        visited.add(cur[0])
        tour.append(cur)

    total += _dist(ctc[cur[0]][cur[1]], ctc[start_cl][start_nd])
    tour.append((start_cl, start_nd))
    return total, tour


def best_greedy(ctc):
    """
    Run greedy from the first node of every cluster and return the best result.
    Trying one start per cluster (C starts) is fast and covers diverse start points.
    """
    best_cost, best_tour = math.inf, []
    for cl, nodes in ctc.items():
        first_nd = min(nodes.keys())
        cost, tour = greedy_from_start(ctc, cl, first_nd)
        if cost < best_cost:
            best_cost, best_tour = cost, tour
    return round(best_cost, 2), best_tour


# ── Exact DP (Held-Karp for CTSP) ─────────────────────────────────────────────

def dp_from_start(ctc, start_cl, start_nd):
    """
    Exact DP from a fixed starting node.
    State:    (current (cluster,node), frozenset of visited clusters)
    Computes: minimum cost to visit all remaining clusters and return to start.
    """
    all_cls = frozenset(ctc.keys())
    start_coord = ctc[start_cl][start_nd]

    @lru_cache(maxsize=None)
    def V(pos, visited):
        remaining = all_cls - visited
        if not remaining:
            return _dist(ctc[pos[0]][pos[1]], start_coord)
        cur_coord = ctc[pos[0]][pos[1]]
        return min(
            _dist(cur_coord, ctc[k][l]) + V((k, l), visited | frozenset([k]))
            for k in remaining
            for l in ctc[k]
        )

    cost = V((start_cl, start_nd), frozenset([start_cl]))
    V.cache_clear()
    return cost


def best_dp(ctc):
    """
    Exact DP from the first node of every cluster; return the minimum tour cost.
    Only feasible for small C (C <= DP_THRESHOLD).
    """
    best_cost = math.inf
    for cl, nodes in ctc.items():
        first_nd = min(nodes.keys())
        cost = dp_from_start(ctc, cl, first_nd)
        if cost < best_cost:
            best_cost = cost
    return round(best_cost, 2)


# ── Per-instance runner ────────────────────────────────────────────────────────

def run_mdp_instance(sample):
    """
    Run MDP approach on one instance dict (from data_prep.build_instances).
    Returns a result dict with greedy and (if small) exact DP results.
    """
    N, C, _, _, all_dict = _prepare_instance(sample)
    ctc = build_clusters_to_coords(all_dict)

    t0 = time.time()
    greedy_cost, _ = best_greedy(ctc)
    greedy_time = round(time.time() - t0, 4)

    dp_cost, dp_time = None, None
    if C <= DP_THRESHOLD:
        t0 = time.time()
        dp_cost = best_dp(ctc)
        dp_time = round(time.time() - t0, 4)

    return {
        'key':         sample['key'],
        'N':           N,
        'C':           C,
        'greedy_cost': greedy_cost,
        'greedy_time': greedy_time,
        'dp_cost':     dp_cost,
        'dp_time':     dp_time,
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def _style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linewidth=0.5, alpha=0.4, zorder=0)


def plot_comparison(mdp_df, milp_df, out_dir):
    """
    Generate comparison plots: greedy vs MILP objective, runtime, greedy gap %.
    out_dir =  base_dir / "figures"
    """
    out_dir = Path(out_dir)
    merged = mdp_df.merge(milp_df[['key', 'obj', 'runtime']], on='key', suffixes=('', '_milp'))
    merged = merged.dropna(subset=['obj'])
    merged['greedy_gap_pct'] = ((merged['greedy_cost'] - merged['obj']) / merged['obj'] * 100).round(2)

    x = np.arange(len(merged))
    x_keys = merged['key'].tolist()

    # 1. Greedy vs MILP objective
    fig, ax = plt.subplots(figsize=(14, 6))
    w = 0.35
    ax.bar(x - w / 2, merged['obj'],         width=w, label='MILP optimal', color='#378ADD', zorder=2)
    ax.bar(x + w / 2, merged['greedy_cost'], width=w, label='Greedy (MDP)', color='#F4A261', zorder=2)
    ax.set_title('Objective: MILP vs Greedy heuristic', fontsize=11)
    ax.set_ylabel('Tour distance')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.set_xticks(x); ax.set_xticklabels(x_keys, rotation=45, ha='right', fontsize=7)
    ax.legend(fontsize=9)
    _style(ax)
    plt.tight_layout()
    plt.savefig(out_dir / 'mdp_vs_milp_objective.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Greedy gap %
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x, merged['greedy_gap_pct'], color='#E76F51', width=0.6, zorder=2)
    ax.set_title('Greedy gap % above MILP optimal', fontsize=11)
    ax.set_ylabel('(greedy − MILP) / MILP × 100 %')
    ax.set_xticks(x); ax.set_xticklabels(x_keys, rotation=45, ha='right', fontsize=7)
    _style(ax)
    plt.tight_layout()
    plt.savefig(out_dir / 'mdp_greedy_gap.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Greedy runtime vs MILP runtime
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(merged['runtime'].astype(float), merged['greedy_time'],
               color='#2A9D8F', s=60, zorder=3, edgecolors='white', linewidths=0.6)
    lim = max(merged['runtime'].astype(float).max(), merged['greedy_time'].max()) * 1.05
    ax.plot([0, lim], [0, lim], '--', color='#aaa', linewidth=1, label='y=x (equal time)')
    ax.set_xlabel('MILP runtime (s)')
    ax.set_ylabel('Greedy runtime (s)')
    ax.set_title('Runtime: MILP vs Greedy', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(linewidth=0.5, alpha=0.4)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / 'mdp_runtime_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. DP vs MILP for small instances (where dp_cost is available)
    small = merged.dropna(subset=['dp_cost'])
    if not small.empty:
        xs = np.arange(len(small))
        x_sm = small['key'].tolist()
        fig, ax = plt.subplots(figsize=(10, 5))
        w = 0.25
        ax.bar(xs - w, small['obj'],         width=w, label='MILP optimal', color='#378ADD', zorder=2)
        ax.bar(xs,     small['dp_cost'],      width=w, label='Exact DP',     color='#2A9D8F', zorder=2)
        ax.bar(xs + w, small['greedy_cost'], width=w, label='Greedy (MDP)', color='#F4A261', zorder=2)
        ax.set_title('Small instances: MILP vs Exact DP vs Greedy', fontsize=11)
        ax.set_ylabel('Tour distance')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
        ax.set_xticks(xs); ax.set_xticklabels(x_sm, rotation=30, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        _style(ax)
        plt.tight_layout()
        plt.savefig(out_dir / 'mdp_small_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nGreedy gap %  — mean: {merged['greedy_gap_pct'].mean():.1f}%  "
          f"max: {merged['greedy_gap_pct'].max():.1f}%  "
          f"min: {merged['greedy_gap_pct'].min():.1f}%")
    return merged


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    base_dir = Path(__file__).resolve().parent

    print("── Loading instances ─────────────────────────────────────────────")
    instances = pickle.load(open(base_dir / "data/tsplib_instances.pkl", "rb"))
    print(f"Loaded {len(instances)} instances.\n")

    print("── Running MDP (greedy + exact DP for small instances) ───────────")
    results = []
    for sample in tqdm(instances[3:]):
        
        # sample=instances[3]  # for quick testing
        r = run_mdp_instance(sample)
        tag = f"  greedy={r['greedy_cost']:,.0f}"
        if r['dp_cost'] is not None:
            tag += f"  dp={r['dp_cost']:,.0f}"
        print(f"  {r['key']:12s}  N={r['N']:4d}  C={r['C']:3d}{tag}")
        results.append(r)

    mdp_df = pd.DataFrame(results)
    mdp_df.sort_values("C").to_excel(base_dir / "reports/results_mdp.xlsx", index=False)
    print(f"\nSaved MDP results → reports/results_mdp.xlsx")

    print("\n── Comparison against MILP ───────────────────────────────────────")
    milp_path = base_dir / "reports/results_extension_TSP.xlsx"
    if milp_path.exists():
        milp_df = pd.read_excel(milp_path, index_col=0)
        merged = plot_comparison(mdp_df, milp_df, base_dir / "figures")
        merged.to_excel(base_dir / "reports/results_comparison.xlsx", index=False)
        print("Saved comparison → reports/results_comparison.xlsx")
        print("Figures saved to figures/")
    else:
        print(f"MILP results not found at {milp_path}. Run main.py first.")

    print("\nDone.")


if __name__ == "__main__":
    main()
