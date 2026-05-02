import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "visualisations"))
from plot_route_result import plot_ctsp_result

sys.path.insert(0, str(Path(__file__).parent.parent / "gurobi_model"))
from gurobi_functions import _prepare_instance, compute_distance_matrix, subtourelim

# LP files are written to <project_root>/reports/lp_models/
_LP_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "lp_models"


def run_gurobi_model_for_instance(input_instances_main, idx, input_time_limit=60, n_thread=8):
    report_list = []
    try:
        sample = input_instances_main[idx]
        N, m, clusters, clusters_to_nodes, all_dict = _prepare_instance(sample)
        M = list(range(1, m + 1))

        raw_counts = Counter(sample["cluster_assignments"])
        n_i = {k: v for k, v in raw_counts.items()}
        N = sum(n_i.values())

        # ── Build cost matrix ──────────────────────────────────────────────────
        cost_matrix = {}
        for n1 in all_dict:
            for n2 in all_dict:
                i = all_dict[n1]["cluster"]
                j = all_dict[n1]["indexed_node"]
                k = all_dict[n2]["cluster"]
                l = all_dict[n2]["indexed_node"]
                if (i, j) == (k, l):
                    cost_matrix[i, j, k, l] = 0
                else:
                    coord_a = all_dict[n1]["coordinates"]
                    coord_b = all_dict[n2]["coordinates"]
                    cost_matrix[i, j, k, l] = compute_distance_matrix(coord_a, coord_b)

        # ── Model setup ────────────────────────────────────────────────────────
        start_time = datetime.now()

        env = gp.Env(empty=True)
        env.setParam('LogToConsole', 0)
        env.start()

        model = gp.Model("DS502_Project", env=env)
        model.Params.TimeLimit = input_time_limit
        model.Params.Threads = n_thread
        model.Params.LazyConstraints = 1

        # ── Variable index sets ────────────────────────────────────────────────
        cluster_node_set = {i: list(range(1, n_i[i] + 1)) for i in M}

        cluster_node_set_ = {(i, j) for i in M for j in range(1, n_i[i] + 1)}

        cross_cluster_nodes_set = {
            (i, j, k, l)
            for i in M for j in range(1, n_i[i] + 1)
            for k in M for l in range(1, n_i[k] + 1)
            if i != k
        }

        # ── Variables ──────────────────────────────────────────────────────────
        y = model.addVars(cluster_node_set_, vtype=GRB.BINARY, name="y")
        x = model.addVars(cross_cluster_nodes_set, vtype=GRB.BINARY, name="x")

        # ── Objective ──────────────────────────────────────────────────────────
        model.setObjective(
            gp.quicksum(
                cost_matrix[i, j, k, l] * x[i, j, k, l]
                for (i, j, k, l) in cross_cluster_nodes_set
            ),
            GRB.MINIMIZE
        )

        # ── Constraints ────────────────────────────────────────────────────────

        # C1: Exactly one node selected per cluster
        for i in M:
            model.addConstr(
                gp.quicksum(y[i, j] for j in cluster_node_set[i]) == 1,
                name=f"one_node_per_cluster_{i}"
            )

        # C2: Out-Flow: selected node sends exactly one arc out
        for i in M:
            for j in cluster_node_set[i]:
                model.addConstr(
                    gp.quicksum(
                        x[i, j, k, l]
                        for k in M for l in cluster_node_set[k]
                        if i != k
                    ) == y[i, j],
                    name=f"outflow_{i}_{j}"
                )

        # C3: In-Flow: selected node receives exactly one arc in
        for k in M:
            for l in cluster_node_set[k]:
                model.addConstr(
                    gp.quicksum(
                        x[i, j, k, l]
                        for i in M for j in cluster_node_set[i]
                        if i != k
                    ) == y[k, l],
                    name=f"inflow_{k}_{l}"
                )

        model._x = x
        model._y = y
        model._M = M

        _LP_DIR.mkdir(parents=True, exist_ok=True)
        model.write(str(_LP_DIR / f"ds502_project_{sample['key']}.lp"))
        model.optimize(subtourelim)

        end_time = datetime.now()
        runtime = f"{(end_time - start_time).total_seconds():.4f}"

        result = {
            'status': model.Status,
            'runtime': runtime,
            'objective': None,
            'gap': None,
            'abs_gap': None,
            'N': N,
            'K': 1,
            'C': m,
            'route': None,
            'visited_nodes': None,
            'visited_clusters': None,
            'cluster_node_inflow': None,
            'cluster_node_outflow': None,
        }

        if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
            result['objective'] = round(model.ObjVal, 2)
            result['gap']     = model.MIPGap if model.Status == GRB.TIME_LIMIT else 0.0
            result['abs_gap'] = abs(model.ObjVal - model.ObjBound) if model.Status == GRB.TIME_LIMIT else 0.0

            vals_x = model.getAttr('x', x)
            selected_edges = [(i, j, k, l) for (i, j, k, l) in cross_cluster_nodes_set
                              if vals_x[i, j, k, l] > 0.5]

            cluster_node_inflow  = [(i, j) for (i, j, k, l) in selected_edges]
            cluster_node_outflow = [(k, l) for (i, j, k, l) in selected_edges]
            result['cluster_node_inflow']  = list(set(cluster_node_inflow))
            result['cluster_node_outflow'] = list(set(cluster_node_outflow))

            print(f"Instance: {sample['key']}  N={N}  C={m}  "
                  f"Obj={result['objective']}  Runtime={runtime}s  "
                  f"Gap={result['gap'] * 100:.2f}%  AbsGap={result['abs_gap']:.2f}")
        else:
            selected_edges = []

        # ── Build tour from solution ───────────────────────────────────────────
        adjacency = {}
        for (i, j, k, l) in selected_edges:
            if (i, j) in adjacency:
                print(f"[Warning] Node ({i},{j}) has multiple outgoing edges")
            adjacency[(i, j)] = (k, l)

        if adjacency:
            start = list(adjacency.keys())[0]
            tour, current, visited = [], start, set()
            while current not in visited:
                visited.add(current)
                tour.append(current)
                current = adjacency.get(current)
                if current is None:
                    break

            out_route    = {0: [node[1] for node in tour] + [tour[0][1]]}
            out_clusters = {0: [node[0] for node in tour] + [tour[0][0]]}

            plot_ctsp_result(
                all_dict, out_route, out_clusters, clusters, result['objective'],
                show=False,
                figsize=None,
                input_title=f"{sample['key']}",
                save_path=f"./figures/{sample['key']}_{N}_{m}_extension",
            )

        report_list.append({
            "key":     sample['key'],
            "obj":     result["objective"],
            "runtime": result["runtime"],
            "gap":     result["gap"],
            "abs_gap": result["abs_gap"],
        })
        return report_list

    except Exception as e:
        print(f"[ERROR] run_gurobi_model_for_instance idx={idx}: {e}")
        return []
