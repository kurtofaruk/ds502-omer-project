import sys
import os
from pathlib import Path,PurePosixPath


import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from pathlib import Path
import pandas as pd
import networkx as nx

import random
import math
from tqdm import tqdm
import pickle 

from sklearn.cluster import KMeans
from collections import Counter
from datetime import date,datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "visualisations"))
from plot_route_result import plot_ctsp_result

out_report_list = []
instances_main = pickle.load(open(f"../../data/tsplib_instances.pkl", "rb"))

for sample_idx in tqdm(range(len(instances_main))):
    out_report_list.append(solve_instance(instances_main[sample_idx]))



def solve_instance(sample):
    report_list = []
    """
    Solves a single TSP instance.
    
    sample = instances_main[2]
    """
    def extract_subtours(input_selected_edges):
        active_edges = [((i, j), (k, l)) for (i, j, k, l) in input_selected_edges]
        G = nx.Graph()
        G.add_edges_from(active_edges)
        subtours = [sorted(list(c)) for c in nx.connected_components(G)]
        return subtours


    def get_perm_of_subtours(input_subtour):
        unique_slices = set()
        n = len(input_subtour)
        for i in range(n):
            node_a = input_subtour[i]
            node_b = input_subtour[(i + 1) % n]
            fwd = tuple(list(node_a) + list(node_b))
            unique_slices.add(fwd)
        return list(unique_slices)


    def get_clusters(labels_inputs, C):
        """Map cluster labels (1-based) to lists of customer indices (1-based)."""
        cluster_mapping = {i + 1: [] for i in range(C)}
        for customer, cluster_label in enumerate(labels_inputs, start=1):
            cluster_mapping[cluster_label].append(customer)
        return cluster_mapping


    def get_clustered_nodes(input_clusters, C):
        """
        For each cluster, create a mapping from original node ID → within-cluster index (1-based).
        Returns: {cluster_id: {original_node: indexed_node}}
        """
        node_mapping = {}
        for cluster_id in range(1, C + 1):
            cluster_nodes = input_clusters[cluster_id]
            node_mapping[cluster_id] = {
                original: idx + 1
                for idx, original in enumerate(cluster_nodes)
            }
        return node_mapping


    def get_all_dict(input_coordinates, input_clustered_nodes):
        """
        Build a lookup dict keyed by original node ID.
        """
        new_dict = {}
        for cluster_id, node_map in input_clustered_nodes.items():
            for original_node, indexed_node in node_map.items():
                new_dict[original_node] = {
                    'coordinates': [int(x) for x in input_coordinates[original_node - 1]],
                    'cluster': cluster_id,
                    'indexed_node': indexed_node,
                }
        return new_dict


    def compute_distance_matrix(coord_i, coord_j):
        dist = round(
            float(np.sqrt((coord_i[0] - coord_j[0]) ** 2 + (coord_i[1] - coord_j[1]) ** 2)),
            2
        )
        return dist


    def _prepare_instance(inst):
        N = len(inst['x_coordinates'])
        C = inst['n_clusters']
        coords = np.column_stack((inst['x_coordinates'], inst['y_coordinates']))
        clusters = get_clusters(np.array(inst['cluster_assignments']), C)
        clusters_to_nodes = get_clustered_nodes(clusters, C)
        all_dict = get_all_dict(coords, clusters_to_nodes)
        all_dict = {k: all_dict[k] for k in sorted(all_dict)}
        return N, C, clusters, clusters_to_nodes, all_dict


    def subtourelim(model, where):
        """
        Lazy constraint callback to eliminate subtours.
        """
        if where != GRB.Callback.MIPSOL:
            return

        try:
            vals_x = model.cbGetSolution(model._x)

            selected_edges = gp.tuplelist(
                (i, j, k, l)
                for (i, j, k, l) in model._x.keys()
                if vals_x[i, j, k, l] > 0.5
            )
            
            vals_x=model.getAttr('x', x)
            selected_edges = gp.tuplelist(
                            (i, j, k, l)
                            for (i, j, k, l),v in vals_x
                            if v[i, j, k, l] > 0.5
                        )
            for key,v in vals_x:
                print(key,v)
            
            tours = extract_subtours(selected_edges)

            for subtour in tours:
                clusters_in_subtour = set(node[0] for node in subtour)

                if len(clusters_in_subtour) == len(model._M):
                    continue

                subtour_edges = get_perm_of_subtours(subtour)
                subtour_nodes = set([(i[0], i[1]) for i in subtour_edges])
                subtour_clusters = [(i[0], i[2]) for i in subtour_edges]

                if not subtour_edges:
                    continue

                lhs = gp.quicksum(
                    model._x[i, j, k, l]
                    for (i, j, k, l) in subtour_edges
                    if (i, j, k, l) in model._x
                )

                lhs_y = gp.quicksum(
                    model._y[i, j]
                    for (i, j) in subtour_nodes
                    if (i, j) in model._y
                )

                lhs_z = gp.quicksum(
                    model._z[i, k]
                    for (i, k) in subtour_clusters
                    if (i, k) in model._z
                )

                rhs = len(subtour_edges) - 1

                model.cbLazy(lhs <= rhs)
                model.cbLazy(lhs_y <= rhs)
                model.cbLazy(lhs_z <= rhs)

        except Exception as e:
            print(f"[Callback Error]: {e}")


    N, m, clusters, clusters_to_nodes, all_dict = _prepare_instance(sample)
    M = list(range(1, m + 1))

    raw_counts = Counter(sample["cluster_assignments"])
    nodes_per_cluster = {k: v for k, v in raw_counts.items()}
    N = sum(nodes_per_cluster.values())

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

    start_time = datetime.now()

    env = gp.Env(empty=True)
    env.setParam('LogToConsole', 0)
    env.start()

    model = gp.Model("DS502_Project", env=env)
    model.Params.TimeLimit = 60
    model.Params.Threads = 8
    model.Params.LazyConstraints = 1

    cluster_node_set = {i: list(range(1, nodes_per_cluster[i] + 1)) for i in M}
    cluster_node_set_ = {(i, j): 0 for i in M for j in range(1, nodes_per_cluster[i] + 1)}
    cross_cluster_nodes_set = {
        (i, j, k, l): 0
        for i in M for j in range(1, nodes_per_cluster[i] + 1)
        for k in M for l in range(1, nodes_per_cluster[k] + 1)
        if i != k
    }
    cross_cluster_set = {(i, k): 0 for i in M for k in M if i != k}

    y = model.addVars(cluster_node_set_.keys(), vtype=GRB.BINARY, name="y")
    x = model.addVars(cross_cluster_nodes_set.keys(), vtype=GRB.BINARY, name="x")
    z = model.addVars(cross_cluster_set.keys(), vtype=GRB.BINARY, name="z")

    model.setObjective(
        gp.quicksum(
            cost_matrix[i, j, k, l] * x[i, j, k, l]
            for i in M
            for j in cluster_node_set[i]
            for k in M
            for l in cluster_node_set[k]
            if i != k
        ),
        GRB.MINIMIZE
    )

    for i in M:
        model.addConstr(
            gp.quicksum(y[i, j] for j in cluster_node_set[i]) == 1,
            name=f"one_node_from_cluster_{i}"
        )

    for i in M:
        for j in cluster_node_set[i]:
            model.addConstr(
                gp.quicksum(
                    x[i, j, k, l]
                    for k in M for l in cluster_node_set[k]
                    if i != k
                ) == y[i, j]
            )

    for k in M:
        for l in cluster_node_set[k]:
            model.addConstr(
                gp.quicksum(
                    x[i, j, k, l]
                    for i in M for j in cluster_node_set[i]
                    if i != k
                ) == y[k, l]
            )

    for i in M:
        for k in M:
            if i != k:
                model.addConstr(
                    gp.quicksum(
                        x[i, j, k, l]
                        for j in cluster_node_set[i]
                        for l in cluster_node_set[k]
                    ) == z[i, k]
                )

    model._x = x
    model._y = y
    model._z = z
    model._M = M

    model.write(f"../../reports/lp_models/ds502_project_{sample['key']}.lp")
    model.optimize()
    
    model.optimize(subtourelim)

    end_time = datetime.now()
    runtime = f"{(end_time - start_time).total_seconds():.4f}"
    print(f"Runtime: {runtime}s | Status: {model.Status} | ObjVal: {model.ObjVal if model.SolCount > 0 else 'N/A'}")

    result = {
        'status': model.Status,
        'runtime': runtime,
        'objective': None,
        'gap': None,
        'abs_gap': None,
        'N':N,
        'K':1,
        'C':m,
        'route': None,
        'visited_nodes' : None,
        'visited_clusters': None,
        'cluster_node_inflow':None,
        'cluster_node_outflow':None
    }

    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        result['objective'] = round(model.ObjVal,2)
        result['gap'] = model.MIPGap if model.Status == GRB.TIME_LIMIT else 0.0
        result['abs_gap'] = abs(model.ObjVal - model.ObjBound) if model.Status == GRB.TIME_LIMIT else 0.0
        
        vals_x = model.getAttr('x', x)
        selected_edges = [(i,j,k,l) for (i,j,k,l) in cross_cluster_nodes_set.keys() if vals_x[i,j,k,l] > 0.5]

        cluster_node_inflow = [(i,j) for (i,j,k,l) in cross_cluster_nodes_set.keys() if vals_x[i,j,k,l] > 0.5]
        cluster_node_outflow = [(k,l) for (i,j,k,l) in cross_cluster_nodes_set.keys() if vals_x[i,j,k,l] > 0.5]
        result['cluster_node_inflow'] = list(set(cluster_node_inflow))
        result['cluster_node_outflow'] = list(set(cluster_node_outflow))

        vals_z = model.getAttr('x', z)
        visited_clusters = [(i,k) for (i,k) in cross_cluster_set.keys() if vals_z[i,k] > 0.5]
        result['visited_clusters'] = list(set(visited_clusters))

        print(f"Objective: {result['objective']}")
        print(f"Runtime: {runtime}")
        print(f"Gap: {result['gap']*100:.2f}%")
        print(f"Gap-Nominal: {result['abs_gap']:.2f}")


        adjacency = {}
        for (i, j, k, l) in selected_edges:
            if (i, j) in adjacency:
                print(f"[Warning] Node ({i},{j}) has multiple outgoing edges in solution")
            adjacency[(i, j)] = (k, l)
        
        if not adjacency:
            print("[Warning] No edges found in the solution.")
            return

        start_node = list(adjacency.keys())[0]

        tour = []
        current = start_node
        visited = set()
        while current not in visited:
            visited.add(current)
            tour.append(current)
            current = adjacency.get(current)
            if current is None:
                print("[Warning] Broken tour chain — solution may be infeasible")
                break

        out_route = {0: [node[1] for node in tour] + [tour[0][1]]}
        out_clusters = {0: [node[0] for node in tour] + [tour[0][0]]}

        plot_ctsp_result(
            all_dict, out_route, out_clusters, clusters, result['objective'],
            show=True,
            figsize=None,
            input_title=f"{sample['key']}"
        )

        report_list.append({"key":sample['key'],
                            "obj":result["objective"],
                            "runtime":result["runtime"],
                            "gap":result["gap"],
                            "abs_gap":result["abs_gap"]
                            })
    return report_list

pd.DataFrame(out_report_list).to_excel("../../reports/results_extension_TSP.xlsx")
if __name__ == '__main__':
    run_gurobi_tsp_extension()
