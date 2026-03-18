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

sys.path.insert(0, str(Path(__file__).parent.parent / "gurobi_model"))
from gurobi_functions import _prepare_instance,compute_distance_matrix,subtourelim
 
 
def run_gurobi_model_for_instance(input_instances_main,idx,input_time_limit=60,n_thread=8):
    # input_instances_main,idx = instances_main,0
    report_list=[]
    try:
        sample = input_instances_main[idx]
        N, m, clusters, clusters_to_nodes, all_dict = _prepare_instance(sample)
        M = list(range(1, m + 1))

        raw_counts = Counter(sample["cluster_assignments"])
        n_i = {k: v for k, v in raw_counts.items()}  # 1-based cluster keys
        N = sum(n_i.values())

        # ── Build cost matrix ──────────────────────────────────────────────────────────
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

        # ── Model setup ────────────────────────────────────────────────────────────────
        start_time = datetime.now()

        env = gp.Env(empty=True)
        env.setParam('LogToConsole', 0)
        env.start()

        model = gp.Model("DS502_Project", env=env)
        model.Params.TimeLimit = input_time_limit
        model.Params.Threads = n_thread
        model.Params.LazyConstraints = 1

        # ── Variable index sets ────────────────────────────────────────────────────────
        cluster_node_set = {i: list(range(1, n_i[i] + 1)) for i in M}

        # y[i,j]: 1 if node j in cluster i is selected — unchanged, already minimal
        cluster_node_set_ = {(i, j) for i in M for j in range(1, n_i[i] + 1)}

        # x[i,j,k,l]: directed arc from node j in cluster i to node l in cluster k.
        # Kept as directed (i != k) to support asymmetric cost matrices.
        # If your costs are symmetric, you can restrict to i < k and halve this set.
        cross_cluster_nodes_set = {
            (i, j, k, l)
            for i in M for j in range(1, n_i[i] + 1)
            for k in M for l in range(1, n_i[k] + 1)
            if i != k
        }

        # ── Variables ──────────────────────────────────────────────────────────────────
        # z removed entirely — it was just an alias for aggregated x, adding no value.
        y = model.addVars(cluster_node_set_, vtype=GRB.BINARY, name="y")
        x = model.addVars(cross_cluster_nodes_set, vtype=GRB.BINARY, name="x")

        # ── Objective ──────────────────────────────────────────────────────────────────
        model.setObjective(
            gp.quicksum(
                cost_matrix[i, j, k, l] * x[i, j, k, l]
                for (i, j, k, l) in cross_cluster_nodes_set
            ),
            GRB.MINIMIZE
        )

        # ── Constraints ────────────────────────────────────────────────────────────────

        #! C1: Exactly one node selected per cluster
        for i in M:
            model.addConstr(
                gp.quicksum(y[i, j] for j in cluster_node_set[i]) == 1,
                name=f"one_node_per_cluster_{i}"
            )

        #! C2: Out-Flow: selected node sends exactly one arc out
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

        #! C3: In-Flow: selected node receives exactly one arc in
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

        #model.optimize()
    
        model.write(f"../../reports/lp_models/ds502_project_{sample['key']}.lp")
        model.optimize(subtourelim)
        
        end_time = datetime.now()
        runtime = f"{(end_time - start_time).total_seconds():.4f}"
        #print(f"Runtime: {runtime}s | Status: {model.Status} | ObjVal: {model.ObjVal if model.SolCount > 0 else 'N/A'}")

        # Extract solution
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
            
            # Extract selected edges
            vals_x = model.getAttr('x', x)
            selected_edges = [(i,j,k,l) for (i,j,k,l) in cross_cluster_nodes_set if vals_x[i,j,k,l] > 0.5]

            # Extract visited nodes
            vals_y = model.getAttr('x', y)
            cluster_node_inflow = [(i,j) for (i,j,k,l) in cross_cluster_nodes_set if vals_x[i,j,k,l] > 0.5]
            cluster_node_outflow = [(k,l) for (i,j,k,l) in cross_cluster_nodes_set if vals_x[i,j,k,l] > 0.5]
            #visited = [(i,j) for (i,j) in cluster_node_set_ if vals_y[i,j] > 0.5]
            #visited = [(i,j) for (i,j) in cluster_node_set_ if vals_y[i,j] > 0.5]
            result['cluster_node_inflow'] = list(set(cluster_node_inflow))  # Remove duplicates from multi-vehicle
            result['cluster_node_outflow'] = list(set(cluster_node_outflow))  # Remove duplicates from multi-vehicle

            
            
            # vals_z = model.getAttr('x', z)
            # visited = [(i,k) for (i,k) in cross_cluster_set.keys() if vals_z[i,k] > 0.5]
            
            #result['visited_clusters'] = list(set(visited))  # Remove duplicates from multi-vehicle
            print(f"Instance: {sample['key']}")
            print(f"Nodes: {N}")
            print(f"Clusters: {m}")
    
            print(f"Objective: {result['objective']}")
            print(f"Runtime: {runtime}")
            print(f"Gap: {result['gap']*100:.2f}%")
            print(f"Gap-Nominal: {result['abs_gap']:.2f}")



        # Build adjacency from (cluster,node) -> (cluster,node)
        adjacency = {}
        for (i, j, k, l) in selected_edges:
            if (i, j) in adjacency:
                print(f"[Warning] Node ({i},{j}) has multiple outgoing edges in solution")
            adjacency[(i, j)] = (k, l)

        # Find starting node (any)
        start = list(adjacency.keys())[0]

        # Traverse the tour with cycle-guard to prevent infinite loops
        tour = []
        current = start
        visited = set()
        while current not in visited:
            visited.add(current)
            tour.append(current)
            current = adjacency.get(current)
            if current is None:
                break

        # Build route and cluster sequences, closing the tour back to start
        out_route = {0: [node[1] for node in tour] + [tour[0][1]]}
        out_clusters = {0: [node[0] for node in tour] + [tour[0][0]]}

        plot_ctsp_result(
            all_dict, out_route, out_clusters, clusters, result['objective'],
            show=False,
            figsize=None,
            input_title=f"{sample['key']}",
            save_path=f"./figures/{sample["key"]+"_" +f"{N}"+"_"+f"{m}"}_extension",    
        )
        report_list.append({"key":sample['key'],
                            "obj":result["objective"],
                            "runtime":result["runtime"],
                            "gap":result["gap"],
                            "abs_gap":result["abs_gap"]
                            })
        return report_list
    except:
        pass

