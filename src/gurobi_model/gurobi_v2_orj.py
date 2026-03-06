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
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent / "visualisations"))
from plot_route_result import plot_ctsp_result

report_list=[]
instances_main = pickle.load(open(f"../../data/tsplib_instances.pkl", "rb"))

for sample_idx in tqdm(range(len(instances_main))):
    try:
        # sample_idx = 3
        sample = instances_main[sample_idx]

        def get_clusters(labels_inputs,C):
            
            cluster_mapping = {i + 1: [] for i in range(C)}  # Renamed from clusters to cluster_mapping
            for customer, cluster_label in enumerate(labels_inputs, start=1):
                cluster_mapping[cluster_label].append(customer)
            #cluster_mapping.update({0:[0]})
            return cluster_mapping  # Return the renamed variable

        def get_clustered_nodes(input_clusters,C):
            """
            input_clusters = .copy()
            
            """
            node_mapping = {}  # Renamed from clustered_nodes to node_mapping
            for cluster_id in range(1, C + 1):
                cluster_nodes = input_clusters[cluster_id]
                cluster_size = len(cluster_nodes)
                
                # Re-map the node indices to [0, len(cluster)-1]
                updated_cluster_nodes = list(range(1, cluster_size + 1))
                node_mapping[cluster_id] = dict(zip(cluster_nodes, updated_cluster_nodes))

            #node_mapping.update({0: {0: 0}})

            return node_mapping  # Return the renamed variable


        def get_all_dict(input_coordinates, input_clustered_nodes):
            """
            input_coordinates, input_clustered_nodes = coords, clusters_to_nodes
            """
            new_dict = {}  # Renamed from all_dict to new_dict
            
            # Loop through each cluster and assign cluster information to each node's coordinates
            for cluster_id, nodes in input_clustered_nodes.items():
                # cluster_id, nodes = 1,input_clustered_nodes[1]

                

                for idx,node in enumerate(nodes):
                    new_dict[node] = {
                        'coordinates': [int(i) for i in input_coordinates[node-1]],
                        'cluster': cluster_id,
                        # Depot (cluster 0) uses indexed_node=0, customer clusters use 1-based indexing
                        'indexed_node': idx if cluster_id == 0 else idx + 1,
                    }
            return new_dict  # Return the renamed variable

        def compute_distance_matrix(coord_i, coord_j):
            # coord_i, coord_j = coord_a, coord_b
            dist = round(float(np.sqrt((coord_i[0] - coord_j[0])**2 + (coord_i[1] - coord_j[1])**2)),2)
            return dist


        def _prepare_instance(inst):
            """Prepare instance data for GA."""
            N = len(inst['x_coordinates'])
            C = inst['n_clusters']
            coords = np.column_stack((inst['x_coordinates'], inst['y_coordinates']))
            #len(coords)
            clusters = get_clusters(np.array(inst['cluster_assignments']), C)
            clusters_to_nodes = get_clustered_nodes(clusters, C)
            all_dict = get_all_dict(coords, clusters)
            all_dict = {k: all_dict[k] for k in sorted(all_dict)}
            return N, C, clusters, clusters_to_nodes, all_dict

        N, m, clusters, clusters_to_nodes, all_dict = _prepare_instance(sample)

        M = list(range(1,m+1)) # set of clusters

        n_i = dict(Counter(sample["cluster_assignments"]))
        N = sum(n_i.values()) # set of nodes

        # ----------------------------
        # Model
        # Create Gurobi model
        start_time = datetime.now()
        env = gp.Env(empty=True)
        env.setParam('LogToConsole', 0)
        env.start()

        # ----------------------------
        model = gp.Model("DS502_Project",env=env)
        #model.setParam('OutputFlag', 0)
        model.Params.TimeLimit = 60

        model.Params.Threads = 8


        # OPTIMIZATION 3: More efficient MIP settings
        #model.setParam('MIPGap', 0.05) # Stop when within 5% of the optimum
        #model.Params.MIPFocus = 1  # Focus on finding good feasible solutions
        #model.Params.Heuristics = 0.5  # Moderate heuristics
        #model.Params.NoRelaxedMIPHeuristic = 0  # Enable relaxation-based heuristic

        #model.Params.Cuts = 2  # Aggressive cut generation
        #model.NumStart = 1  # Number of warm starts
        #model.Params.StartNumber = 0  # Use warm start

        cluster_node_set = {i: list(range(1, n_i[i] + 1)) for i in M}
        cluster_node_set_ = {(i, j): 0 for i in M for j in range(1, n_i[i] + 1)}
        cross_cluster_nodes_set = {(i,j,k,l): 0 for i in M for j in range(1, n_i[i] + 1) for k in M if i!=k for l in range(1, n_i[k] + 1)  }
        cross_cluster_set = {(i, j): 0 for i in M for j in M if i!=j}

        # 2. Build the Cost Matrix (Dictionary format for Gurobi)
        cost_matrix = {}
        for n1 in all_dict.keys():
            for n2 in all_dict.keys():
                i,j,k,l = all_dict[n1]["cluster"],all_dict[n1]["indexed_node"],all_dict[n2]["cluster"],all_dict[n2]["indexed_node"]
                if n1==n2:
                    cost_matrix[i, j, k, l] = 0  # Distance to self
                else:
                    coord_a = all_dict[n1]["coordinates"]
                    coord_b = all_dict[n2]["coordinates"]
                    cost_matrix[i, j, k, l] = compute_distance_matrix(coord_a, coord_b)
                    

        # Decision Variables
        y = model.addVars(cluster_node_set_.keys(), vtype=GRB.BINARY, name="y")
        x = model.addVars(cross_cluster_nodes_set.keys(), vtype=GRB.BINARY, name="x")
        z = model.addVars(cross_cluster_set.keys(), vtype=GRB.BINARY, name="z")
        #p = model.addVars(_passenger_index, vtype=GRB.CONTINUOUS, lb=0.0, name="p")

        #! Objective MIN-Z
        model.setObjective(gp.quicksum(cost_matrix[i,j,k,l] * x[i,j,k,l]
                                    for i in M
                                    for j in cluster_node_set[i] 
                                    for k in M
                                    for l in cluster_node_set[k] 
                                    if i!=k
                                    ), GRB.MINIMIZE)

        #! Constraint 1
        for i in M:
            model.addConstr(gp.quicksum(y[i,j] 
                                        for j in cluster_node_set[i]) == 1, name=f"one_node_from_cluster_{i}")
        #! Constraint 2
        for i in M:
            for j in cluster_node_set[i]:
                model.addConstr(gp.quicksum(x[i,j,k,l]
                                            for k in M if i!=k
                                            for l in cluster_node_set[k] 
                                            ) == y[i,j])
        #! Constraint 3
        for k in M:
            for l in cluster_node_set[k]:
                model.addConstr(gp.quicksum(x[i,j,k,l] 
                                            for i in M 
                                            for j in cluster_node_set[i] 
                                            if i!=k) == y[k,l])
                
        #! Constraint 4
        for i in M:
            for k in M:
                if i!=k:
                    model.addConstr(gp.quicksum(x[i,j,k,l] 
                                                for j in cluster_node_set[i]
                                                for l in cluster_node_set[k]
                                                ) == z[i,k])

        #! Constraint 5
        for size in tqdm(range(1, len(M))):
            # size=16
            for s in combinations(M, size):
                s_set = set(s)
                not_s  = [k for k in M if k not in s_set]
                model.addConstr(
                    gp.quicksum(z[i, k] for i in s_set for k in not_s) >= 1,
                    name=f"C5_subtour_{'_'.join(map(str, s))}"
                )

        model.write(f"../../reports/lp_models/ds502_project_{m}{sample['key']}.lp")  # Human-readable LP format
        #with open(f"ds502_project.lp", "r") as f:
        #    lp_text = f.read()
        #    print(lp_text)

        
        model._x = x
        model._y = y
        model._z = z

        model._vars = x    
        model.optimize()

        #model.Params.LazyConstraints = 1

        #model.optimize(subtourelim)

        end_time = datetime.now()
        runtime = f"{(end_time - start_time).total_seconds():.4f}"   
            
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
            selected_edges = [(i,j,k,l) for (i,j,k,l) in cross_cluster_nodes_set.keys() if vals_x[i,j,k,l] > 0.5]

            # Extract visited nodes
            vals_y = model.getAttr('x', y)
            cluster_node_inflow = [(i,j) for (i,j,k,l) in cross_cluster_nodes_set.keys() if vals_x[i,j,k,l] > 0.5]
            cluster_node_outflow = [(k,l) for (i,j,k,l) in cross_cluster_nodes_set.keys() if vals_x[i,j,k,l] > 0.5]
            #visited = [(i,j) for (i,j) in cluster_node_set_ if vals_y[i,j] > 0.5]
            #visited = [(i,j) for (i,j) in cluster_node_set_ if vals_y[i,j] > 0.5]
            result['cluster_node_inflow'] = list(set(cluster_node_inflow))  # Remove duplicates from multi-vehicle
            result['cluster_node_outflow'] = list(set(cluster_node_outflow))  # Remove duplicates from multi-vehicle

            
            
            vals_z = model.getAttr('x', z)
            visited = [(i,k) for (i,k) in cross_cluster_set.keys() if vals_z[i,k] > 0.5]
            
            result['visited_clusters'] = list(set(visited))  # Remove duplicates from multi-vehicle

            print(f"Objective: {result['objective']}")
            print(f"Runtime: {runtime}")
            print(f"Gap: {result['gap']*100:.2f}%")
            print(f"Gap-Nominal: {result['abs_gap']:.2f}")



        # Build adjacency from (cluster,node) -> (cluster,node)
        adjacency = {(i,j): (k,l) for (i,j,k,l) in selected_edges}

        # Find starting node (any)
        start = list(adjacency.keys())[0]

        # Traverse the tour
        tour = []
        current = start
        while True:
            tour.append(current)
            current = adjacency[current]
            if current == start:
                break
        # Print results
        out_route = {}
        out_clusters = {}
        for v in range(len([tour])):
            out_route[v] = [node[1] for node in tour]
            #out_route[v].append(tour[v][1])
            
            out_clusters[v] = [node[0] for node in tour]
            #out_clusters[v].append(tour[v][0])

        plot_ctsp_result(all_dict, out_route, out_clusters, clusters, result['objective'],
                        show=True,
                        save_path=f"../../figures/{m}{sample["key"]+"_" +f"{N}"+"_"+f"{m}"}",
                        figsize=None,
                        input_title=f"{sample['key']}"
                    #save_path=f"../../../04-reports/figures/{N}_{C+1}_{K}_ga_est_concorde_{PARAM_MODEL_TYPE}_{PATH_K}_{benchmark_param}.png"
                    )

        report_list.append({"key":sample['key'],
                            "obj":result["objective"],
                            "runtime":result["runtime"],
                            "gap":result["gap"],
                            "abs_gap":result["abs_gap"]
                            })
    except (ValueError, TypeError, Exception) as e:
        print(f"Error: {e}")    
pd.DataFrame(report_list).to_excel("../../reports/results.xlsx")