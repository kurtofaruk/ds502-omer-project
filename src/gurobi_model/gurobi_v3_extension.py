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

report_list=[]
instances_main = pickle.load(open(f"../../data/tsplib_instances.pkl", "rb"))
 

for sample_idx in tqdm(range(len(instances_main))):
    try:
        # sample_idx = 10
        sample = instances_main[sample_idx]

        def extract_subtours(input_selected_edges):
            """
            input_selected_edges = selected_edges
            input_unique_route=list(range(C+1))
            input_active_vehicles_=active_vehicles_
            """
            active_edges = [((i, j), (k, l)) for (i, j, k, l) in input_selected_edges]
            # 2. Build the graph
            G = nx.Graph() # Use nx.DiGraph() if your TSP is directed
            G.add_edges_from(active_edges)
            # 3. Find subtours
            # This returns a list of sets: [{(i, j), (k, l), ...}, {...}]
            #subtours = list(nx.connected_components(G))
            subtours = [sorted(list(c)) for c in nx.connected_components(G)]
            return subtours

        def subtourelim(model, where):
            if where != GRB.Callback.MIPSOL:
                return

            try:
                # Get solution values
        #        vals = model.cbGetSolution(model._vars)
                # In callback:
                # Before optimize
                model._x = x
                #model._y = y

                # Inside subtourelim callback
                vals_x = model.cbGetSolution(model._x)
                # vals_y = model.cbGetSolution(model._y)

                selected_edges = gp.tuplelist((i,j,k,l) for (i,j,k,l) in model._x.keys() if vals_x[i,j,k,l] > 0.5)
                # inflow_clusters = gp.tuplelist((i,j) for (i,j,k,l) in model._x.keys() if vals_x[i,j,k,l] > 0.5)
                # outflow_clusters = gp.tuplelist((k,l) for (i,j,k,l) in model._x.keys() if vals_x[i,j,k,l] > 0.5)
                #selected_clusters = gp.tuplelist((i,j) for (i,j) in model._y.keys() if vals_y[i,j] > 0.5)
                
                # Extract selected arcs: (v, i, j)
                #selected_edges = gp.tuplelist((i,j,k,l) for (i,j,k,l) in model._vars.keys() if vals_x[i,j,k,l] > 0.5)
                #selected_edges = gp.tuplelist((i,j,k,l) for (i,j,k,l) in model._y.keys() if vals_y[i,k] > 0.5)
                
                #selected_edges = gp.tuplelist((i,j,k,l) for (i,j,k,l), var_ in model._vars.items() if var_.x > 0.5)
                #selected_clusters = gp.tuplelist((i,k) for (i,j,k,l), var_ in model._vars.items() if var_.x > 0.5)

                #if not selected_edges:
                #    return

                # Compute subtours per vehicle
                tours = extract_subtours(selected_edges)
                
                #print(tours)
                # Add lazy constraints for all violating vehicles
                for subtour in tours:
                    #print(subtour)
                    clusters_in_subtour = set(node[0] for node in subtour)
                    if len(clusters_in_subtour) == len(M):
                        #print(f"[Skip] Full tour found with {len(M)} clusters")
                        break
                    # print(subtour)
                    # subtour = tours[0]
                    # subtour = list((i for i in tours[0]))
                    # subtour_edges = get_perm_of_subtours(list(tours[0]))
                    # Get arcs inside this subtour
                    # subtour_edges = get_perm_of_subtours(subtour)
                    subtour_edges = get_perm_of_subtours(subtour)
                    subtour_nodes = set([(i[0],i[1]) for i in subtour_edges])
                    subtour_clusters = [(i[0],i[2]) for i in subtour_edges]
                    
                    if not subtour_edges:
                        continue

                    # Lazy constraint: eliminate this subtour
                    lhs = gp.quicksum(
                        model._x[i,j,k,l]
                        for (i,j,k,l) in subtour_edges
                        if (i,j,k,l) in model._x
                    )
                    
                    lhs_y = gp.quicksum(
                        model._y[i,j]
                        for (i,j) in subtour_nodes
                        if (i,j) in model._y
                    )
                    
                    
                    lhs_z = gp.quicksum(
                        model._z[i,k]
                        for (i,k) in subtour_clusters
                        if (i,k) in model._z
                    )
                    
                    #lhs = gp.quicksum(model._x[i,j,k,l] for (i,j,k,l) in subtour_edges)
                    #rhs_1 = gp.quicksum(model._y[i,j] for (i,j,k,l) in subtour_edges)
                    #rhs_2 = gp.quicksum(model._y[k,l] for (i,j,k,l) in subtour_edges)
                    rhs = len(subtour_edges) - 1
                    
                    #model.cbLazy(lhs <= rhs)
                    model.cbLazy(lhs <= rhs)
                    model.cbLazy(lhs_y <= rhs)
                    model.cbLazy(lhs_z <= rhs)
                    
                    #model.addConstr(lhs <= rhs)
                    #model.addConstr(lhs_y <= rhs)
                    #model.addConstr(lhs_z <= rhs)
                    """
                    print(lhs,"<=",rhs )
                    print(lhs_y,"<=",rhs )
                    print(lhs_z,"<=",rhs )
                    """
                #model.optimize()
                    # print(f"[Lazy]| Nodes={len(subtour)} | " f"Edges={len(subtour_edges)} | RHS={rhs}")
            except Exception as e:
                print(f"[Callback Error]: {e}")
                print(lhs,"<=",rhs )

        def complete_routes_of_nodes(input_data):
            #input_data=[0,4]
            # input_data = input_subtour.copy()
            last_arc=[]
            if not input_data:
                return input_data  # Return empty list if data is empty
            # Extract the starting (cluster_1, city_1) and ending (cluster_2, city_2) from the first and last tuples
            start_city = input_data[0]
            end_city = input_data[-1]
            # Add the closing edge (end_cluster, end_city) to (start_cluster, start_city)
            last_arc.extend(end_city)    
            last_arc.extend(start_city)
            return last_arc

        """
        def get_perm_of_subtours(input_subtour):
            # input_subtour = subtour
            unique_slices = set()
            for i in range(len(input_subtour)-1):
                # i=0
                # input_subtour[0]
                tmp_subtour=list(input_subtour[i])
                tmp_subtour.extend(input_subtour[i+1])
                unique_slices.add(tuple(tmp_subtour))
                unique_slices.add(tuple(complete_routes_of_nodes(input_subtour)))
            final_combinations = list(unique_slices)  # Convert set to list for indexing
            return final_combinations
        """
        def get_perm_of_subtours(input_subtour):
            # input_subtour = list(subtour)
            unique_slices = set()
            n = len(input_subtour)
            for i in range(n):
                # i=0
                node_a = input_subtour[i]
                node_b = input_subtour[(i + 1) % n]  # wraps around correctly
                fwd = tuple(list(node_a) + list(node_b))
                #bwd = tuple(list(node_b) + list(node_a))
                unique_slices.add(fwd)
                #unique_slices.add(bwd)
            return list(unique_slices)

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
        cross_cluster_nodes_set = {(i,j,k,l): 0 for i in M for j in range(1, n_i[i] + 1) for k in M for l in range(1, n_i[k] + 1) if i!=k }
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
                                            for k in M 
                                            for l in cluster_node_set[k] 
                                            if i!=k) == y[i,j])
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

        #! Constraint 4
        for i in M:
            model.addConstr(gp.quicksum(z[i,k] for k in M  if i!=k) >= 1)

        #! Constraint 5
        model.addConstr(gp.quicksum(z[i,k] for k in M  for i in M if i!=k) == m)


        model.write(f"ds502_project_{sample["key"]}.lp")  # Human-readable LP format
        #with open(f"ds502_project.lp", "r") as f:
        #    lp_text = f.read()
        #    print(lp_text)

        
        model._x = x
        model._y = y
        model._z = z

        model._vars = x    
        #model.optimize()

        model.Params.LazyConstraints = 1

        model.optimize(subtourelim)

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
                        save_path=f"../../figures/{sample["key"]+"_" +f"{N}"+"_"+f"{m}"}",
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
    except:
        pass

pd.DataFrame(report_list).to_excel("../../reports/results.xlsx")