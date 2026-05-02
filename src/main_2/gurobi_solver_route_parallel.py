import pandas as pd
import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
import os


def compute_distance_matrix(coords):
    """Compute pairwise distance matrix using vectorized operations."""
    coords = np.array(coords)
    n = len(coords)

    # Vectorized distance computation (MUCH faster than loops)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))

    return np.round(dist_matrix, 2)


def subtourelim(model, where):
    """
    Optimized subtour elimination callback.
    """
    if where == GRB.Callback.MIPSOL:
        vals = model.cbGetSolution(model._vars)
        selected_edges = gp.tuplelist(
            (i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5
        )

        n_nodes = model._num_nodes

        # Build graph and find subtours
        G = nx.DiGraph()
        G.add_nodes_from(range(n_nodes))
        G.add_edges_from(selected_edges)

        # Find weakly connected components
        components = list(nx.weakly_connected_components(G))

        # If single component with all nodes, we have a valid tour
        if len(components) == 1 and len(components[0]) == n_nodes:
            return

        # Add lazy constraints for each subtour
        for component in components:
            if len(component) < n_nodes:
                subtour_edges = [
                    (i, j) for i, j in selected_edges
                    if i in component and j in component
                ]
                if subtour_edges:
                    model.cbLazy(
                        gp.quicksum(model._vars[i, j] for i, j in subtour_edges)
                        <= len(component) - 1
                    )


def gurobi_model_tsp_optimized(idx_route, input_coords, n_threads=1, time_limit=None):
    """
    Optimize a single TSP route using Gurobi.

    Args:
        idx_route: Index within input_coords
        input_coords: Coordinates for this route
        n_threads: Number of threads for Gurobi
        time_limit: Time limit in seconds

    Returns:
        (objective_Z, obj_route): Total distance and route order
    """
    sub_coords = input_coords[idx_route]
    len_edges = len(sub_coords)

    # Compute distance matrix
    node_dist_matrix = compute_distance_matrix(sub_coords)

    # Create environment and model
    env = gp.Env(empty=True)
    env.setParam('LogToConsole', 0)
    env.start()

    model = gp.Model("TSP", env=env)
    model.Params.OutputFlag = 0
    model.Params.Threads = n_threads
    model.Params.LazyConstraints = 1

    if time_limit:
        model.Params.TimeLimit = time_limit

    model.Params.MIPFocus = 1
    model.Params.Heuristics = 0.2
    model.Params.Cuts = 2

    # Create variables for edges
    edges = [(i, j) for i in range(len_edges) for j in range(len_edges) if i != j]
    variables = model.addVars(edges, vtype=GRB.BINARY, name="x")

    # Set objective
    model.setObjective(
        gp.quicksum(node_dist_matrix[i, j] * variables[i, j] for i, j in edges),
        GRB.MINIMIZE
    )

    # Constraints: Each node visited exactly once (outgoing)
    for i in range(len_edges):
        model.addConstr(
            gp.quicksum(variables[i, j] for j in range(len_edges) if i != j) == 1,
            name=f"out_{i}"
        )

    # Constraints: Each node visited exactly once (incoming)
    for j in range(len_edges):
        model.addConstr(
            gp.quicksum(variables[i, j] for i in range(len_edges) if i != j) == 1,
            name=f"in_{j}"
        )

    # Total arcs constraint
    model.addConstr(
        gp.quicksum(variables[i, j]
                    for i in range(len_edges)
                    for j in range(len_edges)
                    if i != j) == len_edges,
        name=f"total_arcs_equals_to{len_edges}"
    )

    # Store info for callback and optimize
    model._vars = variables
    model._num_nodes = len_edges
    model.optimize(subtourelim)

    # Extract results
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_Z = round(float(model.ObjVal), 2)

        # Extract route
        solution = model.getAttr('x', variables)
        selected_route = [(i, j) for (i, j) in edges if solution[i, j] > 0.5]

        # Build route
        G = nx.DiGraph()
        G.add_nodes_from(range(len_edges))
        G.add_edges_from(selected_route)

        try:
            obj_route = list(nx.simple_cycles(G))[0]
        except IndexError:
            obj_route = list(range(len_edges))
    else:
        objective_Z = float('inf')
        obj_route = list(range(len_edges))

    # Clean up
    model.dispose()
    env.dispose()

    return objective_Z, obj_route


def call_gurobi_model_route(args):
    """
    Worker function for route-level parallelization.

    Args:
        args: (sample_idx, route_idx, route_coords, n_threads, time_limit)

    Returns:
        (sample_idx, route_idx, objective_Z, obj_route)
    """
    sample_idx, route_idx, route_coords, n_threads, time_limit = args
    # route_coords is already a single route, not a list of routes
    objective_Z, obj_route = gurobi_model_tsp_optimized(0, [route_coords], n_threads=n_threads, time_limit=time_limit)
    return sample_idx, route_idx, objective_Z, obj_route
"""
def generate_route_coords(n_cities, seed=None):
    rng = np.random.default_rng(seed)
    flat = rng.uniform(0, 100, 2 * n_cities)
    return np.column_stack([flat[:n_cities], flat[n_cities:]])
generate_route_coords(10, seed=None)
gurobi_model_tsp_optimized(0, [generate_route_coords(10, seed=None)], n_threads=1, time_limit=10)
"""

def call_mp_gurobi_route_level(input_coordinates, show_gurobi_progress=True, n_threads=1, time_limit=None):
    """
    Route-level parallelization for Gurobi optimization.

    This improves upon sample-level parallelization by distributing work across
    500 routes (100 samples × 5 routes) instead of just 100 samples.
    Better CPU utilization and load balancing.

    Args:
        input_coordinates: List of 100 samples, each containing 5 routes
        show_gurobi_progress: Whether to show progress bar
        n_threads: Threads per Gurobi model (usually 1 for parallelization)
        time_limit: Time limit per route in seconds

    Returns:
        List of 100 tuples: each (objective_Z_all, obj_route_all)
        where:
            - objective_Z_all: sum of objectives for all 5 routes in sample
            - obj_route_all: list of 5 route orders
    """
    available_cpu = max(os.cpu_count() - 2, 1)

    # ===== STEP 1: Flatten to route level =====
    # Create list of (sample_idx, route_idx, route_coords, n_threads, time_limit)
    route_tasks = []
    for sample_idx, sample_routes in enumerate(input_coordinates):
        for route_idx, route_coords in enumerate(sample_routes):
            route_tasks.append((sample_idx, route_idx, route_coords, n_threads, time_limit))

    # ===== STEP 2: Solve all routes in parallel =====
    from multiprocess import Pool
    from tqdm import tqdm

    chunksize = max(1, len(route_tasks) // (available_cpu * 4))
    route_results = []
    with Pool(processes=available_cpu) as p:
        route_results = list(tqdm(
            p.imap(call_gurobi_model_route, route_tasks, chunksize=chunksize),
            total=len(route_tasks),
            desc="Gurobi optimization (route-level)",
            disable=show_gurobi_progress
        ))

    # ===== STEP 3: Reorganize results back to sample level =====
    num_samples = len(input_coordinates)
    sample_route_results = [[] for _ in range(num_samples)]

    # Group route results by sample
    for sample_idx, route_idx, objective_Z, obj_route in route_results:
        sample_route_results[sample_idx].append((route_idx, objective_Z, obj_route))

    # Aggregate and return in order
    final_results = []
    for sample_idx in range(num_samples):
        routes = sorted(sample_route_results[sample_idx], key=lambda x: x[0])  # Sort by route_idx
        objective_Z_all = round(sum(item[1] for item in routes), 2)
        obj_route_all = [item[2] for item in routes]
        final_results.append((objective_Z_all, obj_route_all))

    return final_results


# Keep original function for backward compatibility
def call_mp_gurobi(input_coordinates, show_gurobi_progress=True):
    """
    Original sample-level parallelization (deprecated - use call_mp_gurobi_route_level instead).
    """
    gurobi_3_results = []
    loop_index = range(len(input_coordinates))
    available_cpu = max(os.cpu_count() - 2, 1)

    def call_gurobi_model(args):
        """Worker for sample-level parallelization."""
        idx_sample, input_coordinates_param = args
        index_coords = input_coordinates_param[idx_sample]
        gurobi_result = [gurobi_model_tsp_optimized(i, index_coords, n_threads=1, time_limit=None)
                        for i in range(len(index_coords))]
        objective_Z_all = round(sum(item[0] for item in gurobi_result), 2)
        obj_route_all = [item[1] for item in gurobi_result]
        return objective_Z_all, obj_route_all

    worker_args = [(idx, input_coordinates) for idx in loop_index]

    from multiprocess import Pool
    from tqdm import tqdm
    with Pool(processes=available_cpu) as p:
        gurobi_3_results = list(tqdm(
            p.imap(call_gurobi_model, worker_args),
            total=len(worker_args),
            desc="Gurobi optimization",
            disable=show_gurobi_progress
        ))
        p.close()
        p.join()

    return gurobi_3_results


def tour_cost(node_sel, ctc, n_threads=1, time_limit=20):
    """
    Compute the optimal tour cost through selected nodes using Gurobi TSP.

    Builds a C-node TSP from the selected nodes and solves it with Gurobi,
    returning both the minimum tour distance and the optimal cluster visit order.

    Args:
        node_sel:   dict {cluster_id: node_id} — one node selected per cluster
        ctc:        dict {cluster_id: {node_id: (x, y)}}
        n_threads:  Gurobi threads per solve
        time_limit: per-solve time limit in seconds

    Returns:
        (cost, optimal_cluster_seq):
            cost                — optimal tour distance (float)
            optimal_cluster_seq — cluster IDs in optimal visit order (list)
    """
    clusters = list(node_sel.keys())
    coords = np.array([ctc[cl][node_sel[cl]] for cl in clusters], dtype=float)

    cost, route_indices = gurobi_model_tsp_optimized(
        idx_route=0,
        input_coords=[coords],
        n_threads=n_threads,
        time_limit=time_limit,
    )

    optimal_cluster_seq = [clusters[i] for i in route_indices]
    return cost, optimal_cluster_seq
