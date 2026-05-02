import os
import random
import math

# from tqdm import tqdm
# Third-party imports
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from datetime import date,datetime
from tqdm import tqdm
import pickle
#from docplex.mp.model import Model  # Moved to the top
import gurobipy as gp
from gurobipy import GRB, quicksum, tuplelist
from collections import defaultdict
from itertools import product, permutations,combinations
import seaborn as sns

import sys
import os
from pathlib import Path,PurePosixPath
import importlib.util
import json

def extract_subtours(input_selected_edges):
    active_edges = [((i, j), (k, l)) for (i, j, k, l) in input_selected_edges]
    G = nx.DiGraph()  # Use directed graph to preserve arc direction
    G.add_edges_from(active_edges)
    subtours = [sorted(list(c)) for c in nx.weakly_connected_components(G)]
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
    FIX: was incorrectly iterating over dict keys with enumerate(nodes),
        now correctly uses (original_node, indexed_node) pairs.
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
    all_dict = get_all_dict(coords, clusters_to_nodes)  # FIX: pass clusters_to_nodes, not clusters
    all_dict = {k: all_dict[k] for k in sorted(all_dict)}
    return N, C, clusters, clusters_to_nodes, all_dict


# ─────────────────────────────────────────────
# Lazy constraint callback
# ─────────────────────────────────────────────
def subtourelim(model, where):
    if where != GRB.Callback.MIPSOL:
        return

    try:
        vals_x = model.cbGetSolution(model._x)

        selected_edges = gp.tuplelist(
            (i, j, k, l)
            for (i, j, k, l) in model._x.keys()
            if vals_x[i, j, k, l] > 0.5
        )

        #model.optimize()
        #vals_x = model.getAttr('x', x)
        #selected_edges = [(i,j,k,l) for (i,j,k,l) in cross_cluster_nodes_set if vals_x[i,j,k,l] > 0.5]


        tours = extract_subtours(selected_edges)

        for subtour in tours:
            # subtour = tours[2]
            clusters_in_subtour = set(node[0] for node in subtour)

            if len(clusters_in_subtour) == len(model._M):
                continue

            subtour_node_set = set(subtour)  # e.g. {(1,3), (8,2), (9,2)}

            # Filter original selected edges — preserves exact arc direction
            subtour_edges = [
                (i, j, k, l)
                for (i, j, k, l) in selected_edges
                if (i, j) in subtour_node_set and (k, l) in subtour_node_set
            ]

            if not subtour_edges:
                continue

            lhs_x = gp.quicksum(
                model._x[i, j, k, l]
                for (i, j, k, l) in subtour_edges
                if (i, j, k, l) in model._x
            )

            #model.cbLazy(lhs_x - lhs_y <= -1)
            #rhs = len(subtour_edges) - 1
            
            model.cbLazy(lhs_x <= len(subtour_edges) - 1)
            #model.addConstr(lhs_x  <= len(subtour_edges) - 1)
            #model.addConstr(lhs <= rhs)
            #print(lhs_x,"<=",len(subtour_edges),"- 1")
            #print(lhs_x, "-", lhs_y, "<= -1")
            

    except Exception as e:
        print(f"[Callback Error]: {e}")
