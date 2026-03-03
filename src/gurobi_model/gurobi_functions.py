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

def complete_routes_of_nodes(input_data):
    #input_data=[0,4]
    last_arc=[]
    if not input_data:
        return input_data  # Return empty list if data is empty
    # Extract the starting (cluster_1, city_1) and ending (cluster_2, city_2) from the first and last tuples
    start_city = input_data[0]
    end_city = input_data[-1]
    # Add the closing edge (end_cluster, end_city) to (start_cluster, start_city)
    last_arc.append((end_city,start_city))    
    return last_arc

def get_perm_of_subtours(input_subtour):
    # input_subtour = [0, 3, 5]
    unique_slices = set()
    for i in range(len(input_subtour)-1):
        unique_slices.add((input_subtour[i],input_subtour[i+1]))
        unique_slices.update(complete_routes_of_nodes(input_subtour))
    final_combinations = list(unique_slices)  # Convert set to list for indexing
    return final_combinations

def extract_subtours(input_selected_edges):
    """
    input_selected_edges = selected_edges_tl
    input_unique_route=list(range(C+1))
    input_active_vehicles_=active_vehicles_
    """
    # Separate edges by vehicle
    vehicle_edges = {}
    for i,j,v in input_selected_edges:
        vehicle_edges.setdefault(v, []).append((i, j))

    subtours_per_vehicle = {}

    for v, e_list in vehicle_edges.items():
        G = nx.DiGraph()
        G.add_edges_from(e_list)

        # Extract all simple cycles (subtours)
        cycles = list(nx.simple_cycles(G))

        subtours_per_vehicle[v] = cycles
    return subtours_per_vehicle


def subtourelim(model, where):
    if where != GRB.Callback.MIPSOL:
        return

    try:
        # Get solution values
        vals = model.cbGetSolution(model._vars)

        # Extract selected arcs: (v, i, j)
        selected_edges = gp.tuplelist((i,j,v) for (i,j,v) in model._vars.keys() if vals[i,j,v] > 0.5)
        # selected_edges = gp.tuplelist((i,j,v) for (i,j,v), var_ in model._vars.items() if var_.x > 0.5)

        if not selected_edges:
            return

        # Compute subtours per vehicle
        tours = extract_subtours(selected_edges)

        # Vehicles with >1 subtour → need constraints
        violating_vehicles = [v for v, subtour_list in tours.items() if len(subtour_list) > 1]

        # ⛔ EARLY EXIT: If no subtours → stop immediately
        if not violating_vehicles:
            return

        # Add lazy constraints for all violating vehicles
        for v in violating_vehicles:
            for r, subtour in enumerate(tours[v]):

                # Get arcs inside this subtour
                subtour_edges = get_perm_of_subtours(subtour)

                # Lazy constraint: eliminate this subtour
                lhs = gp.quicksum(model._vars[i,j,v] for (i, j) in subtour_edges)
                rhs = len(subtour_edges) - 1
                model.cbLazy(lhs <= rhs)
                #model.addConstr(lhs <= rhs)

                #print(f"[Lazy] Vehicle={v} | Subtour={r} | Nodes={len(subtour)} | " f"Edges={len(subtour_edges)} | RHS={rhs}")
    except Exception as e:
        print(f"[Callback Error]: {e}")