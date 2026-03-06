import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle

import sys
import os

from pathlib import Path,PurePosixPath
import re
import random

RANDOM_STATE=502
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from data_utils import get_tsp_coords

sys.path.insert(0, str(Path(__file__).parent.parent / "visualisations"))
from plot_instance import plot_instance_simple
# Efficient way to grab just the coordinates
# Skip the first column (index) and take columns 1 and 2

# Get the directory where the script lives
base_dir = Path(__file__).resolve().parent
# Navigate and define the pattern
search_pattern = base_dir / f"../../data/tsplib/*.tsp"

# Use .glob() on the path object
path_list = list(search_pattern.parent.glob(search_pattern.name))
instance_names = [path.stem for path in path_list]

filtered_instances = []
article_instances = [
    "att48", "eil51", "st70", "eil76", "pr76", 
    "kroA100", "rat99", "rd100", "eil101", "lin105", "pr107", 
    "pr124", "bier127", "pr136", "pr144", "kroA150", 
    "pr152", "u159", "rat195", "d198", "kroA200", "kroB200", 
    "ts225", "pr226", "gil262", "pr264", "pr299", "lin318", 
    "rd400", "fl417", "pr439"
]

c_values = [
    10, 11, 14, 16, 16, 20, 20, 20, 21, 21, 22, 25, 26, 
    28, 29, 30, 31, 32, 39, 40, 40, 40, 45, 46, 53, 53, 60, 64, 
    80, 84, 88
]
for idx,path in enumerate(path_list):
    name = path.stem  # e.g., 'ulysses16'
    if name in article_instances:
        match = re.search(r'(\d+)', name)
        node_count = int(match.group(1))
        filtered_instances.append({
                'name': name,
                'nodes': node_count,
                #'cluster':c_values[]
                'path': path
            })
    #else: 
    #    print(name)
        
filtred_keys = [filtered_instances[key_idx]["name"] for key_idx in range(len(filtered_instances))]

for key in article_instances:
    #name = article_instances[key_idx]["name"]
    if key not in filtred_keys:
        print(key,"not found")

filtered_instances.sort(key=lambda x: (x['nodes'], x['name']))


instances=[]
for inst_idx in range(len(filtered_instances)):
    # inst_idx = 0
    
    instance_name = filtered_instances[inst_idx]["name"]
    coords = get_tsp_coords(instance_name)
    if coords is not None:
        C = c_values[inst_idx]
        kmeans = KMeans(n_init=10, n_clusters=C, random_state=RANDOM_STATE)
        clusters = kmeans.fit_predict(coords)
        clusters = [int(v)+1 for v in clusters]
        instances.append({"key":instance_name,
                         "x_coordinates":coords[:,:1].flatten(),
                         "y_coordinates":coords[:,1:2].flatten(),
                         "cluster_assignments":clusters,
                         "n_clusters":C})
        
    else:
        pass

instance_path="../../data/tsplib_instances.pkl"
pickle.dump(instances, open(instance_path, "wb"))

#plot_instance_simple(instances[30])