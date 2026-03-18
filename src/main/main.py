# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle
import sys
import os
import re
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

#sys.path.insert(0, str(Path(__file__).parent / "data_prep"))
from data_prep import build_instances, get_filtered_instances, check_missing, save_instances
from model import run_gurobi_model_for_instance
from report import plot_outputs


def main():
    base_dir = Path(__file__).resolve().parent

    filtered_instances = get_filtered_instances(base_dir)
    check_missing(filtered_instances)
    instances = build_instances(filtered_instances)
    save_instances(instances, base_dir / "./data/tsplib_instances.pkl")
    
    report_list = []
    
    for idx in tqdm(range(len(instances))[:]):
        #run_gurobi_model_for_instance(instances,0)
        report_idx=run_gurobi_model_for_instance(instances,idx)
        report_list.extend(report_idx)
#    pd.DataFrame(report_list).to_excel("../../reports/results_extension_TSP.xlsx")
    report_df = pd.DataFrame(report_list)

    n_values = [k["x_coordinates"].shape[0] for k in instances]
    c_values = [max(k["cluster_assignments"]) for k in instances]
    ins_names = [k["key"] for k in instances]


    for idx in tqdm(range(len(report_df))):
        if report_df.loc[idx,"key"] in ins_names:
            report_df.loc[idx,"N"] = int(n_values[ins_names.index(report_df.loc[idx,"key"])])
            report_df.loc[idx,"C"] = int(c_values[ins_names.index(report_df.loc[idx,"key"])])

    report_df[["N","C"]]=report_df[["N","C"]].astype(int)
    report_df[["runtime","gap","abs_gap"]]=report_df[["runtime","gap","abs_gap"]].round(2)
    report_df.to_excel("./reports/results_extension_TSP.xlsx")
    
    # ── Data from dataframe ───────────────────────────────────────────────────────
    x_keys = report_df['key'] + "-" + report_df['N'].astype(str) + '-' + report_df['C'].astype(str)
    instances = report_df['key'].tolist()
    obj       = report_df['obj'].tolist()
    runtime   = report_df['runtime'].tolist()
    gap       = report_df['gap'].tolist()
    abs_gap   = report_df['abs_gap'].tolist()
    N         = report_df['N'].tolist()
    C         = report_df['C'].tolist()

    n = len(instances)
    x = np.arange(len(x_keys))
    
    plot_outputs(x,x_keys, obj,runtime,gap,abs_gap,N,C)


if __name__ == "__main__":
    main()