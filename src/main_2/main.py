import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

from data_prep import build_instances, get_filtered_instances, check_missing, save_instances
from model import run_gurobi_model_for_instance
from mdp import run_mdp_instance
from genetic_algorithm import run_ga_instance
from report import plot_outputs, plot_ga_report


MILP_TIME_LIMIT = 120   # seconds per instance for MILP
MILP_THREAD_LIMIT = 4   # seconds per instance for MILP

RUN_MDP=False
RUN_GA=True
GA_CLUSTER_LIMIT = 21   # skip GA on instances with more clusters than this

def main():
    base_dir    = Path(__file__).resolve().parent
    figures_dir = str(base_dir / "figures")
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Build / load instances ─────────────────────────────────────────────
    print("\n── Building instances ────────────────────────────────────────────")
    filtered_instances = get_filtered_instances(base_dir)
    check_missing(filtered_instances)
    instances = build_instances(filtered_instances)
    save_instances(instances, base_dir / "data/tsplib_instances.pkl")

    # ── 2. MILP ───────────────────────────────────────────────────────────────
    print("\n── Running MILP ──────────────────────────────────────────────────")
    report_list = []
    for idx in tqdm(range(len(instances))):
        try:
            # idx=0
            report_idx = run_gurobi_model_for_instance(
                instances, idx, input_time_limit=MILP_TIME_LIMIT, n_thread=MILP_THREAD_LIMIT)
            report_list.extend(report_idx)
        except Exception as e:
            print(f"[ERROR] MILP instance {idx}: {e}")

    report_df = pd.DataFrame(report_list)

    n_values  = [k["x_coordinates"].shape[0] for k in instances]
    c_values  = [max(k["cluster_assignments"]) for k in instances]
    ins_names = [k["key"] for k in instances]

    for idx in report_df.index:
        key = report_df.loc[idx, "key"]
        if key in ins_names:
            pos = ins_names.index(key)
            report_df.loc[idx, "N"] = int(n_values[pos])
            report_df.loc[idx, "C"] = int(c_values[pos])

    report_df[["N", "C"]] = report_df[["N", "C"]].astype(int)
    report_df[["runtime", "gap", "abs_gap"]] = report_df[["runtime", "gap", "abs_gap"]].round(2)
    report_df.to_excel(reports_dir / "results_extension_TSP.xlsx")
    print(f"Saved MILP results → {reports_dir / 'results_extension_TSP.xlsx'}")

    x_keys  = report_df['key'] + "-" + report_df['N'].astype(str) + '-' + report_df['C'].astype(str)
    x       = np.arange(len(x_keys))
    plot_outputs(x, x_keys,
                 report_df['obj'].tolist(), report_df['runtime'].tolist(),
                 report_df['gap'].tolist(), report_df['abs_gap'].tolist(),
                 report_df['N'].tolist(),   report_df['C'].tolist(),
                 figures_dir=figures_dir)

    # ── 3. MDP (greedy + exact DP for small instances) ────────────────────────
    if RUN_MDP:
        print("\n── Running MDP ───────────────────────────────────────────────────")
        mdp_results = []
        for sample in tqdm(instances):
            # sample  = instances[idx]
            try:
                r = run_mdp_instance(sample)
                tag = f"greedy={r['greedy_cost']:,.0f}"
                if r['dp_cost'] is not None:
                    tag += f"  dp={r['dp_cost']:,.0f}"
                print(f"  {r['key']:12s}  N={r['N']:4d}  C={r['C']:3d}  {tag}")
                mdp_results.append(r)
            except Exception as e:
                print(f"[ERROR] MDP {sample['key']}: {e}")

        mdp_df = pd.DataFrame(mdp_results).sort_values("C")
        mdp_df.to_excel(reports_dir / "results_mdp.xlsx", index=False)
        print(f"Saved MDP results → {reports_dir / 'results_mdp.xlsx'}")

    # ── 4. Genetic Algorithm ──────────────────────────────────────────────────
    if RUN_GA:
        print("\n── Running Genetic Algorithm ─────────────────────────────────────")
        ga_results = []
        for sample in tqdm(instances):
            if sample["n_clusters"] > GA_CLUSTER_LIMIT:
                print(f"  Skipping {sample['key']} (C={sample['n_clusters']} > {GA_CLUSTER_LIMIT})")
            continue
        try:
            r = run_ga_instance(
                sample, pop_size=100, n_generations=100,
                time_limit_eval=20.0, total_time_limit=120.0,
                max_stall_generations=20, verbose=True,
            )
            print(f"  {r['key']:12s}  N={r['N']:4d}  C={r['C']:3d}  "
                  f"ga_cost={r['ga_cost']:,.0f}  time={r['ga_time']:.1f}s  "
                  f"gens={r['generations']}  cache={r['cache_hit_rate']:.0f}%")
            ga_results.append(r)
        except Exception as e:
            print(f"[ERROR] GA {sample['key']}: {e}")

        ga_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("ga_tour", "obj_list")} for r in ga_results])
        ga_df.to_excel(reports_dir / "results_ga.xlsx", index=False)
        print(f"Saved GA results → {reports_dir / 'results_ga.xlsx'}")

        # ── 5. Comparison plots ───────────────────────────────────────────────────
        print("\n── Generating comparison plots ───────────────────────────────────")
        plot_ga_report(report_df, ga_df, figures_dir=figures_dir)
    print("Done.")
        

if __name__ == "__main__":
    main()
