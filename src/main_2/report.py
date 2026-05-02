import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

instance_path="../../data/tsplib_instances.pkl"
instances = pickle.load(open(instance_path, "rb"))


def create_report_df(instances):    
    report_df=pd.read_excel("../../src/main/reports/results_extension_TSP.xlsx",index_col=0)


    n_values = [k["x_coordinates"].shape[0] for k in instances]
    c_values = [max(k["cluster_assignments"]) for k in instances]
    ins_names = [k["key"] for k in instances]


    for idx in range(len(report_df)):
        if report_df.loc[idx,"key"] in ins_names:
            report_df.loc[idx,"N"] = int(n_values[ins_names.index(report_df.loc[idx,"key"])])
            report_df.loc[idx,"C"] = int(c_values[ins_names.index(report_df.loc[idx,"key"])])

    report_df[["N","C"]]=report_df[["N","C"]].astype(int)
    report_df[["runtime","gap","abs_gap"]]=report_df[["runtime","gap","abs_gap"]].round(2)
    return report_df

def export_report_df(input_df,input_path=""):
    # input_path = "../../reports/results_extension_TSP.xlsx"
    return input_df.to_excel("./reports/results_extension_TSP.xlsx")

def style_ax(ax,x,x_keys):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(x_keys, rotation=45, ha='right', fontsize=7)

def scatter_style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(linewidth=0.5, alpha=0.4, zorder=0)

def label_outliers(ax, xs, ys, labels, k=3):
    top = sorted(range(len(ys)), key=lambda i: ys[i], reverse=True)[:k]
    for i in top:
        ax.annotate(labels[i], (xs[i], ys[i]),
                    textcoords='offset points', xytext=(6, 4),
                    fontsize=7.5, color='#555')

def plot_outputs(x,x_keys, obj,runtime,gap,abs_gap,N,C):
    # ── 1. Objective value ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x_keys, obj, color='#378ADD', width=0.6, zorder=2)
    ax.set_title('Objective value by instance', fontsize=11)
    ax.set_ylabel('Objective value')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    style_ax(ax,x,x_keys)
    plt.tight_layout()
    plt.savefig('./figures/plot_objective.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ── 2. Runtime ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x_keys, runtime, color='#7F77DD', width=0.6, zorder=2)
    ax.set_title('Runtime by instance (seconds)', fontsize=11)
    ax.set_ylabel('Seconds')
    style_ax(ax,x,x_keys)
    plt.tight_layout()
    plt.savefig('./figures/plot_runtime.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ── 3. Gap analysis ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    gap_pct = np.array(gap) * 100

    ax.bar(x, gap_pct, width=0.6, label='gap (%)', color='#378ADD', zorder=2)
    ax.legend(fontsize=8, framealpha=0.4)
    ax.set_title('Gap % analysis', fontsize=11)
    ax.set_ylabel('Gap %')
    style_ax(ax,x,x_keys)
    plt.tight_layout()
    plt.savefig('./figures/plot_gap.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ── 3. Gap analysis ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    
    
    ax.bar(x - 0.2, gap,     width=0.35, label='gap (%)',  color='#378ADD', zorder=2)
    ax.bar(x + 0.2, abs_gap, width=0.35, label='abs_gap',  color='#1D9E75', zorder=2)
    ax.legend(fontsize=8, framealpha=0.4)
    ax.set_title('Gap analysis', fontsize=11)
    ax.set_ylabel('Gap value')
    style_ax(ax,x,x_keys)
    plt.tight_layout()
    plt.savefig('./figures/plot_gap.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ── 4. Runtime vs N ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(N, runtime, color='#7F77DD', s=60, zorder=3, edgecolors='white', linewidths=0.6)
    ax.set_title('Runtime vs N (nodes)', fontsize=11)
    ax.set_xlabel('N (nodes)')
    ax.set_ylabel('Runtime (s)')
    scatter_style(ax)
    #label_outliers(ax, N, runtime, instances)
    plt.tight_layout()
    plt.savefig('./figures/plot_runtime_vs_N.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ── 5. Runtime vs C ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(C, runtime, color='#1D9E75', s=60, zorder=3, edgecolors='white', linewidths=0.6)
    ax.set_title('Runtime vs C (clusters)', fontsize=11)
    ax.set_xlabel('C (clusters)')
    ax.set_ylabel('Runtime (s)')
    scatter_style(ax)
    #label_outliers(ax, C, runtime, instances)
    plt.tight_layout()
    plt.savefig('./figures/plot_runtime_vs_C.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ── 6. Objective vs N ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(N, obj, color='#378ADD', s=60, zorder=3, edgecolors='white', linewidths=0.6)
    ax.set_title('Objective value vs N', fontsize=11)
    ax.set_xlabel('N (nodes)')
    ax.set_ylabel('Objective value')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    scatter_style(ax)
    #label_outliers(ax, N, obj, instances)
    plt.tight_layout()
    plt.savefig('./figures/plot_obj_vs_N.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return "Plots Created and Exported Successfully"