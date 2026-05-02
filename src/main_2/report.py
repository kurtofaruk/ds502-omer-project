from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _style_bar(ax, x, x_keys):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(x_keys, rotation=45, ha='right', fontsize=7)


def _style_scatter(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(linewidth=0.5, alpha=0.4, zorder=0)


def _fmt_int(ax):
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))


# =============================================================================
# MILP REPORT
# =============================================================================

def plot_outputs(x, x_keys, obj, runtime, gap, abs_gap, N, C,
                 figures_dir='./figures'):

    # 1. Objective value
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x_keys, obj, color='#378ADD', width=0.6, zorder=2)
    ax.set_title('Objective value by instance', fontsize=11)
    ax.set_ylabel('Objective value')
    _fmt_int(ax)
    _style_bar(ax, x, x_keys)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/plot_objective.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Runtime
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x_keys, runtime, color='#7F77DD', width=0.6, zorder=2)
    ax.set_title('Runtime by instance (seconds)', fontsize=11)
    ax.set_ylabel('Seconds')
    _style_bar(ax, x, x_keys)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/plot_runtime.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Gap %
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x, np.array(gap) * 100, width=0.6, color='#378ADD', zorder=2)
    ax.set_title('MIP gap % by instance', fontsize=11)
    ax.set_ylabel('Gap %')
    _style_bar(ax, x, x_keys)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/plot_gap_pct.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Gap % vs absolute gap (side-by-side)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x - 0.2, gap,     width=0.35, label='gap (%)',  color='#378ADD', zorder=2)
    ax.bar(x + 0.2, abs_gap, width=0.35, label='abs_gap',  color='#1D9E75', zorder=2)
    ax.legend(fontsize=8, framealpha=0.4)
    ax.set_title('Gap analysis', fontsize=11)
    ax.set_ylabel('Gap value')
    _style_bar(ax, x, x_keys)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/plot_gap.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 5. Runtime vs N
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(N, runtime, color='#7F77DD', s=60, zorder=3,
               edgecolors='white', linewidths=0.6)
    ax.set_title('Runtime vs N (nodes)', fontsize=11)
    ax.set_xlabel('N (nodes)')
    ax.set_ylabel('Runtime (s)')
    _style_scatter(ax)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/plot_runtime_vs_N.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 6. Runtime vs C
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(C, runtime, color='#1D9E75', s=60, zorder=3,
               edgecolors='white', linewidths=0.6)
    ax.set_title('Runtime vs C (clusters)', fontsize=11)
    ax.set_xlabel('C (clusters)')
    ax.set_ylabel('Runtime (s)')
    _style_scatter(ax)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/plot_runtime_vs_C.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 7. Objective vs N
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(N, obj, color='#378ADD', s=60, zorder=3,
               edgecolors='white', linewidths=0.6)
    ax.set_title('Objective value vs N', fontsize=11)
    ax.set_xlabel('N (nodes)')
    ax.set_ylabel('Objective value')
    _fmt_int(ax)
    _style_scatter(ax)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/plot_obj_vs_N.png', dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# GA COMPARISON REPORT
# =============================================================================

def plot_ga_report(milp_df, ga_df, figures_dir='./figures'):
    """
    Generate MILP vs GA comparison plots.

    Figures saved:
      ga_vs_milp_objective.png — grouped bar: MILP vs GA tour distances
      ga_gap_comparison.png    — GA gap % above MILP per instance
      ga_runtime_scatter.png   — MILP runtime vs GA runtime scatter
      ga_obj_vs_C.png          — tour distance vs C for both methods
    """
    merged = milp_df.merge(ga_df[['key', 'ga_cost', 'ga_time']], on='key', how='inner')
    merged = merged.dropna(subset=['obj', 'ga_cost'])
    merged = merged.sort_values('C').reset_index(drop=True)

    merged['ga_gap_pct'] = ((merged['ga_cost'] - merged['obj']) / merged['obj'] * 100).round(2)

    x      = np.arange(len(merged))
    x_keys = (merged['key'] + '\n(C=' + merged['C'].astype(str) + ')').tolist()

    # 1. Tour distance: MILP vs GA
    fig, ax = plt.subplots(figsize=(14, 6))
    w = 0.35
    ax.bar(x - w / 2, merged['obj'],      width=w, label='MILP optimal',      color='#378ADD', zorder=2)
    ax.bar(x + w / 2, merged['ga_cost'],  width=w, label='Genetic Algorithm', color='#E9C46A', zorder=2)
    ax.set_title('Tour distance: MILP optimal vs Genetic Algorithm', fontsize=11)
    ax.set_ylabel('Tour distance')
    _fmt_int(ax)
    _style_bar(ax, x, x_keys)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/ga_vs_milp_objective.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. GA gap % above MILP
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x, merged['ga_gap_pct'], width=0.6, color='#E9C46A', zorder=2)
    ax.axhline(0, color='#aaa', linewidth=0.8, linestyle='--')
    ax.set_title('GA optimality gap % above MILP (lower is better)', fontsize=11)
    ax.set_ylabel('(GA − MILP) / MILP × 100 %')
    _style_bar(ax, x, x_keys)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/ga_gap_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Runtime: MILP vs GA scatter
    fig, ax = plt.subplots(figsize=(9, 6))
    milp_rt = merged['runtime'].astype(float)
    ga_rt   = merged['ga_time'].astype(float)
    ax.scatter(milp_rt, ga_rt, color='#E9C46A', s=60, zorder=3,
               edgecolors='#c49a1a', linewidths=0.7)
    for _, row in merged.iterrows():
        ax.annotate(row['key'], (float(row['runtime']), float(row['ga_time'])),
                    textcoords='offset points', xytext=(5, 3), fontsize=6.5, color='#555')
    lim = max(milp_rt.max(), ga_rt.max()) * 1.08
    ax.plot([0, lim], [0, lim], '--', color='#aaa', linewidth=1, label='y = x (equal time)')
    ax.set_xlabel('MILP runtime (s)')
    ax.set_ylabel('GA runtime (s)')
    ax.set_title('Runtime: MILP vs Genetic Algorithm', fontsize=11)
    _style_scatter(ax)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/ga_runtime_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Solution quality vs C
    fig, ax = plt.subplots(figsize=(10, 6))
    C_vals = merged['C'].tolist()
    ax.scatter(C_vals, merged['obj'],     color='#378ADD', s=60, zorder=3,
               edgecolors='white', linewidths=0.5, label='MILP optimal')
    ax.scatter(C_vals, merged['ga_cost'], color='#E9C46A', s=60, zorder=3,
               edgecolors='white', linewidths=0.5, label='Genetic Algorithm')
    ax.set_xlabel('C (number of clusters)')
    ax.set_ylabel('Tour distance')
    ax.set_title('Solution quality vs problem size (C)', fontsize=11)
    _fmt_int(ax)
    _style_scatter(ax)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/ga_obj_vs_C.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n── GA vs MILP summary ────────────────────────────────────────────")
    print(f"  GA gap — mean: {merged['ga_gap_pct'].mean():.1f}%  "
          f"median: {merged['ga_gap_pct'].median():.1f}%  "
          f"max: {merged['ga_gap_pct'].max():.1f}%  "
          f"min: {merged['ga_gap_pct'].min():.1f}%")
    print(f"  GA matched or beat MILP in "
          f"{(merged['ga_gap_pct'] <= 0).sum()}/{len(merged)} instances")

    reports_dir = Path(figures_dir).parent / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    merged.to_excel(reports_dir / 'results_ga_comparison.xlsx', index=False)
    print(f"  Saved → {reports_dir / 'results_ga_comparison.xlsx'}")
    return merged
