import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from pathlib import Path
import math

def plot_ctsp_result(
    input_all_dict,
    result_route,       # list of local node indices in tour order
    result_clusters,    # list of cluster indices in tour order
    input_clusters,     # dict: cluster_id -> list of member global node ids
    input_result_distance,
    show=False,
    save_path=None,
    figsize=None,
    input_title=None
):
    """
    input_all_dict=all_dict
    result_route=out_route
    result_clusters=out_clusters
    input_clusters=clusters
    input_result_distance=result['objective']
    show=True
    save_path=None
    figsize=None
    input_title=None
    
    """
    N = len(input_all_dict)          # ← removed -1 (no depot)
    C = len(input_clusters)

    if figsize is None:
        if C <= 20:
            figsize = (8,10)
        elif C <= 50:
            figsize = (10, 12)
        elif C <= 100:
            figsize = (15, 20)
        else:
            figsize = (15, 20)
            
    fig, ax = plt.subplots(figsize=figsize)

    # Coordinate lookup: (cluster, indexed_node) -> coords
    coord_lookup = {}
    for node_data in input_all_dict.values():
        key = (node_data['cluster'], node_data['indexed_node'])
        coord_lookup[key] = node_data['coordinates']

    # Cluster colors
    cluster_colors = sns.color_palette("pastel", C + 1)

    # Plot all cluster nodes (faded background)
    for cluster_id, members in input_clusters.items():
        cluster_coords = [input_all_dict[node]["coordinates"] for node in members]
        if not cluster_coords:
            continue
        x, y = zip(*cluster_coords)
        ax.scatter(x, y,
                   c=[cluster_colors[cluster_id % len(cluster_colors)]],
                   s=150, alpha=0.4, edgecolors='gray', linewidths=0.5,
                   zorder=2)

    # Plot tour edges — close the loop with % len()
    # Generate colors per route
    route_colors = sns.color_palette("husl", len(result_route))

    for idx in range(len(result_route)):
        v_route   = result_route[idx]
        v_cluster = result_clusters[idx]
        n_stops   = len(v_route)
        color     = route_colors[idx]

        for i in range(n_stops):
            node_from,    node_to    = v_route[i],    v_route[(i + 1) % n_stops]
            cluster_from, cluster_to = v_cluster[i], v_cluster[(i + 1) % n_stops]

            coord_from = coord_lookup.get((cluster_from, node_from))
            coord_to   = coord_lookup.get((cluster_to,   node_to))

            if coord_from is None:
                raise ValueError(f"Not found: cluster={cluster_from}, node={node_from}")
            if coord_to is None:
                raise ValueError(f"Not found: cluster={cluster_to},   node={node_to}")

            # --- Edge ---
            ax.plot(
                [coord_from[0], coord_to[0]],
                [coord_from[1], coord_to[1]],
                linestyle='-', linewidth=2, color=color, alpha=0.8, zorder=3
            )

            # --- Arrow at midpoint ---
            mid_x = (coord_from[0] + coord_to[0]) / 2
            mid_y = (coord_from[1] + coord_to[1]) / 2
            dx    = coord_to[0] - coord_from[0]
            dy    = coord_to[1] - coord_from[1]
            norm  = max(math.sqrt(dx**2 + dy**2), 1e-9)   # avoid division by zero
            ax.annotate(
                '',
                xy    =(mid_x + dx / norm * 3, mid_y + dy / norm * 3),
                xytext=(mid_x - dx / norm * 3, mid_y - dy / norm * 3),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                zorder=4
            )

            # --- Node marker + label with step number ---
            is_start = (i == 0)
            marker   = '*' if is_start else 's'
            size     = 300 if is_start else 100
            ax.scatter(
                *coord_from,
                c=[color], s=size, marker=marker,
                edgecolors='black', linewidths=1.2,
                zorder=7 if is_start else 5
            )
            ax.text(
                coord_from[0], coord_from[1],
                f"[{i+1}] C{cluster_from}:N{node_from}",   # step number added
                fontsize=8, ha='center', va='top', color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8),
                zorder=6
            )

    # --- Legend: one entry per route ---
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=route_colors[i], linewidth=2, label=f'Route {i+1}')
        for i in range(len(result_route))
    ]
    handles.append(
        Line2D([0], [0], color='none', marker='*', markerfacecolor='gray',
            markersize=10, label='Start node')
    )
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)

    if input_title:
        title = (
                f"{input_title}: {N} nodes, {C} clusters | \n"
                f"Distance: {input_result_distance:.2f}"
                )
    else:
        title = (f"CTSP Solution: {N} nodes, {C} clusters\n"
                 f"Distance: {input_result_distance:.2f}"
                 )

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X Coordinate', fontsize=10)
    ax.set_ylabel('Y Coordinate', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax