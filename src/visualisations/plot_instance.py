import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.lines import Line2D


def plot_instance_simple(instance, title="TSPLIB Instance", save_path=None, figsize=(10, 8)):
    """
    Visualize a TSP instance showing only nodes and cluster information (No Depot).

    Args:
        instance: Dictionary containing 'x_coordinates', 'y_coordinates', 
                  'cluster_assignments', and 'n_clusters'.
    """
    # 1. Extract and Prep Data
    x_coords = np.array(instance['x_coords'])
    y_coords = np.array(instance['y_coords'])
    cluster_assignments = np.array(instance['clusters'])
    n_clusters = instance['cluster_count']
    n_nodes = len(x_coords)

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate distinct colors for each cluster
    cmap = plt.cm.get_cmap('tab10', n_clusters) 
    colors = [cmap(i) for i in range(n_clusters)]

    # 3. Plot Nodes by Cluster
    # We iterate through unique cluster IDs (assuming 0 to n_clusters-1)
    for cluster_id in range(n_clusters):
        mask = (cluster_assignments == cluster_id)
        
        ax.scatter(x_coords[mask], y_coords[mask],
                   color=colors[cluster_id], marker='o', s=150,
                   edgecolors='black', linewidths=1.5,
                   label=f'Cluster {cluster_id}', zorder=3)

        # Optional: Add index labels next to points
        for i in np.where(mask)[0]:
            ax.text(x_coords[i], y_coords[i] + 0.2, str(i+1), 
                    fontsize=9, ha='center', va='bottom')

    # 4. Create Legend
    cluster_patches = [mpatches.Patch(color=colors[i], label=f'Cluster {i}')
                      for i in range(min(n_clusters, 10))]
    
    if n_clusters > 10:
        cluster_patches.append(mpatches.Patch(color='gray', label='... more clusters'))

    ax.legend(handles=cluster_patches, loc='upper left', bbox_to_anchor=(1.02, 1), 
              title="Clusters", borderaxespad=0)

    # 5. Formatting
    ax.set_xlabel('Latitude (X)', fontsize=12)
    ax.set_ylabel('Longitude (Y)', fontsize=12)
    ax.set_title(f"{title}\n{n_nodes} Nodes | {n_clusters} Clusters", fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    return fig, ax