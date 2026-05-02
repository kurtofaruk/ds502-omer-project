import pickle
import re
import sys
from pathlib import Path

from sklearn.cluster import KMeans

RANDOM_STATE = 502

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from data_utils import get_tsp_coords


ARTICLE_INSTANCES = [
    "att48", "eil51", "st70", "eil76", "pr76",
    "kroA100", "rat99", "rd100", "eil101", "lin105", "pr107",
    "pr124", "bier127", "pr136", "pr144", "kroA150",
    "pr152", "u159", "rat195", "d198", "kroA200", "kroB200",
    "ts225", "pr226", "gil262", "pr264", "pr299", "lin318",
    "rd400", "fl417", "pr439"
]

C_VALUES = [
    10, 11, 14, 16, 16, 20, 20, 20, 21, 21, 22, 25, 26,
    28, 29, 30, 31, 32, 39, 40, 40, 40, 45, 46, 53, 53, 60, 64,
    80, 84, 88
]

# Name → cluster count lookup; safe even when some .tsp files are absent
C_MAP = dict(zip(ARTICLE_INSTANCES, C_VALUES))


def get_filtered_instances(base_dir):
    path_list = list((base_dir / "data/tsplib").glob("*.tsp"))

    filtered = []
    for path in path_list:
        name = path.stem
        if name in ARTICLE_INSTANCES:
            match = re.search(r'(\d+)', name)
            node_count = int(match.group(1))
            filtered.append({'name': name, 'nodes': node_count, 'path': path})

    filtered.sort(key=lambda x: (x['nodes'], x['name']))
    return filtered


def check_missing(filtered_instances):
    found = {inst['name'] for inst in filtered_instances}
    for key in ARTICLE_INSTANCES:
        if key not in found:
            print(f"[MISSING] {key} not found")


def build_instances(filtered_instances):
    instances = []
    n_total = len(filtered_instances)
    for idx, inst in enumerate(filtered_instances):
        name   = inst['name']
        coords = get_tsp_coords(name)
        if coords is None:
            print(f"[WARNING] Could not load coords for {name}, skipping.")
            continue
        C = C_MAP[name]
        kmeans   = KMeans(n_init=10, n_clusters=C, random_state=RANDOM_STATE)
        clusters = [int(v) + 1 for v in kmeans.fit_predict(coords)]
        instances.append({
            "key":               name,
            "x_coordinates":     coords[:, :1].flatten(),
            "y_coordinates":     coords[:, 1:2].flatten(),
            "cluster_assignments": clusters,
            "n_clusters":        C,
        })
        print(f"  [{idx + 1}/{n_total}] {name} — N={len(coords)}, C={C}")
    return instances


def save_instances(instances, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(instances, f)
    print(f"\nSaved {len(instances)} instances to {output_path}")


def main():
    base_dir = Path(__file__).resolve().parent

    print("── Scanning instance files ───────────────────────────────────────")
    filtered_instances = get_filtered_instances(base_dir)
    check_missing(filtered_instances)
    print(f"Found {len(filtered_instances)} matching instances.\n")

    print("── Building instances ────────────────────────────────────────────")
    instances = build_instances(filtered_instances)

    print("\n── Saving ────────────────────────────────────────────────────────")
    save_instances(instances, base_dir / "data/tsplib_instances.pkl")


if __name__ == "__main__":
    main()
