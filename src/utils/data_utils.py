import re 
import numpy as np

def get_filtered_and_sorted_instances(input_path_list,input_filtered_instances):
    for path in input_path_list:
        name = path.stem  # e.g., 'ulysses16'
        
        # Use regex to find all digits in the name
        match = re.search(r'(\d+)', name)
        
        if match:
            node_count = int(match.group(1))
            
            # Filter logic: between 10 and 100 nodes (inclusive)
            if 48 <= node_count <= 447:
                input_filtered_instances.append({
                    'name': name,
                    'nodes': node_count,
                    'path': path
                })
    input_filtered_instances.sort(key=lambda x: (x['nodes'], x['name']))

    seen_sizes = set()
    unique_size_instances = []

    # Assuming filtered_instances is already sorted by (nodes, name)
    for inst in input_filtered_instances:
        size = inst['nodes']
        if size not in seen_sizes:
            unique_size_instances.append(inst)
            seen_sizes.add(size)
    # View your filtered results
    for inst in unique_size_instances:
        print(f"Instance: {inst['name']} | Nodes: {inst['nodes']}")
    return unique_size_instances

def get_tsp_coords(instance_name):
    """
    Checks for NODE_COORD_SECTION and returns coordinates as a numpy array.
    """
    coords = []
    found_section = False
    
    with open(f'../../data/tsplib/{instance_name}.tsp', 'r') as f:
        lines = f.readlines()
        
        for i, line in enumerate(lines):
            # 1. Look for the start of the coordinate section
            if "NODE_COORD_SECTION" in line:
                found_section = True
                # Start reading from the next line
                for coord_line in lines[i+1:]:
                    parts = coord_line.strip().split()
                    
                    # Stop if we hit EOF or an empty line
                    if not parts or "EOF" in parts:
                        break
                    
                    # 2. Extract X and Y (skipping the first column/index)
                    try:
                        # parts[0] is Index, parts[1] is X, parts[2] is Y
                        x, y = float(parts[1]), float(parts[2])
                        coords.append([x, y])
                    except (ValueError, IndexError):
                        # Skip lines that don't match the coordinate format
                        continue
                break
    
    if not found_section:
        print(f"Warning: NODE_COORD_SECTION not found in {instance_name}")
        return None
        
    return np.array(coords)