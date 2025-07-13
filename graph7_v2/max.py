import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
import pickle
import os
import logging
from itertools import combinations
import math
from math import isclose    



# Load G and D from the local files
with open('graph.pkl', 'rb') as f:
    G = pickle.load(f)
with open('D.pkl', 'rb') as f:
    D = pickle.load(f)
with open('shortest_paths.pkl', 'rb') as f:
    shortest_paths = pickle.load(f) 
# load maximizer_dict from the local file   







def floor_power_of_2(x):
    if x <= 0:
        return 0 # Return 1 for non-positive input
    elif math.isinf(x):
        return float("inf")  # Return infinity for infinite input
    else:
        return 2 ** math.floor(math.log2(x))
    
    


def find_max_distance(G, distance_oracle):
    max_distance = float("-inf")
    for key1, value1 in distance_oracle.items():
        for key2, value2 in value1.items():
            if value2 > max_distance:
                max_distance = value2
    return max_distance
    
max_d_value = int(find_max_distance(G, D))
d1_d2_list = [0]
i = floor_power_of_2((max_d_value))


while i >= 1:
    d1_d2_list.append(i)
    i //= 2
# print(max_d_value)








def maximizer(G, x, y, d1, d2):
    G = G.copy()
    max_edges = set()
    edges_set = set()
    max_xy_edge = None
    max_xy_path = None
    max_xy_path_new = None

    # Cache distance calculations
    # distance_cache = {}

    # def get_distance_cached(node1, node2):
    #     if (node1, node2) not in distance_cache:
    #         distance_cache[(node1, node2)] = distance_oracle.get_distance(node1, node2)
    #     return distance_cache[(node1, node2)]

    if nx.has_path(G, x, y):
        path = shortest_paths[(x, y)]
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            edges_set.add((u, v))

    for u, v in edges_set:
        # if get_distance_cached(x, u) >= d1 and get_distance_cached(y, v) >= d2:
        #     max_edges.add((u, v))
        if (D[x][u] >= d1 and D[y][v] >= d2) :
        
            max_edges.add((u, v))   

    max_xy_distance = float('-inf')
    for u, v in max_edges:
        edge_data = G.get_edge_data(u, v)
        if G.has_edge(u, v):
            G.remove_edge(u, v)
        # D = preprocess_graph(G)
        # distance_oracle_new = DistanceOracle(D)
        if nx.has_path(G, x, y):
            xy_distance , xy_path = nx.single_source_dijkstra(G, x, y, weight="weight")
            # xy_distance = distance_oracle_new.get_distance(x, y)

            if xy_distance > max_xy_distance:
                max_xy_edge = (u, v)
                max_xy_path = xy_path
                max_xy_distance = xy_distance
        G.add_edge(u, v, **edge_data)

    if max_xy_path is None or len(max_xy_path) < 2:
        return max_xy_edge, [max_xy_path] if max_xy_path else []

    s = 0  # Start index of current subpath
    max_xy_path_new = []
    deviation_found = False

    for i in range(len(max_xy_path) - 1):
        u = max_xy_path[s]
        v = max_xy_path[i + 1]
        
        # Direct shortest path distance
        uv_distance = D[u][v]   
        
        # Path distance along max_xy_path from u to v
        uv_path_distance = sum(
            G[max_xy_path[j]][max_xy_path[j + 1]]['weight']
            for j in range(s, i + 1)
        )

        
        # Check if path deviates from shortest path
        if uv_distance < uv_path_distance:
            deviation_found = True
            # Define nodes: s (start), a (before deviation), b (deviation), t (end)
            s_node = max_xy_path[0]  # Start node
            a = max_xy_path[i] if i > 0 else max_xy_path[0]  # Node before deviation
            b = v  # Deviation node
            t = max_xy_path[-1]  # End node
            # Construct the output: [[s, a], (a, b), [b, t]]
            s_to_a = [s_node, a]  # Exactly [s, a]
            a_to_b = (a, b)  # Edge from a to b
            b_to_t = [b, t]  # Exactly [b, t]
            max_xy_path_new = [s_to_a, a_to_b, b_to_t]
            break
    
    # If no deviation, return original path wrapped in a list
    if not deviation_found:
        max_xy_path_new = [max_xy_path]
    
    return max_xy_edge, max_xy_path_new






# Initialize a dictionary to store the maximizer output
maximizer_dict = {}

maximizer_function = maximizer

# Collect errors to print after the loop
errors = []

start_time = time.time()

# Ensure G is copied for thread safety
G = G.copy()

# Define a function to process a single pair of nodes
def process_pair(G, x, y, d1, d2):
    try:
        result = maximizer_function(G, x, y, d1, d2)
        max_edge, max_path = result
        return (x, y, d1, d2), (max_edge, max_path)
    except nx.NetworkXNoPath:
        print(f"Error processing pair ({x}, {y}, {d1}, {d2}): No path found")
        return (x, y, d1, d2), None



nodes = list(G.nodes)
num_cores = os.cpu_count()

tasks = []

# Generate tasks
for x in nodes:
    for y in nodes:
        if x != y:
            for d1 in d1_d2_list:
                for d2 in d1_d2_list:
                        tasks.append((x, y, d1, d2))
                        

tasks = list(set(tasks))  # Remove duplicates
# Use ProcessPoolExecutor to parallelize the computation
with ProcessPoolExecutor(max_workers=num_cores) as executor:
    futures = {executor.submit(process_pair, G, x, y, d1, d2): (x, y, d1, d2) for x, y, d1, d2 in tasks}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing pairs"):

        pair_key, result = future.result()

        max_edge, max_path = result
        maximizer_dict[pair_key] = (max_edge, max_path)
        print(f"Pair: {pair_key}, Max Edge: {max_edge}, Max Path: {max_path}")

            


# Save the results
with open('maximizer_dict.pkl', 'wb') as f:
    pickle.dump(maximizer_dict, f)
    
    
with open('processing_times.txt', 'a') as f:
    f.write(f"Maximizer Execution time with multiprocessing: {time.time() - start_time} seconds\n")


print("Execution time:", time.time() - start_time)
print("maximizer_dict.pkl file has been created.")
