# from concurrent.futures import ProcessPoolExecutor, as_completed
# import networkx as nx
# import tqdm
# import time
# import os
# import pickle
# import math
# from itertools import combinations


# # os.environ["NX_CUGRAPH_AUTOCONFIG"] = "True"

# # import networkx as nx
# # print(f"using networkx version {nx.__version__}")

# # # Optional: Suppress specific warnings (requires NetworkX ≥ 3.2)
# # if hasattr(nx, "config") and hasattr(nx.config, "warnings_to_ignore"):
# #     nx.config.warnings_to_ignore.add("cache")

# # Load shared distance dictionary
# with open('D.pkl', 'rb') as f:
#     D = pickle.load(f)

# def nearest_power_of_2(x):
#     if x <= 0:
#         return 0
#     elif math.isinf(x):
#         return float("inf")
#     else:
#         return 2 ** math.floor(math.log2(x))

# def get_edge_weight(G, u, v):
#     return G[u][v].get('weight', float('inf')) if G.has_edge(u, v) else float('inf')

# def find_max_distance(distance_oracle):
#     max_distance = float("-inf")
#     for key1, value1 in distance_oracle.items():
#         for key2, value2 in value1.items():
#             max_distance = max(max_distance, value2)
#     return max_distance

# max_d_value = int(find_max_distance(D))
# d1_d2_list = [0]
# i = nearest_power_of_2(max_d_value)
# while i >= 1:
#     d1_d2_list.append(i)
#     i //= 2

# def maximizer1(G, x, y, d1, d2):
#     G = G.copy()
#     max_xy_edge = None
#     max_xy_path = None
#     max_xy_distance = float("-inf")
#     max_xy_path_new = None

#     possible_edges = combinations(G.edges, 2)

#     for e1, e2 in possible_edges:
#         eu, ev = e1
#         eu1, ev1 = e2

#         if all(D[x][n] >= d1 for n in (eu, ev, eu1, ev1)) and all(D[y][n] >= d2 for n in (eu, ev, eu1, ev1)):
#             edge1_data = G.get_edge_data(eu, ev)
#             edge2_data = G.get_edge_data(eu1, ev1)
#             G.remove_edge(eu, ev)
#             G.remove_edge(eu1, ev1)

#             if not nx.has_path(G, x, y):
#                 G.add_edge(eu, ev, **edge1_data)
#                 G.add_edge(eu1, ev1, **edge2_data)
#                 continue

#             path = nx.dijkstra_path(G, x, y, weight="weight")
#             path_distance = sum(get_edge_weight(G, path[i], path[i + 1]) for i in range(len(path) - 1))

#             if path_distance > max_xy_distance:
#                 max_xy_edge = [e1, e2]
#                 max_xy_path = path
#                 max_xy_distance = path_distance

#             G.add_edge(eu, ev, **edge1_data)
#             G.add_edge(eu1, ev1, **edge2_data)

#     if max_xy_path:
#         s = 0
#         max_xy_path_new = []
#         for i in range(len(max_xy_path) - 1):
#             u = max_xy_path[s]
#             v = max_xy_path[i + 1]
#             uv_distance = D[u][v]
#             uv_distance_path = sum(get_edge_weight(G, max_xy_path[j], max_xy_path[j + 1]) for j in range(s, i + 1))
#             if uv_distance != uv_distance_path and i < len(max_xy_path) - 2:
#                 s_to_a_path = [u, max_xy_path[i]]
#                 max_xy_path_new.append(s_to_a_path)
#                 max_xy_path_new.append((v, max_xy_path[i + 2]))
#                 s = i + 2
#         max_xy_path_new.append([u, max_xy_path[-1]])
#         if len(max_xy_path_new) in [1, 3]:
#             max_xy_path_new = [max_xy_path]

#     return max_xy_edge, max_xy_path_new

# def process_pair(G, x, y, d1, d2):
#     # with open('graph.pkl', 'rb') as f:
#     #     G_local = pickle.load(f)
#     try:
#         result = maximizer1(G, x, y, d1, d2)
#         return (x, y, d1, d2), result
#     except nx.NetworkXNoPath:
#         return (x, y, d1, d2), None

# # Setup
# with open('graph.pkl', 'rb') as f:
#     G = pickle.load(f)

# nodes = list(G.nodes())
# tasks = [(x, y, d1, d2) for x in nodes for y in nodes for d1 in d1_d2_list for d2 in d1_d2_list if x != y]
# print(f"Number of combinations: {len(tasks)}")

# maximizer_dict1 = {}
# start_time = time.time()
# num_cores = os.cpu_count()

# with ProcessPoolExecutor(max_workers=num_cores) as executor:
#     futures = {executor.submit(process_pair, G,  x, y, d1, d2): (x, y, d1, d2) for x, y, d1, d2 in tasks}
#     for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing pairs"):
#         pair_key, result = future.result()
#         if result:
#             max_edge, max_path = result
#             maximizer_dict1[pair_key] = (max_edge, max_path)

# end_time = time.time()
# total_time = end_time - start_time

# print(f"\nMaximizer1 Performance Summary:")
# print(f"Total time: {total_time:.2f} seconds")
# print(f"Pairs processed: {len(tasks)}")
# print(f"Successful pairs: {len(maximizer_dict1)}")
# print(f"Success rate: {(len(maximizer_dict1)/len(tasks))*100:.2f}%")

# with open('maximizer_dict1.pkl', 'wb') as f:
#     pickle.dump(maximizer_dict1, f)

# with open('processing_times.txt', 'a') as f:
#     f.write(f"Time Taken to process Maximizer1: {total_time:.6f} seconds\n")


# print(f"Maximizer dictionary saved to maximizer_dict1.pkl in {total_time:.6f} seconds")











from concurrent.futures import ProcessPoolExecutor, as_completed
import networkx as nx
import tqdm
import time
import os
import pickle
import math
from itertools import combinations

# Load shared distance dictionary
with open('D.pkl', 'rb') as f:
    D = pickle.load(f)

def floor_power_of_2(x):
    if x <= 0:
        return 0
    elif math.isinf(x):
        return float("inf")
    else:
        return 2 ** math.floor(math.log2(x))


def find_max_distance(distance_oracle):
    max_distance = float("-inf")
    for key1, value1 in distance_oracle.items():
        for key2, value2 in value1.items():
            max_distance = max(max_distance, value2)
    return max_distance

max_d_value = int(find_max_distance(D))
d1_d2_list = [0]
i = floor_power_of_2(max_d_value)
while i >= 1:
    d1_d2_list.append(i)
    i //= 2

def maximizer1(G, x, y, d1, d2):
    max_xy_edge = None
    max_xy_path = None
    max_xy_distance = float("-inf")
    max_xy_path_new = None

    possible_edges = combinations(G.edges, 2)

    for e1, e2 in possible_edges:
        eu, ev = e1
        eu1, ev1 = e2
        
        # Check if the distances are valid for both edges
        if all(D.get(x, {}).get(n, float('-inf')) >= d1 for n in (eu, ev, eu1, ev1)) and \
           all(D.get(y, {}).get(n, float('-inf')) >= d2 for n in (eu, ev, eu1, ev1)):
            edge1_data = G.get_edge_data(eu, ev)    
            edge2_data = G.get_edge_data(eu1, ev1)
            G.remove_edge(eu, ev)
            G.remove_edge(eu1, ev1)

            try:
                if not nx.has_path(G, x, y):
                    continue

                path_distance, path = nx.single_source_dijkstra(G, x, y, weight="weight")

                if path_distance > max_xy_distance:
                    max_xy_edge = [e1, e2]
                    max_xy_path = path
                    max_xy_distance = path_distance
            except nx.NetworkXNoPath:
                pass
            finally:
                # Restore the edges in every case
                G.add_edge(eu, ev, **edge1_data)
                G.add_edge(eu1, ev1, **edge2_data)


    if max_xy_path is None or len(max_xy_path) < 2:
        return max_xy_edge, []

    s = 0  # Start index of current subpath
    max_xy_path_new = []
    deviations_found = 0  # Counter for deviations

    i = 0
    while i < len(max_xy_path) - 1 and deviations_found < 2:
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
            deviations_found += 1
            # Define nodes for the deviation
            s_node = max_xy_path[s]  # Start of current subpath
            a = max_xy_path[i] if i > s else s_node  # Node before deviation
            b = v  # Deviation node
            
            if deviations_found == 1:
                # First deviation: store [s, a], (a, b)
                first_s_to_a = [s_node, a]  # Subpath from start to a
                first_a_to_b = (a, b)  # First deviating edge
                # Update start index to b for next subpath
                s = i + 1
            elif deviations_found == 2:
                # Second deviation: store [b_prev, c], (c, d), [d, t]
                # b_prev is the b from the first deviation
                c = a  # Node before second deviation
                d = b  # Second deviation node
                t = max_xy_path[-1]  # End node
                b_to_c = [first_a_to_b[1], c]  # Subpath from first b to c
                c_to_d = (c, d)  # Second deviating edge
                d_to_t = [d, t]  # Subpath from d to t
                # Construct the full output
                max_xy_path_new = [first_s_to_a, first_a_to_b, b_to_c, c_to_d, d_to_t]
                break
        
        i += 1
    
    # If fewer than 2 deviations were found
    if deviations_found < 2:
        if deviations_found == 1:
            # Only one deviation: construct [s, a], (a, b), [b, t]
            t = max_xy_path[-1]
            b_to_t = [first_a_to_b[1], t]
            max_xy_path_new = [first_s_to_a, first_a_to_b, b_to_t]
        else:
            # No deviations: return original path wrapped in a list
            max_xy_path_new = [max_xy_path[0], max_xy_path[-1]] 
    
    return max_xy_edge, max_xy_path_new

def process_pair(G, x, y, d1, d2):
    try:
        result = maximizer1(G, x, y, d1, d2)
        return (x, y, d1, d2), result
    except nx.NetworkXNoPath:
        return (x, y, d1, d2), None

# Setup
with open('graph.pkl', 'rb') as f:
    G = pickle.load(f)

nodes = list(G.nodes())
tasks = [(x, y, d1, d2) for x in nodes for y in nodes for d1 in d1_d2_list for d2 in d1_d2_list if x != y]
print(f"Number of combinations: {len(tasks)}")

maximizer_dict1 = {}
start_time = time.time()
num_cores = os.cpu_count()

with ProcessPoolExecutor(max_workers=num_cores) as executor:
    futures = {executor.submit(process_pair, G, x, y, d1, d2): (x, y, d1, d2) for x, y, d1, d2 in tasks}
    for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing pairs"):
        pair_key, result = future.result()
        if result:
            max_edge, max_path = result
            maximizer_dict1[pair_key] = (max_edge, max_path)

end_time = time.time()
total_time = end_time - start_time

print(f"\nMaximizer1 Performance Summary:")
print(f"Total time: {total_time:.2f} seconds")
print(f"Pairs processed: {len(tasks)}")
print(f"Successful pairs: {len(maximizer_dict1)}")
print(f"Success rate: {(len(maximizer_dict1)/len(tasks))*100:.2f}%")

with open('maximizer_dict1.pkl', 'wb') as f:
    pickle.dump(maximizer_dict1, f)

with open('processing_times.txt', 'a') as f:
    f.write(f"Time Taken to process Maximizer1: {total_time:.6f} seconds\n")

print(f"Maximizer dictionary saved to maximizer_dict1.pkl in {total_time:.6f} seconds")