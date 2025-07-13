# import networkx as nx
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from tqdm import tqdm
# import time
# import pickle
# import os
# import logging
# from itertools import combinations
# import math
# from math import isclose




# # Load G and D from the local files
# with open('graph.pkl', 'rb') as f:
#     G = pickle.load(f)
# with open('D.pkl', 'rb') as f:
#     D = pickle.load(f)
# with open('shortest_paths.pkl', 'rb') as f:
#     shortest_paths = pickle.load(f) 
# # load maximizer_dict from the local file   
# with open('maximizer_dict1.pkl', 'rb') as f:
#     maximizer_dict1 = pickle.load(f)

# with open('txu_dict.pkl', 'rb') as f:
#     txu_dict = pickle.load(f)   
    







# def floor_power_of_2(x):
#     if x <= 0:
#         return 0 # Return 1 for non-positive input
#     elif math.isinf(x):
#         return float("inf")  # Return infinity for infinite input
#     else:
#         return 2 ** math.floor(math.log2(x))
    


# def find_max_distance(G, distance_oracle):
#     max_distance = float("-inf")
#     for key1, value1 in distance_oracle.items():
#         for key2, value2 in value1.items():
#             if value2 > max_distance:
#                 max_distance = value2
#     return max_distance
    
# max_d_value = int(find_max_distance(G, D))
# d1_d2_list = [0]
# i = floor_power_of_2((max_d_value))


# while i >= 1:
#     d1_d2_list.append(i)
#     i //= 2
# # print(max_d_value)


# def intact_from_failure_path(path, F):
#     if path is None:
#         return True
#     if len(path)==1:
#         return True

#     path_dist = D[path[0]][path[-1]]

#     if len(F) == 0:
#         return True


#     for edge in F:
#         u, v = edge[0], edge[1]
#         wt = G[u][v]["weight"] 

#         if (
#             isclose(D[path[0]][u] + wt + D[path[-1]][v], path_dist) or
#             isclose(D[path[0]][v] + wt + D[path[-1]][u], path_dist)
#         ):
#             return False

#     return True





# def intact_from_failure_tree(T, F):
#     # Check if F is empty
#     if T is None:
#         # print("bfs_tree_of_S_rooted_x returned None")
#         return True
#     if not F:
#         return True


#     # Check if any vertex in F is in the tree T
#     for edge in F:
#         # Unpack edge into u and v
#         u, v = edge[0], edge[1]

#         if u in T or v in T:
#             return False

#     return True





    






# def maximizer21(G, x, y, d1, V):
#     G = G.copy()
#     max_xy_edge = None
#     max_xy_path = None
#     max_xy_distance = float("-inf")
#     max_xy_path_new = None
#     Vy_path = shortest_paths[(V, y)]
#     tyV_tree = txu_dict[(y, V)]
#     possible_edges = [(eu, ev) for eu, ev in G.edges() if D[x][ev] >= d1 and D[x][eu] >= d1 and intact_from_failure_path(Vy_path, [(eu, ev)]) and intact_from_failure_tree(tyV_tree, [(eu, ev)])]   

#     possible_edges = list(combinations(possible_edges, 2))
#     for F_star in possible_edges:
#         eu , ev = F_star[0];
#         eu1 , ev1 = F_star[1]
#         # if (
#         #     # nx.has_path(G, x, eu1)
#         #     # and nx.has_path(G, y, ev1)
#         #     # and 
#         #     (
#         #         D[x][eu1] >= d1  and D[x][eu] >= d1 and D[x][ev1] >= d1 and D[x][ev] >= d1
#         #     )
#         #     and intact_from_failure_path(shortest_paths[(V, y)], F_star)
#         #     and intact_from_failure_tree(txu_dict[(y, V)], F_star)
#         # ):
#         edge1_data = G.get_edge_data(eu, ev)    
#         edge2_data = G.get_edge_data(eu1, ev1)
        
#         G.remove_edge(eu, ev)
#         G.remove_edge(eu1, ev1) 
#         if not nx.has_path(G, x, y):
#             G.add_edge(eu, ev, **edge1_data)
#             G.add_edge(eu1, ev1, **edge2_data)
#             continue

#         path2_distance, path2 = nx.single_source_dijkstra(G, x, y, weight="weight")


#         if path2_distance > max_xy_distance:
#             max_xy_edge = [(eu, ev), (eu1, ev1)]
#             max_xy_path = path2
#             max_xy_distance = path2_distance

#         G.add_edge(eu1, ev1, **edge2_data)
#         G.add_edge(eu, ev, **edge1_data)

#     if max_xy_path is None or len(max_xy_path) < 2:
#         return max_xy_edge, []

#     s = 0  # Start index of current subpath
#     max_xy_path_new = []
#     deviations_found = 0  # Counter for deviations

#     i = 0
#     while i < len(max_xy_path) - 1 and deviations_found < 2:
#         u = max_xy_path[s]
#         v = max_xy_path[i + 1]
        
#         # Direct shortest path distance
#         uv_distance = D[u][v]
        
#         # Path distance along max_xy_path from u to v
#         uv_path_distance = sum(
#             G[max_xy_path[j]][max_xy_path[j + 1]]['weight']
#             for j in range(s, i + 1)
#         )

        
#         # Check if path deviates from shortest path
#         if uv_distance < uv_path_distance:
#             deviations_found += 1
#             # Define nodes for the deviation
#             s_node = max_xy_path[s]  # Start of current subpath
#             a = max_xy_path[i] if i > s else s_node  # Node before deviation
#             b = v  # Deviation node
            
#             if deviations_found == 1:
#                 # First deviation: store [s, a], (a, b)
#                 first_s_to_a = [s_node, a]  # Subpath from start to a
#                 first_a_to_b = (a, b)  # First deviating edge
#                 # Update start index to b for next subpath
#                 s = i + 1
#             elif deviations_found == 2:
#                 # Second deviation: store [b_prev, c], (c, d), [d, t]
#                 # b_prev is the b from the first deviation
#                 c = a  # Node before second deviation
#                 d = b  # Second deviation node
#                 t = max_xy_path[-1]  # End node
#                 b_to_c = [first_a_to_b[1], c]  # Subpath from first b to c
#                 c_to_d = (c, d)  # Second deviating edge
#                 d_to_t = [d, t]  # Subpath from d to t
#                 # Construct the full output
#                 max_xy_path_new = [first_s_to_a, first_a_to_b, b_to_c, c_to_d, d_to_t]
#                 break
        
#         i += 1
    
#     # If fewer than 2 deviations were found
#     if deviations_found < 2:
#         if deviations_found == 1:
#             # Only one deviation: construct [s, a], (a, b), [b, t]
#             t = max_xy_path[-1]
#             b_to_t = [first_a_to_b[1], t]
#             max_xy_path_new = [first_s_to_a, first_a_to_b, b_to_t]
#         else:
#             # No deviations: return original path wrapped in a list
#             max_xy_path_new = [max_xy_path[0], max_xy_path[-1]] 
    
#     return max_xy_edge, max_xy_path_new




# # Initialize a dictionary to store the maximizer output
# maximizer_dict21 = {}

# maximizer_function = maximizer21

# # Collect errors to print after the loop
# errors = []

# start_time = time.time()

# # Define a function to process a single pair of nodes
# def process_pair(G, x, y, d1, v):
#     try:
#         result = maximizer_function(G, x, y, d1, v)
#         max_edge, max_path = result
#         return (x, y, d1, v), (max_edge, max_path)
#     except nx.NetworkXNoPath:
#         print(f"Error processing pair ({x}, {y}, {d1}, {v}): No path found")
#         return (x, y, d1, v), None

# # Ensure G is copied for thread safety
# G = G.copy()

# nodes = list(G.nodes)
# num_cores = os.cpu_count()

# tasks = []

# # Generate tasks
# for x in nodes:
#     for y in nodes:
#         if x != y:
#             for d1 in d1_d2_list:
#                 for d2 in d1_d2_list:

#                     F_star, xy_f_star = maximizer_dict1[(x, y, d1, d2)]  
#                     F_star_vertex = []

#                     if F_star is not None and F_star != []:
#                         if not isinstance(F_star, list):
#                             F_star = list(F_star)
#                         if isinstance(F_star[0], int):
#                             F_star_vertex = [F_star[0], F_star[1]]
#                         else:
#                             F_star_vertex = [vertex for E in F_star for vertex in E]
#                     for v in F_star_vertex:
#                         tasks.append((x, y, d1, v))
                        

# tasks = list(set(tasks))  # Remove duplicates
# # Use ProcessPoolExecutor to parallelize the computation
# with ProcessPoolExecutor(max_workers=num_cores) as executor:
#     futures = {executor.submit(process_pair, G, x, y, d1, v): (x, y, d1, v) for x, y, d1, v in tasks}
#     for future in tqdm(as_completed(futures), total=len(futures), desc="Processing pairs"):

#         pair_key, result = future.result()

#         max_edge, max_path = result
#         maximizer_dict21[pair_key] = (max_edge, max_path)
#         print(f"Pair: {pair_key}, Max Edge: {max_edge}, Max Path: {max_path}")

            


# # Save the results
# with open('maximizer_dict21.pkl', 'wb') as f:
#     pickle.dump(maximizer_dict21, f)

# end_time = time.time()
# execution_time = end_time - start_time

# with open('processing_times.txt', 'a') as f:
#     f.write(f"Processing time for maximizer21: {execution_time} seconds\n")

# print("Maximizer dictionary saved to maximizer_dict21.pkl")

# print(f"Execution time: {execution_time} seconds")




























import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
import pickle
import os
from itertools import combinations
import math
from math import isclose

# Load data
with open('graph.pkl', 'rb') as f:
    G = pickle.load(f)
with open('D.pkl', 'rb') as f:
    D = pickle.load(f)
with open('shortest_paths.pkl', 'rb') as f:
    shortest_paths = pickle.load(f)
with open('maximizer_dict1.pkl', 'rb') as f:
    maximizer_dict1 = pickle.load(f)
with open('txu_dict.pkl', 'rb') as f:
    txu_dict = pickle.load(f)

# Validate node consistency
graph_nodes = set(G.nodes)
d_nodes = set(D.keys())
missing_nodes = graph_nodes - d_nodes
if missing_nodes:
    print(f"Warning: Nodes in G but not in D: {missing_nodes}")
for node in d_nodes:
    missing_subnodes = graph_nodes - set(D[node].keys())
    if missing_subnodes:
        print(f"Warning: Node {node} missing distances to: {missing_subnodes}")

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
            if value2 > max_distance:
                max_distance = value2
    return max_distance

max_d_value = int(find_max_distance(D))
d1_d2_list = [0]
i = floor_power_of_2(max_d_value)
while i >= 1:
    d1_d2_list.append(i)
    i //= 2

def intact_from_failure_path(path, F):
    if path is None or len(path) == 1:
        return True
    try:
        path_dist = D[path[0]][path[-1]]
    except KeyError as ke:
        print(f"KeyError in intact_from_failure_path: {ke}, path={path}")
        return False
    if len(F) == 0:
        return True
    for edge in F:
        try:
            u, v = edge[0], edge[1]
            wt = G[u][v]["weight"]
            if (
                isclose(D[path[0]][u] + wt + D[path[-1]][v], path_dist) or
                isclose(D[path[0]][v] + wt + D[path[-1]][u], path_dist)
            ):
                return False
        except KeyError as ke:
            print(f"KeyError in intact_from_failure_path for edge {edge}: {ke}")
            return False
    return True

def intact_from_failure_tree(T, F):
    if T is None or not F:
        return True
    for edge in F:
        try:
            u, v = edge[0], edge[1]
            if u in T or v in T:
                return False
        except Exception as e:
            print(f"Error in intact_from_failure_tree for edge {edge}: {e}")
            return False
    return True

def maximizer21(G, x, y, d1, v):
    G = G.copy()  # Ensure thread safety
    try:
        max_xy_edge = None
        max_xy_path = None
        max_xy_distance = float("-inf")
        max_xy_path_new = []
        
        Vy_path = shortest_paths.get((v, y), None)
        tyV_tree = txu_dict.get((y, v), None)
        if Vy_path is None or tyV_tree is None:
            print(f"Missing shortest_paths or txu_dict for ({x}, {y}, {d1}, {v})")
            return None, []
        
        possible_edges = []
        for eu, ev in G.edges():
            try:
                dist_x_ev = D.get(x, {}).get(ev, float('inf'))
                dist_x_eu = D.get(x, {}).get(eu, float('inf'))
                if dist_x_ev >= d1 and dist_x_eu >= d1 and intact_from_failure_path(Vy_path, [(eu, ev)]) and intact_from_failure_tree(tyV_tree, [(eu, ev)]):
                    possible_edges.append((eu, ev))
            except Exception as e:
                print(f"Error processing edge ({eu}, {ev}) in maximizer21({x}, {y}, {d1}, {v}): {e}")
                continue
        
        possible_edges = list(combinations(possible_edges, 2))
        for F_star in possible_edges:
            try:
                eu, ev = F_star[0]
                eu1, ev1 = F_star[1]
                edge1_data = G.get_edge_data(eu, ev)
                edge2_data = G.get_edge_data(eu1, ev1)
                G.remove_edge(eu, ev)
                G.remove_edge(eu1, ev1)
                if not nx.has_path(G, x, y):
                    G.add_edge(eu1, ev1, **edge2_data)
                    G.add_edge(eu, ev, **edge1_data)
                    continue
                path2_distance, path2 = nx.single_source_dijkstra(G, x, y, weight="weight")
                if path2_distance > max_xy_distance:
                    max_xy_edge = [(eu, ev), (eu1, ev1)]
                    max_xy_path = path2
                    max_xy_distance = path2_distance
                G.add_edge(eu1, ev1, **edge2_data)
                G.add_edge(eu, ev, **edge1_data)
            except Exception as e:
                print(f"Error processing F_star {F_star} in maximizer21({x}, {y}, {d1}, {v}): {e}")
                continue

        if max_xy_path is None or len(max_xy_path) < 2:
            return max_xy_edge, []

        s = 0
        max_xy_path_new = []
        deviations_found = 0
        i = 0
        while i < len(max_xy_path) - 1 and deviations_found < 2:
            try:
                u = max_xy_path[s]
                v = max_xy_path[i + 1]
                uv_distance = D.get(u, {}).get(v, float('inf'))
                uv_path_distance = sum(
                    G[max_xy_path[j]][max_xy_path[j + 1]]['weight']
                    for j in range(s, i + 1)
                )
                if uv_distance < uv_path_distance:
                    deviations_found += 1
                    s_node = max_xy_path[s]
                    a = max_xy_path[i] if i > s else s_node
                    b = v
                    if deviations_found == 1:
                        first_s_to_a = [s_node, a]
                        first_a_to_b = (a, b)
                        s = i + 1
                    elif deviations_found == 2:
                        c = a
                        d = b
                        t = max_xy_path[-1]
                        b_to_c = [first_a_to_b[1], c]
                        c_to_d = (c, d)
                        d_to_t = [d, t]
                        max_xy_path_new = [first_s_to_a, first_a_to_b, b_to_c, c_to_d, d_to_t]
                        break
                i += 1
            except Exception as e:
                print(f"Error in deviation loop for max_xy_path[{i}] in maximizer21({x}, {y}, {d1}, {v}): {e}")
                break

        if deviations_found < 2:
            if deviations_found == 1:
                t = max_xy_path[-1]
                b_to_t = [first_a_to_b[1], t]
                max_xy_path_new = [first_s_to_a, first_a_to_b, b_to_t]
            else:
                max_xy_path_new = [max_xy_path[0], max_xy_path[-1]]
        
        return max_xy_edge, max_xy_path_new
    except Exception as e:
        print(f"Error in maximizer21({x}, {y}, {d1}, {v}): {e}")
        return None, []

def process_pair(G, x, y, d1, v):
    try:
        # start_time = time.time()
        result = maximizer21(G, x, y, d1, v)
        # print(f"Completed task ({x}, {y}, {d1}, {v}) in {time.time() - start_time:.2f} seconds")
        return (x, y, d1, v), result
    except Exception as e:
        print(f"Error in process_pair({x}, {y}, {d1}, {v}): {e}")
        return (x, y, d1, v), None

# Prepare tasks with validation
nodes = list(G.nodes)
num_cores = os.cpu_count()  
tasks = []
for x in nodes:
    for y in nodes:
        if x != y:
            for d1 in d1_d2_list:
                for d2 in d1_d2_list:
                    try:
                        F_star, xy_f_star = maximizer_dict1.get((x, y, d1, d2), (None, None))
                        if F_star is None or F_star == []:
                            continue
                        if not isinstance(F_star, list):
                            print(f"Invalid F_star for {(x, y, d1, d2)}: {F_star}")
                            continue
                        F_star_vertex = []
                        for edge in F_star:
                            if not (isinstance(edge, (list, tuple)) and len(edge) == 2):
                                print(f"Invalid edge in F_star for {(x, y, d1, d2)}: {edge}")
                                continue
                            if edge[0] not in G.nodes or edge[1] not in G.nodes:
                                print(f"Edge {edge} in F_star for {(x, y, d1, d2)} contains invalid nodes")
                                continue
                            if edge[0] not in D or edge[1] not in D:
                                print(f"Edge {edge} nodes not in D for {(x, y, d1, d2)}")
                                continue
                            F_star_vertex.extend(edge)
                        for v in F_star_vertex:
                            if v not in G.nodes or v not in D:
                                print(f"Vertex {v} in F_star for {(x, y, d1, d2)} not in G or D")
                                continue
                            tasks.append((x, y, d1, v))
                    except Exception as e:
                        print(f"Error generating task for {(x, y, d1, d2)}: {e}")
                        continue
tasks = list(set(tasks))
print(f"Total tasks: {len(tasks)}")

# Run in batches
batch_size = 1000
maximizer_dict21 = {}
start_time = time.time()
for i in range(0, len(tasks), batch_size):
    batch = tasks[i:i + batch_size]
    print(f"Processing batch {i//batch_size + 1} with {len(batch)} tasks")
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = {executor.submit(process_pair, G, x, y, d1, v): (x, y, d1, v) for x, y, d1, v in batch}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {i//batch_size + 1}"):
            try:
                pair_key, result = future.result()
                if result:
                    max_edge, max_path = result
                    maximizer_dict21[pair_key] = (max_edge, max_path)
                    print(f"Pair: {pair_key}, Max Edge: {max_edge}, Max Path: {max_path}")
            except Exception as e:
                print(f"Error processing future for {futures[future]}: {e}")

# Save results
with open('maximizer_dict21.pkl', 'wb') as f:
    pickle.dump(maximizer_dict21, f)
end_time = time.time()
execution_time = end_time - start_time
with open('processing_times.txt', 'a') as f:
    f.write(f"Processing time for maximizer21: {execution_time} seconds\n")
print("Maximizer dictionary saved to maximizer_dict21.pkl")
print(f"Execution time: {execution_time} seconds")