
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from itertools import combinations
# from itertools import product
# import networkx as nx
# import time
# import os
# from tqdm import tqdm   
# import pickle
# import math
# from math import isclose





# # Load G and D from the local files
# with open('graph.pkl', 'rb') as f:
#     G = pickle.load(f)
# with open('D.pkl', 'rb') as f:
#     D = pickle.load(f)
# with open('shortest_paths.pkl', 'rb') as f:
#     shortest_paths = pickle.load(f)
# with open('maximizer_dict1.pkl', 'rb') as f:
#     maximizer_dict1 = pickle.load(f)

# with open('maximizer_dict2.pkl', 'rb') as f:
#     maximizer_dict2 = pickle.load(f)
# with open('maximizer_dict21.pkl', 'rb') as f:
#     maximizer_dict21 = pickle.load(f)
    
# with open('txu_dict.pkl', 'rb') as f:   
#     txu_dict = pickle.load(f)




# def floor_power_of_2(x):
#     if x <= 0:
#         return 0
#     elif math.isinf(x):
#         return float("inf")
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
# i = floor_power_of_2(max_d_value)
# while i >= 1:
#     d1_d2_list.append(i)
#     i //= 2
    

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





# def maximizer3(G, x, y, U, V):
#     G = G.copy()    

#     max_xy_edge = None
#     max_xy_path = None
#     max_xy_distance = float("-inf")
#     max_xy_path_new = []
    
#     xu_path = shortest_paths[(x, U)]
#     vy_path = shortest_paths[(V, y)]
#     bfsTree_txu = txu_dict[(x, U)]
#     bfsTree_tyv = txu_dict[(y, V)]

#     possible_edges = [(u, v) for u, v in G.edges() if
#                       (intact_from_failure_path(xu_path, [(u, v)]) and intact_from_failure_tree(bfsTree_txu, [(u, v)]) and
#                        intact_from_failure_path(vy_path, [(u, v)]) and intact_from_failure_tree(bfsTree_tyv, [(u, v)]))]

#     # possible_edges = combinations(list(G.edges) , 2)
#     possible_edges = list(combinations(possible_edges, 2))

#     for F_star in possible_edges:
#         eu , ev = F_star[0]
#         eu1 , ev1 = F_star[1]


        
#         # if (
#         #     # nx.has_path(G, x, eu1)
#         #     # and nx.has_path(G, y, ev1)
#         #     # and
#         #     (intact_from_failure_path(xu_path, F_star)
#         #     and intact_from_failure_tree(bfsTree_txu, F_star))
#         #     and ((intact_from_failure_path(vy_path, F_star)
#         #     and intact_from_failure_tree(bfsTree_tyv, F_star)))
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
#         uv_distance = D[u][v]  # Direct distance from u to v
        
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





# start_time = time.time()    
# maximizer_dict3 = {}
# maximizer_function = maximizer3
# errors = []



# # Define the process_pair function
# def process_pair(G, x, y, u, v):
#     try:
#         result = maximizer_function(G, x, y, u, v)
#         if result is not None:
#             max_edge, max_path = result
#             return (x, y, u, v), (max_edge, max_path)
#     except nx.NetworkXNoPath:
#         return (x, y, u, v), None
#     except Exception as e:
#         return (x, y, u, v), f"Error: {str(e)}"

# # Prepare tasks as combinations
# tasks = []
# for x, y, d1, d2 in product(G.nodes, G.nodes, d1_d2_list, d1_d2_list):
#     if x != y:
#         try:
#             F_star, xy_f_star = maximizer_dict1[(x, y, d1, d2)]
#         except KeyError:
#             continue

#         F_star_m1 = F_star
#         F_star_m1_vertex = []
#         if F_star is not None and F_star != []:
#             if not isinstance(F_star, list):
#                 F_star = list(F_star)
#             if isinstance(F_star[0], int):
#                 F_star_m1_vertex = [F_star[0], F_star[1]]
#             else:
#                 F_star_m1_vertex = [vertex for E in F_star for vertex in E]

#         for v in F_star_m1_vertex:
#             try:
#                 F_star_m21, xy_f_star = maximizer_dict21[(x, y, d1, v)]
#                 F_star_m2, xy_f_starM2 = maximizer_dict2[(x, y, v, d2)]
#             except KeyError:
#                 print(f"KeyError for {(x, y, d1, v)} or {(x, y, v, d2)}")
#                 continue

#             F_star_m2_vertex = []
#             F_star_m21_vertex = []
#             if F_star_m2 is not None and F_star_m2 != []:
#                 if not isinstance(F_star_m2, list):
#                     F_star_m2 = list(F_star_m2)
#                 if isinstance(F_star_m2[0], int):
#                     F_star_m2_vertex = [F_star_m2[0], F_star_m2[1]]
#                 else:
#                     F_star_m2_vertex = [vertex for E in F_star_m2 for vertex in E]
#             if F_star_m21 is not None and F_star_m21 != []:
#                 if not isinstance(F_star_m21, list):
#                     F_star_m21 = list(F_star_m21)
#                 if isinstance(F_star_m21[0], int):
#                     F_star_m21_vertex = [F_star_m21[0], F_star_m21[1]]
#                 else:
#                     F_star_m21_vertex = [vertex for E in F_star_m21 for vertex in E]

#             for u in F_star_m21_vertex:
#                 tasks.append((x, y, u, v))
#             for u in F_star_m2_vertex:
#                 tasks.append((x, y, v, u))  

# # Use ProcessPoolExecutor to parallelize the computation
# maximizer_dict3 = {}
# errors = []

# tasks = list(set(tasks))  # Remove duplicates

# with ProcessPoolExecutor() as executor:
#     futures = {executor.submit(process_pair, G,  *task): task for task in tasks}

#     for future in tqdm(as_completed(futures), total=len(futures), desc="Processing pairs"):
#         try:
#             key, result = future.result()
#             if isinstance(result, str) and result.startswith("Error"):
#                 errors.append(f"Error for {key}: {result}")
#             elif result is not None:
#                 maximizer_dict3[key] = result
#                 print(f"Processed {key}: {result}")
#             else:
#                 errors.append(f"No path for {key}")
#         except Exception as e:
#             errors.append(f"Unexpected error: {str(e)}")

# # Log errors
# for error in errors:
#     print(error)

# # Save the results
# with open('maximizer_dict3.pkl', 'wb') as f:
#     pickle.dump(maximizer_dict3, f)


# with open('processing_times.txt', 'a') as f:
#     f.write(f"Maximizer3 processing time: {time.time() - start_time:.2f} seconds\n")


# print("Results saved to maximizer_dict3.pkl")






from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations, product
import networkx as nx
import time
import os
from tqdm import tqdm
import pickle
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
with open('maximizer_dict2.pkl', 'rb') as f:
    maximizer_dict2 = pickle.load(f)
with open('maximizer_dict21.pkl', 'rb') as f:
    maximizer_dict21 = pickle.load(f)
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

def maximizer3(G, x, y, U, V):
    G = G.copy()  # Ensure thread safety
    try:
        max_xy_edge = None
        max_xy_path = None
        max_xy_distance = float("-inf")
        max_xy_path_new = []

        # Retrieve paths and trees
        xu_path = shortest_paths.get((x, U), None)
        vy_path = shortest_paths.get((V, y), None)
        bfsTree_txu = txu_dict.get((x, U), None)
        bfsTree_tyv = txu_dict.get((y, V), None)

        if xu_path is None or vy_path is None or bfsTree_txu is None or bfsTree_tyv is None:
            print(f"Missing data for ({x}, {y}, {U}, {V}): xu_path={xu_path is None}, vy_path={vy_path is None}, "
                  f"bfsTree_txu={bfsTree_txu is None}, bfsTree_tyv={bfsTree_tyv is None}")


        # Filter edges that satisfy all constraints
        possible_edges = []
        for u, v in G.edges:
            try:
                if (
                    intact_from_failure_path(xu_path, [(u, v)]) and
                    intact_from_failure_tree(bfsTree_txu, [(u, v)]) and
                    intact_from_failure_path(vy_path, [(u, v)]) and
                    intact_from_failure_tree(bfsTree_tyv, [(u, v)])
                ):
                    possible_edges.append((u, v))
            except Exception as e:
                print(f"Error processing edge ({u}, {v}) in maximizer3({x}, {y}, {U}, {V}): {e}")
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
                    G.add_edge(eu, ev, **edge1_data)
                    G.add_edge(eu1, ev1, **edge2_data)
                    continue
                path2_distance, path2 = nx.single_source_dijkstra(G, x, y, weight="weight")
                if path2_distance > max_xy_distance:
                    max_xy_edge = [(eu, ev), (eu1, ev1)]
                    max_xy_path = path2
                    max_xy_distance = path2_distance
                G.add_edge(eu, ev, **edge1_data)
                G.add_edge(eu1, ev1, **edge2_data)
            except Exception as e:
                print(f"Error processing F_star {F_star} in maximizer3({x}, {y}, {U}, {V}): {e}")
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
                print(f"Error in deviation loop for max_xy_path[{i}] in maximizer3({x}, {y}, {U}, {V}): {e}")
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
        print(f"Error in maximizer3({x}, {y}, {U}, {V}): {e}")
        return None, []

def process_pair(G, x, y, u, v):
    try:
        result = maximizer3(G, x, y, u, v)
        return (x, y, u, v), result
    except Exception as e:
        print(f"Error in process_pair({x}, {y}, {u}, {v}): {e}")
        return (x, y, u, v), None

# Prepare tasks with validation
nodes = list(G.nodes)
num_cores = os.cpu_count()
tasks = []
for x, y, d1, d2 in product(nodes, nodes, d1_d2_list, d1_d2_list):
    if x != y:
        try:
            F_star, _ = maximizer_dict1.get((x, y, d1, d2), (None, None))
            if F_star is None or F_star == []:
                continue
            if not isinstance(F_star, list):
                print(f"Invalid F_star for {(x, y, d1, d2)}: {F_star}")
                continue
            F_star_m1_vertex = []
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
                F_star_m1_vertex.extend(edge)
            for v in F_star_m1_vertex:
                try:
                    F_star_m21, _ = maximizer_dict21.get((x, y, d1, v), (None, None))
                    F_star_m2, _ = maximizer_dict2.get((x, y, v, d2), (None, None))
                    # if F_star_m21 is None or F_star_m2 is None:
                        # print(f"Missing maximizer_dict21 or maximizer_dict2 for {(x, y, d1, v)} or {(x, y, v, d2)}")
                        # continue
                    F_star_m21_vertex = []
                    F_star_m2_vertex = []
                    for edge in F_star_m21 or []:
                        if not (isinstance(edge, (list, tuple)) and len(edge) == 2):
                            continue
                        if edge[0] not in G.nodes or edge[1] not in G.nodes:
                            continue
                        F_star_m21_vertex.extend(edge)
                    for edge in F_star_m2 or []:
                        if not (isinstance(edge, (list, tuple)) and len(edge) == 2):
                            continue
                        if edge[0] not in G.nodes or edge[1] not in G.nodes:
                            continue
                        F_star_m2_vertex.extend(edge)
        
                    for u in F_star_m21_vertex:
                        if u not in G.nodes or u not in D:
                            continue
                        tasks.append((x, y, u, v))
                    for u in F_star_m2_vertex:
                        if u not in G.nodes or u not in D:
                            continue
                        tasks.append((x, y, v, u))
                except KeyError as ke:
                    print(f"KeyError for {(x, y, d1, v)} or {(x, y, v, d2)}: {ke}")
                    continue
        except Exception as e:
            print(f"Error generating task for {(x, y, d1, d2, v)}: {e}")
            continue
    
tasks = list(set(tasks))
print(f"Total tasks: {len(tasks)}")

# Run in batches
batch_size = 1000
maximizer_dict3 = {}
errors = []
start_time = time.time()



with open('maximizer_dict3.pkl', 'ab') as f:  # append-binary mode
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1} with {len(batch)} tasks")
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = {executor.submit(process_pair, G, x, y, u, v): (x, y, u, v) for x, y, u, v in batch}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {i//batch_size + 1}"):
                try:
                    pair_key, result = future.result()
                    if result:
                        max_edge, max_path = result
                        # Write each result as soon as it's ready
                        # print(f"Processed {pair_key}: {max_edge}, {max_path}")
                        pickle.dump({pair_key: (max_edge, max_path)}, f)
                    else:
                        errors.append(f"No result for {pair_key}")
                except Exception as e:
                    errors.append(f"Error processing future for {futures[future]}: {e}")





for error in errors:
    print(error)


with open('processing_times.txt', 'a') as f:
    f.write(f"Maximizer3 Processing Time: {time.time() - start_time:.2f} seconds\n")
print(f"Maximizer3 Processing Time: {time.time() - start_time:.2f} seconds")
print("maximizer_dict3.pkl file has been created.")