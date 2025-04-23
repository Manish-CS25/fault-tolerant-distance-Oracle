import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
import pickle
import os
import logging
from itertools import combinations
import math

class Edge:
    def __init__(self, u, v, weight):
        self.u = u
        self.v = v
        self.weight = weight



# Load G and D from the local files
with open('graph.pkl', 'rb') as f:
    G = pickle.load(f)
with open('distances.pkl', 'rb') as f:
    D = pickle.load(f)
with open('shortest_paths.pkl', 'rb') as f:
    shortest_paths = pickle.load(f) 
# load maximizer_dict from the local file   
with open('maximizer_dict1.pkl', 'rb') as f:
    maximizer_dict1 = pickle.load(f)

with open('txu_dict.pkl', 'rb') as f:
    txu_dict = pickle.load(f)   
    







def nearest_power_of_2(x):
    if x <= 0:
        return 0 # Return 1 for non-positive input
    elif math.isinf(x):
        return float("inf")  # Return infinity for infinite input
    else:
        return 2 ** math.floor(math.log2(x))
    
    
def get_edge_weight(G, u, v):
    if G.has_edge(u, v):
        return G[u][v].get('weight', float('inf'))  # Provide a default value if 'weight' is missing
    else:
        return float('inf')

def find_max_distance(G, distance_oracle):
    max_distance = float("-inf")
    for key1, value1 in distance_oracle.items():
        for key2, value2 in value1.items():
            if value2 > max_distance:
                max_distance = value2
    return max_distance
    
max_d_value = int(find_max_distance(G, D))
d1_d2_list = [0]
i = nearest_power_of_2((max_d_value))


while i >= 1:
    d1_d2_list.append(i)
    i //= 2
# print(max_d_value)






def edge_in_path(p, F2):
    if len(p) < 2:
        return False
    p_edges = [(p[i], p[i+ 1]) for i in range(len(p) - 1)]
    for edge in F2:
        if (edge.u ,edge.v) in p_edges or (edge.v , edge.u) in p_edges:
            return True
    return False


def intact_from_failure_path(path, F):
    if path is None:
        return False

    path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    # print(f"F: {F}")

    if len(F) == 0:
        return True

    if isinstance(F, list) and len(F) == 2 and all(isinstance(x, int) for x in F):
        if (F[0], F[1]) in path_edges or (F[1], F[0]) in path_edges:
            return False
        return True

    for edge in F:
        if isinstance(edge, tuple):
            if (edge[0], edge[1]) in path_edges or (edge[1], edge[0]) in path_edges:
                return False
        elif isinstance(edge, list):
            if (edge[0], edge[1]) in path_edges or (edge[1], edge[0]) in path_edges:
                return False
        elif hasattr(edge, 'u') and hasattr(edge, 'v'):
            if (edge.u, edge.v) in path_edges or (edge.v, edge.u) in path_edges:
                return False
        else:
            print(f"Unexpected edge type: {type(edge)}")
            return False

    return True
def intact_from_failure_tree(T, F):
    # Check if F is empty
    if T is None:
        # print("bfs_tree_of_S_rooted_x returned None")
        return True
    if not F:
        return True
    
    if isinstance(F, list) and len(F) == 2 and all(isinstance(x, int) for x in F):
        if F[0] in T or F[1] in T:
            return False
        return True

    # Check if any vertex in F is in the tree T
    for edge in F:
        # Unpack edge into u and v
        if isinstance(edge, Edge):
            u, v = edge.u, edge.v
        elif isinstance(edge, tuple) or isinstance(edge, list):
            u, v = edge
        else:
            print(f"Unexpected edge type: {type(edge)}")
            return False

        if u in T or v in T:
            return False

    return True
def single_edge_in_path(p, F2):
    if p is not None:
        p_edges = [(p[i], p[i + 1]) for i in range(len(p) - 1)]
        
        for edge in F2:
            # unpack edge into u and v
            if isinstance(edge, Edge):
                u, v = edge.u, edge.v
            else:
                u,v = edge
            # check if the edge is in the path
            if (u, v) in p_edges or (v, u) in p_edges:
                return True
        return False
    

def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result




def maximizer21(G, x, y, d1, V):
    G = G.copy()
    max_xy_edge = None
    max_xy_path = None
    max_xy_distance = float("-inf")
    max_xy_path_new = None


    possible_edges = combinations(list(G.edges) , 2)
    for F_star in possible_edges:
        eu , ev = F_star[0];
        eu1 , ev1 = F_star[1]
        if (
            nx.has_path(G, x, eu1)
            and nx.has_path(G, y, ev1)
            and (
                D[x][eu1] >= d1  and D[x][eu] >= d1 and D[x][ev1] >= d1 and D[x][ev] >= d1
            )
            and intact_from_failure_path(shortest_paths[(V, y)], F_star)
            and intact_from_failure_tree(txu_dict[(y, V)], F_star)
        ):
            edge1_data = G.get_edge_data(eu, ev)    
            edge2_data = G.get_edge_data(eu1, ev1)
            
            G.remove_edge(eu, ev)
            G.remove_edge(eu1, ev1) 
            if not nx.has_path(G, x, y):
                G.add_edge(eu, ev, **edge1_data)
                G.add_edge(eu1, ev1, **edge2_data)
                continue

            path2 = nx.dijkstra_path(G, x, y, weight="weight")
            path2_distance = sum(
                get_edge_weight(G, path2[i], path2[i + 1])
                for i in range(len(path2) - 1)
            )
            if path2_distance > max_xy_distance:
                max_xy_edge = [(eu, ev), (eu1, ev1)]
                max_xy_path = path2
                max_xy_distance = path2_distance

            G.add_edge(eu1, ev1, **edge2_data)
            G.add_edge(eu, ev, **edge1_data)

    if max_xy_path is not None:
        s = 0
        max_xy_path_new = []
        for i in range(len(max_xy_path) - 1):
            u = max_xy_path[s]
            v = max_xy_path[i + 1]
            uv_distance = D[u][v]
            uv_distance_path = sum(
                get_edge_weight(G, max_xy_path[j], max_xy_path[j + 1])
                for j in range(s, i + 1)
            )
            if uv_distance != uv_distance_path:
                if i < (len(max_xy_path) - 2):
                    s_to_a_path = [u]
                    intermediate_edge = (v, max_xy_path[i + 2])
                    s_to_a_path.append(max_xy_path[i])
                    max_xy_path_new.append(s_to_a_path)
                    max_xy_path_new.append(intermediate_edge)
                    s = i + 2
        max_xy_path_new.append([u, max_xy_path[len(max_xy_path) - 1]])
        if len(max_xy_path_new) == 1:
            max_xy_path_new = [max_xy_path]
        if len(max_xy_path_new) == 3:
            max_xy_path_new = [max_xy_path]

    return max_xy_edge, max_xy_path_new




# Initialize a dictionary to store the maximizer output
maximizer_dict21 = {}

maximizer_function = maximizer21

# Collect errors to print after the loop
errors = []

start_time = time.time()

# Define a function to process a single pair of nodes
def process_pair(G, x, y, d1, v):
    try:
        result = maximizer_function(G, x, y, d1, v)
        max_edge, max_path = result
        return (x, y, d1, v), (max_edge, max_path)
    except nx.NetworkXNoPath:
        print(f"Error processing pair ({x}, {y}, {d1}, {v}): No path found")
        return (x, y, d1, v), None

# Ensure G is copied for thread safety
G = G.copy()

nodes = list(G.nodes)
num_cores = os.cpu_count()

tasks = []

# Generate tasks
for x in nodes:
    for y in nodes:
        if x != y:
            for d1 in d1_d2_list:
                for d2 in d1_d2_list:

                    F_star, xy_f_star = maximizer_dict1[(x, y, d1, d2)]  
                    F_star_vertex = []

                    if F_star is not None and F_star != []:
                        if not isinstance(F_star, list):
                            F_star = list(F_star)
                        if isinstance(F_star[0], int):
                            F_star_vertex = [F_star[0], F_star[1]]
                        else:
                            F_star_vertex = [vertex for E in F_star for vertex in E]
                    for v in F_star_vertex:
                        tasks.append((x, y, d1, v))
                        

tasks = list(set(tasks))  # Remove duplicates
# Use ProcessPoolExecutor to parallelize the computation
with ProcessPoolExecutor(max_workers=num_cores) as executor:
    futures = {executor.submit(process_pair, G, x, y, d1, v): (x, y, d1, v) for x, y, d1, v in tasks}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing pairs"):

        pair_key, result = future.result()

        max_edge, max_path = result
        maximizer_dict21[pair_key] = (max_edge, max_path)
        print(f"Pair: {pair_key}, Max Edge: {max_edge}, Max Path: {max_path}")

            


# Save the results
with open('maximizer_dict21.pkl', 'wb') as f:
    pickle.dump(maximizer_dict21, f)

logging.info("Results saved to maximizer_dict21.pkl")
logging.info(f"Total execution time: {time.time() - start_time:.2f} seconds")