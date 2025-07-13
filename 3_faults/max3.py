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

from itertools import product

class Edge:
    def __init__(self, u, v, weight):
        self.u = u
        self.v = v
        self.weight = weight



# Load G and D from the local files
with open('../graph.pkl', 'rb') as f:
    G = pickle.load(f)
with open('../D.pkl', 'rb') as f:
    D = pickle.load(f)
with open('../shortest_paths.pkl', 'rb') as f:
    shortest_paths = pickle.load(f) 
# load maximizer_dict from the local file   
with open('maximizer_dict1.pkl', 'rb') as f:
    maximizer_dict1 = pickle.load(f)
    
with open('maximizer_dict2.pkl', 'rb') as f:
    maximizer_dict2 = pickle.load(f)
with open('maximizer_dict21.pkl', 'rb') as f:
    maximizer_dict21 = pickle.load(f)

with open('../txu_dict.pkl', 'rb') as f:
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
    p_edges = set([(p[i], p[i + 1]) for i in range(len(p) - 1)])
    for edge in F2:
        if (edge.u ,edge.v) in p_edges or (edge.v , edge.u) in p_edges:
            return True
    return False


def intact_from_failure_path(path, F):
    if path is None:
        return False
    if len(path)==1:
        return False

    path_dist = D[path[0]][path[-1]]

    if len(F) == 0:
        return True


    for edge in F:
        u, v = edge[0], edge[1]
        wt = G[u][v]["weight"] if G.has_edge(u, v) else G[v][u]["weight"]

        if (
            isclose(D[path[0]][u] + wt + D[path[-1]][v], path_dist) or
            isclose(D[path[0]][v] + wt + D[path[-1]][u], path_dist)
        ):
            return False
        
    return True

def intact_from_failure_tree(T, F):
    # Check if F is empty
    if T is None:
        # print("bfs_tree_of_S_rooted_x returned None")
        return True
    if not F:
        return True
    
    if isinstance(F, list):
        if set(F) & set(T):
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


def maximizer3(G, x, y, U, V):
    G = G.copy()    

    max_xy_edge = None
    max_xy_path = None
    max_xy_distance = float("-inf")
    max_xy_path_new = []



    possible_edges = combinations(list(G.edges) , 3)

    for F_star in possible_edges:
        eu , ev = F_star[0]
        eu1 , ev1 = F_star[1]
        eu2 , ev2 = F_star[2]
        f_vertices = set([eu, ev, eu1, ev1, eu2, ev2])
        
        f_vertices = list(f_vertices)
        xu_path = shortest_paths[(x, U)]
        vy_path = shortest_paths[(V, y)]
        bfsTree_txu = txu_dict[(x, U)]
        bfsTree_tyv = txu_dict[(y, V)]

        
        if (
            (intact_from_failure_path(xu_path, F_star)
            and intact_from_failure_tree(bfsTree_txu, f_vertices))  
            and ((intact_from_failure_path(vy_path, F_star)
            and intact_from_failure_tree(bfsTree_tyv, f_vertices)))
        ):  
            
            edge1_data = G.get_edge_data(eu, ev)
            edge2_data = G.get_edge_data(eu1, ev1)
            edge3_data = G.get_edge_data(eu2, ev2)
            
            G.remove_edge(eu, ev)
            G.remove_edge(eu1, ev1)
            G.remove_edge(eu2, ev2)
            if not nx.has_path(G, x, y):
                G.add_edge(eu, ev, **edge1_data)
                G.add_edge(eu1, ev1, **edge2_data)
                G.add_edge(eu2, ev2, **edge3_data)
                continue
            


            path2 = nx.dijkstra_path(G, x, y, weight="weight")
            path2_distance = sum(
                get_edge_weight(G, path2[i], path2[i + 1]) for i in range(len(path2) - 1)
            )
            if path2_distance > max_xy_distance:
                max_xy_edge = [(eu, ev), (eu1, ev1), (eu2, ev2)]
                max_xy_path = path2
                max_xy_distance = path2_distance
            G.add_edge(eu1, ev1, **edge2_data)
            G.add_edge(eu, ev, **edge1_data)
            G.add_edge(eu2, ev2, **edge3_data)

                # print(f"added edge: {eu1, ev1}")

        # G.add_edge(eu, ev, **edge1_data)
        # added_edges.append((eu, ev))
        # print(f"added edge: {eu, ev}")

    if max_xy_path is not None:
        s = 0
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
        if len(max_xy_path_new) < 7:
            max_xy_path_new = []
            max_xy_path_new.append(max_xy_path)


    # Compare removed and added edges
    # missing_edges = set(removed_edges) - set(added_edges)
    # if missing_edges:
    #     print(f"Missing edges: {missing_edges}")

    return max_xy_edge, max_xy_path_new



maximizer_dict3 = {}
maximizer_function = maximizer3
errors = []




# Define the process_pair function
def process_pair(G, x, y, u, v):
    try:
        result = maximizer_function(G, x, y, u, v)
        if result is not None:
            max_edge, max_path = result
            return (x, y, u, v), (max_edge, max_path)
    except nx.NetworkXNoPath:
        return (x, y, u, v), None
    except Exception as e:
        return (x, y, u, v), f"Error: {str(e)}"

# Prepare tasks as combinations
tasks = []
for x, y, d1, d2 in product(G.nodes, G.nodes, d1_d2_list, d1_d2_list):
    if x != y:
        try:
            F_star, xy_f_star = maximizer_dict1[(x, y, d1, d2)]
        except KeyError:
            continue

        F_star_m1 = F_star
        F_star_m1_vertex = []
        if F_star is not None and F_star != []:
            if not isinstance(F_star, list):
                F_star = list(F_star)
            if isinstance(F_star[0], int):
                F_star_m1_vertex = [F_star[0], F_star[1]]
            else:
                F_star_m1_vertex = [vertex for E in F_star for vertex in E]

        for v in F_star_m1_vertex:
            try:
                F_star_m21, xy_f_star = maximizer_dict21[(x, y, d1, v)]
                F_star_m2, xy_f_starM2 = maximizer_dict2[(x, y, v, d2)]
            except KeyError:
                print(f"KeyError for {(x, y, d1, v)} or {(x, y, v, d2)}")
                continue

            F_star_m2_vertex = []
            F_star_m21_vertex = []
            if F_star_m2 is not None and F_star_m2 != []:
                if not isinstance(F_star_m2, list):
                    F_star_m2 = list(F_star_m2)
                if isinstance(F_star_m2[0], int):
                    F_star_m2_vertex = [F_star_m2[0], F_star_m2[1]]
                else:
                    F_star_m2_vertex = [vertex for E in F_star_m2 for vertex in E]
            if F_star_m21 is not None and F_star_m21 != []:
                if not isinstance(F_star_m21, list):
                    F_star_m21 = list(F_star_m21)
                if isinstance(F_star_m21[0], int):
                    F_star_m21_vertex = [F_star_m21[0], F_star_m21[1]]
                else:
                    F_star_m21_vertex = [vertex for E in F_star_m21 for vertex in E]

            for u in F_star_m21_vertex:
                tasks.append((x, y, u, v))
            for u in F_star_m2_vertex:
                tasks.append((x, y, v, u))  

# Use ProcessPoolExecutor to parallelize the computation
maximizer_dict3 = {}
errors = []

tasks = list(set(tasks))  # Remove duplicates

with ProcessPoolExecutor() as executor:
    futures = {executor.submit(process_pair, G,  *task): task for task in tasks}

    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing pairs"):
        try:
            key, result = future.result()
            if isinstance(result, str) and result.startswith("Error"):
                errors.append(f"Error for {key}: {result}")
            elif result is not None:
                maximizer_dict3[key] = result
                print(f"Processed {key}: {result}")
            else:
                errors.append(f"No path for {key}")
        except Exception as e:
            errors.append(f"Unexpected error: {str(e)}")

# Log errors
for error in errors:
    print(error)

# Save the results
with open('maximizer_dict3.pkl', 'wb') as f:
    pickle.dump(maximizer_dict3, f)

print("Results saved to maximizer_dict3.pkl")