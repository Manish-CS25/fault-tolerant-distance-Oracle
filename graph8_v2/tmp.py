import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
import pickle
import os
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
with open('maximizer_dict1.pkl', 'rb') as f:
    maximizer_dict1 = pickle.load(f)
with open('txu_dict.pkl', 'rb') as f:
    txu_dict = pickle.load(f)

def floor_power_of_2(x):
    if x <= 0:
        return 0
    elif math.isinf(x):
        return float("inf")
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
i = floor_power_of_2(max_d_value)
while i >= 1:
    d1_d2_list.append(i)
    i //= 2

def intact_from_failure_path(path, F):
    if path is None or len(path) == 1:
        return True
    path_dist = D[path[0]][path[-1]]
    if len(F) == 0:
        return True
    for edge in F:
        u, v = edge[0], edge[1]
        wt = G[u][v]["weight"]
        if (
            isclose(D[path[0]][u] + wt + D[path[-1]][v], path_dist) or
            isclose(D[path[0]][v] + wt + D[path[-1]][u], path_dist)
        ):
            return False
    return True

def intact_from_failure_tree(T, F):
    if T is None or not F:
        return True
    for edge in F:
        u, v = edge[0], edge[1]
        if u in T or v in T:
            return False
    return True

def maximizer2(G, x, y, V, d2):
    G = G.copy()
    max_xy_edge = None
    max_xy_path = None
    max_xy_distance = float("-inf")
    max_xy_path_new = []
    Vx_path = shortest_paths.get((V, x))
    txV = txu_dict.get((x, V))
    if Vx_path is None or txV is None:
        return None, []
    possible_edges = [
        e for e in G.edges
        if D[y].get(e[1], float('-inf')) >= d2 and D[y].get(e[0], float('-inf')) >= d2
        and intact_from_failure_path(Vx_path, [e])
        and intact_from_failure_tree(txV, [e])
    ]
    possible_edges = list(combinations(possible_edges, 2))
    for F_star in possible_edges:
        eu, ev = F_star[0]
        eu1, ev1 = F_star[1]
        edge1_data = G.get_edge_data(eu, ev)
        edge2_data = G.get_edge_data(eu1, ev1)
        G.remove_edge(eu, ev)
        G.remove_edge(eu1, ev1)
        try:
            if not nx.has_path(G, x, y):
                continue
            path2_distance, path2 = nx.single_source_dijkstra(G, x, y, weight="weight")
            if path2_distance > max_xy_distance:
                max_xy_edge = [(eu, ev), (eu1, ev1)]
                max_xy_path = path2
                max_xy_distance = path2_distance
        except nx.NetworkXNoPath:
            pass
        finally:
            G.add_edge(eu1, ev1, **edge2_data)
            G.add_edge(eu, ev, **edge1_data)
    if max_xy_path is None or len(max_xy_path) < 2:
        return max_xy_edge, []
    s = 0
    max_xy_path_new = []
    deviations_found = 0
    i = 0
    while i < len(max_xy_path) - 1 and deviations_found < 2:
        u = max_xy_path[s]
        v = max_xy_path[i + 1]
        uv_distance = D[u][v]
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
    if deviations_found < 2:
        if deviations_found == 1:
            t = max_xy_path[-1]
            b_to_t = [first_a_to_b[1], t]
            max_xy_path_new = [first_s_to_a, first_a_to_b, b_to_t]
        else:
            max_xy_path_new = [max_xy_path[0], max_xy_path[-1]]
    return max_xy_edge, max_xy_path_new

def process_pair(G, x, y, V, d2):
    try:
        result = maximizer2(G, x, y, V, d2)
        max_edge, max_path = result
        return (x, y, V, d2), (max_edge, max_path)
    except Exception as e:
        print(f"Error processing pair ({x}, {y}, {V}, {d2}): {e}")
        return (x, y, V, d2), None

nodes = list(G.nodes)
num_cores = os.cpu_count()
tasks = []
for x in nodes:
    for y in nodes:
        if x != y:
            for d1 in d1_d2_list:
                for d2 in d1_d2_list:
                    F_star, xy_f_star = maximizer_dict1.get((x, y, d1, d2), (None, None))
                    F_star_vertex = []
                    if F_star is not None and F_star != []:
                        if not isinstance(F_star, list):
                            F_star = list(F_star)
                        if isinstance(F_star[0], int):
                            F_star_vertex = [F_star[0], F_star[1]]
                        else:
                            F_star_vertex = [vertex for E in F_star for vertex in E]
                    for v in F_star_vertex:
                        tasks.append((x, y, v, d2))

print(f"Number of tasks: {len(tasks)}")
# For debugging, you can limit the number of tasks:
tasks = tasks[:100]

tasks = list(set(tasks))  # Remove duplicates

maximizer_dict2 = {}
start_time = time.time()

with ProcessPoolExecutor(max_workers=num_cores) as executor:
    futures = {executor.submit(process_pair, G, x, y, V, d2): (x, y, V, d2) for x, y, V, d2 in tasks}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing pairs"):
        pair_key, result = future.result()
        if result:
            max_edge, max_path = result
            maximizer_dict2[pair_key] = (max_edge, max_path)
            # Optionally print for debugging:
            print(f"Pair: {pair_key}, Max Edge: {max_edge}, Max Path: {max_path}")

# with open('maximizer_dict2.pkl', 'wb') as f:
#     pickle.dump(maximizer_dict2, f)

# with open('processing_times.txt', 'a') as f:
#     f.write(f"Maximizer2 Processing Time: {time.time() - start_time} seconds\n")

# print("Maximizer2 Processing Time:", time.time() - start_time)
# print("maximizer_dict2.pkl file has been created.")