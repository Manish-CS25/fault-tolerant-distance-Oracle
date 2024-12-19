import networkx as nx
# import matplotlib.pyplot as plt
import pickle

# Define the Edge class
class Edge:
    def __init__(self, u, v, weight):
        self.u = u
        self.v = v
        self.weight = weight
def preprocess_graph(G):
    # assuming 'weight' is the name of the attribute for edge weights
    return dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))

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

class DistanceOracle:
    def __init__(self,distances=None):
        if distances is None:
            self.data = {}
        else:
            self.data = distances
    def get_distance(self, u, v):
        return self.data.get(u, {}).get(v, float("inf"))
    def __getitem__(self, key):
        return self.data[key]
    def add_distance(self, u, v, distance):
        if u not in self.data:
            self.data[u] = {}
        self.data[u][v] = distance
        
distance_oracle = DistanceOracle(D)



import math

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
def bfs_tree_of_S_rooted_x(graph, s, x):
    # Generate BFS tree rooted at x
    bfs_tree_s = nx.bfs_tree(graph, s)
    # Check if u is in the BFS tree rooted at x
    if x in bfs_tree_s.nodes:
        # Generate BFS tree roted at u from the BFS tree rooted at x
        bfs_tree_x = nx.bfs_tree(bfs_tree_s, x)
        bfs_tree_nodes = list(bfs_tree_x.nodes)
        return bfs_tree_nodes
    else:
        # print(f"Node {x} is not in the BFS tree rooted at {s}")
        return None
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



from itertools import combinations


def maximizer2(G ,x, y, V, d2, F_star , f):
    max_edges = set()
    edges_set = set()
    max_xy_edge = None
    max_xy_path = None
    max_xy_distance = float("-inf")
    
    max_xy_path_new = None
    
    max_edges2Avoid = []
    for e in G.edges:
            if (
                nx.has_path(G, x, e[0])
                and nx.has_path(G, y, e[1])
                and (
                    distance_oracle.get_distance(y, e[1]) >= d2
                    or distance_oracle.get_distance(y, e[0]) >= d2
                    and intact_from_failure_path(shortest_paths[(x, V)], F_star)
                    and intact_from_failure_tree(
                        bfs_tree_of_S_rooted_x(G, x, V), F_star
                    )
                )
                ):
                    max_edges2Avoid.append(e)

   
    for edge_combination in combinations(max_edges2Avoid, f):
            # Check if all edges in the combination exist in the graph
            if not all(G.has_edge(eu, ev) for eu, ev in edge_combination):
                continue

            # Store edge data and remove edges
            edge_data = [(eu, ev, G.get_edge_data(eu, ev)) for eu, ev in edge_combination]
            for eu, ev, _ in edge_data:
                G.remove_edge(eu, ev)
                # print(f"Removed edge: ({eu}, {ev})")
                    # Check if there is still a path from x to y
            if not nx.has_path(G, x, y):
            # Restore edges and continue
                for eu, ev, data in edge_data:
                    G.add_edge(eu, ev, **data)
                    # print(f"Restored edge: ({eu}, {ev})")
                continue

            # Use QUERY function to find path avoiding f edges
            path = nx.dijkstra_path(G, x, y, weight="weight")
            if not path or path == float("inf"):
                # Restore edges and continue
                for eu, ev, data in edge_data:
                    G.add_edge(eu, ev, **data)
                    # print(f"Restored edge: ({eu}, {ev})")
                continue

            # Find the path and its distance
            path_distance = sum(get_edge_weight(G, path[j], path[j + 1]) for j in range(len(path) - 1))

            # Check if the path satisfies the conditions
            
            if path_distance > max_xy_distance:
                max_xy_edge = edge_combination
                max_xy_path = path
                max_xy_distance = path_distance

            # Restore edges
            for eu, ev, data in edge_data:
                G.add_edge(eu, ev, **data)
                # print(f"Restored edge: ({eu}, {ev})")
 
    



    # print(max_xy_path)
# chandge max_xy_path to 3D-composable form
    if max_xy_path is not None:
        s = 0
        max_xy_path_new = []
        for i in range(len(max_xy_path) - 1):
            u = max_xy_path[s]
            v = max_xy_path[i +1]
            uv_distance = distance_oracle.get_distance(u, v)
            uv_distance_path = sum(
                get_edge_weight(G, max_xy_path[j], max_xy_path[j + 1])
                for j in range(s, i + 1)
            )
            # print(f"uv_distance:{uv_distance}")
            # print(f"uv_distance_path:{uv_distance_path}")
            # s_to_a_path = [u]
            if uv_distance != uv_distance_path:
                if i < (len(max_xy_path) - 2):
                    s_to_a_path = [u]
                    intermediate_edge = (v, max_xy_path[i + 2])
                    # print(f"intermediate:{intermediate_edge}")
                    # print(f"i:{i}")
                    s_to_a_path.append(max_xy_path[i])
                    max_xy_path_new.append(s_to_a_path)
                    max_xy_path_new.append(intermediate_edge)
                    s = i + 2
        max_xy_path_new.append([u, max_xy_path[len(max_xy_path) - 1]])
        if len(max_xy_path_new) == 1:
            max_xy_path_new = []
            max_xy_path_new.append(max_xy_path)
        if len(max_xy_path_new) == 3:
            max_xy_path_new = []
            max_xy_path_new.append(max_xy_path)
    return max_xy_edge, max_xy_path_new



from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import networkx as nx
import threading

f =2

# Initialize a dictionary to store the maximizer output
maximizer_dict2 = {}

# Store the maximizer function reference directly
maximizer_function = maximizer2  # Replace 'maximizer21' with the actual function name

# Collect errors to print after the loop
errors = []

# Define a lock
lock = threading.Lock()

# Define a function to process a single pair of nodes
def process_pair(x, y, V, d2, F_star, f):
    try:
        # Make a copy of the graph
        G_copy = G.copy()
        result = maximizer_function(G_copy, x, y, V, d2, F_star, f)
        if result is not None:
            max_edge, max_path = result
            return (x, y, V, d2, tuple(F_star), f), (max_edge, max_path)
    except nx.NetworkXNoPath:
        return (x, y, V , d2, tuple(F_star), f), None

# Use ThreadPoolExecutor to parallelize the computation
with ThreadPoolExecutor() as executor:
    futures = []
    for x in list(G.nodes):
        for y in list(G.nodes):
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
                            futures.append(
                                executor.submit(process_pair, x, y, d1, v, F_star, f)
                            )

    # Use tqdm to show progress for the futures
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing pairs"):
        key, result = future.result()
        if result is not None:
            with lock:
                # print(f"result for {key}: {result}")
                maximizer_dict2[key] = result
        else:
            errors.append(
                f"No path between {key[0]} and {key[1]} for d1: {key[2]}, d2: {key[3]}."
            )

# Print all errors after the loop
for error in errors:
    print(error)

# print(maximizer_dict21)

with open('maximizer_dict2.pkl', 'wb') as f:
    pickle.dump(maximizer_dict2, f)
