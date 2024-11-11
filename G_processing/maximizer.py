 
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle


def maximizer(G, x, y, d1, d2):
    max_edges = set()
    edges_set = set()
    max_xy_edge = None
    max_xy_path = None
    max_xy_path_new = None

    # Cache distance calculations
    distance_cache = {}

    def get_distance_cached(node1, node2):
        if (node1, node2) not in distance_cache:
            distance_cache[(node1, node2)] = distance_oracle.get_distance(node1, node2)
        return distance_cache[(node1, node2)]

    if nx.has_path(G, x, y):
        path = shortest_paths[(x, y)]
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            edges_set.add((u, v))

    for u, v in edges_set:
        if get_distance_cached(x, u) >= d1 and get_distance_cached(y, v) >= d2:
            max_edges.add((u, v))

    max_xy_distance = float('-inf')
    for u, v in max_edges:
        G_copy = G.copy()
        if G_copy.has_edge(u, v):
            G_copy.remove_edge(u, v)
        D = preprocess_graph(G_copy)
        distance_oracle_new = DistanceOracle(D)
        if nx.has_path(G_copy, x, y):
            xy_path = nx.dijkstra_path(G_copy, x, y, weight='weight')
            max_uv_distance = distance_oracle_new.get_distance(x, y)
            if max_uv_distance > max_xy_distance:
                max_xy_edge = (u, v)
                max_xy_path = xy_path
                max_xy_distance = max_uv_distance

    if max_xy_path is not None:
        s = 0
        max_xy_path_new = []
        for i in range(len(max_xy_path) - 1):
            u = max_xy_path[s]
            v = max_xy_path[i + 1]
            uv_distance = get_distance_cached(u, v)
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
        max_xy_path_new.append([u, max_xy_path[-1]])

        if len(max_xy_path_new) == 1:
            max_xy_path_new = [max_xy_path]

    return max_xy_edge, max_xy_path_new




# Initialize a dictionary to store the maximizer output
maximizer_dict = {}

# Store the maximizer function reference directly
maximizer_function = maximizer  # Replace 'maximizer' with the actual function name

# Collect errors to print after the loop
errors = []

# Define a function to process a single pair of nodes
def process_pair(G, x, y, d1, d2):
    try:
        result = maximizer_function(G, x, y, d1, d2)
        if result is not None:
            max_edge, max_path = result
            return (x, y, d1, d2), (max_edge, max_path)
    except nx.NetworkXNoPath:
        return (x, y, d1, d2), None

# Use ThreadPoolExecutor to parallelize the computation
with ThreadPoolExecutor() as executor:
    futures = []
    for x in G.nodes:
        for y in G.nodes:
            if x != y:
                for d1 in d1_d2_list:
                    for d2 in d1_d2_list:
                        futures.append(executor.submit(process_pair, G, x, y, d1, d2))

    for future in as_completed(futures):
        key, result = future.result()
        if result is not None:
            maximizer_dict[key] = result
        else:
            errors.append(f"No path between {key[0]} and {key[1]} for d1: {key[2]}, d2: {key[3]}.")

# Print all errors after the loop
for error in errors:
    print(error)


with open('maximizer_dict.pkl', 'wb') as f:
    pickle.dump(maximizer_dict, f)
