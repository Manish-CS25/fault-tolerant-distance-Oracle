import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed

def maximizer1(G, x, y, d1, d2):
    max_edges = set()
    edges_set = set()
    max_xy_edge = None
    max_xy_path = None
    max_xy_distance = float("-inf")
    max_xy_path_new = None
    # xy_distance = distance_oracle.get_distance(x, y)
    # make the set of edges in xy path
    if nx.has_path(G, x, y):
        # Get the path and its length
        path = shortest_paths[(x, y)]
        # print(path)
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            edge = (u, v)
            edges_set.add(edge)
    # print(edges_set)
    # check max edges in edge list
    for u, v in edges_set:
        # Check if the distance from x to the edge and y to the edge are at least d1 and d2
        if (
            nx.has_path(G, x, u)
            and nx.has_path(G, y, v)
            and (
                distance_oracle.get_distance(x, u) >= d1
                and distance_oracle.get_distance(y, v) >= d2
            )
            # r (distance_oracle.get_distance(x, u) >= d2
            #     and distance_oracle.get_distance(y, v) >= d1)
        ):
            max_edge1 = (u, v)
            max_edges.add(max_edge1)
    max_edges = list(max_edges)
    # print(max_edges)
    max_edges_2 = []
    if max_edges == []:
        return [], [shortest_paths[(x, y)]]

    if len(max_edges) == 1:
        max_edges_2.append(max_edges[0][0])
        max_edges_2.append(max_edges[0][1])

    else:
        for i in range(len(max_edges)):
            for j in range(i + 1, len(max_edges)):
                max_edges_2.append((max_edges[i], max_edges[j]))
    # print(max_edges_2)

    G_copy = G.copy()
    if isinstance(max_edges_2[0], int):
        # max_xy_distance = float("-inf")
        if G_copy.has_edge(max_edges_2[0], max_edges_2[1]):
            G_copy.remove_edge(max_edges_2[0], max_edges_2[1])
            # Calculate the xy path distance
        D = preprocess_graph(G_copy)
        distance_oracle_new = DistanceOracle(D)
        if nx.has_path(G_copy, x, y):
            xy_path = nx.dijkstra_path(G_copy, x, y, weight="weight")
            # print(xy_path)
            max_uv_distance = distance_oracle_new.get_distance(x, y)
            # print(f"max_uv_distance:{max_uv_distance}")
            if max_uv_distance > max_xy_distance:
                max_xy_edge = [max_edges_2[0], max_edges_2[1]]
                max_xy_path = xy_path
                max_xy_distance = max_uv_distance

    else:
        # G_copy = G.copy()
        for e1 in max_edges_2:
            G_copy = G.copy()
            # print(f"e:{e1 , e2}")
            # max_xy_distance = float("-inf")

            if G_copy.has_edge(e1[0][0], e1[0][1]) and G_copy.has_edge(
                e1[1][0], e1[1][1]
            ):
                G_copy.remove_edge(e1[0][0], e1[0][1])
                G_copy.remove_edge(e1[1][0], e1[1][1])

            # Calculate the xy path distance
            D = preprocess_graph(G_copy)
            distance_oracle_new = DistanceOracle(D)
            if nx.has_path(G_copy, x, y):
                xy_path = nx.dijkstra_path(G_copy, x, y, weight="weight")
                # print(xy_path)
                max_uv_distance = distance_oracle_new.get_distance(x, y)
                # print(f"max_uv_distance:{max_uv_distance}")
                if max_uv_distance > max_xy_distance:
                    max_xy_edge = (e1[0], e1[1])
                    max_xy_path = xy_path
                    max_xy_distance = max_uv_distance

    # chandge max_xy_path to 3D-composable form
    if max_xy_path is not None:
        s = 0
        max_xy_path_new = []
        for i in range(len(max_xy_path) - 1):
            u = max_xy_path[s]
            v = max_xy_path[i + 1]
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


# ===============Storage of Maximizer =======================



# Initialize a dictionary to store the maximizer output
maximizer_dict1 = {}

# Store the maximizer function reference directly
maximizer_function = maximizer1  # Replace 'maximizer1' with the actual function name

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
        # print(f"key:{key}, result:{result}")
        if result is not None:
            maximizer_dict1[key] = result
        else:
            errors.append(f"No path between {key[0]} and {key[1]} for d1: {key[2]}, d2: {key[3]}.")

# Print all errors after the loop
for error in errors:
    print(error)
