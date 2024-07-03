import networkx as nx
import matplotlib.pyplot as plt


# Define the Edge class
class Edge:
    def __init__(self, u, v, weight):
        self.u = u
        self.v = v
        self.weight = weight


class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v, weight):
        if u not in self.graph:
            self.graph[u] = {}
        if v not in self.graph:
            self.graph[v] = {}
        self.graph[u][v] = weight
        self.graph[v][u] = weight  # add the edge (v, u) as well

    def get_edge_weight(self, u, v):
        return self.graph.get(u, {}).get(v, self.graph.get(v, {}).get(u, float("inf")))


def preprocess_graph(G):
    # assuming 'weight' is the name of the attribute for edge weights
    return dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))


# Define a small fraction
fraction = []
for i in range(19):
    fraction.append(1 / (2 ** (19 + i)))
# Generate edges with unique weights
edges = [
    Edge(1, 2, 1 + fraction[0]),
    Edge(1, 3, 2 + fraction[1]),
    Edge(2, 4, 3 + fraction[2]),
    Edge(2, 5, 4 + fraction[3]),
    Edge(3, 6, 4 + fraction[4]),
    Edge(3, 7, 6 + fraction[5]),
    Edge(4, 8, 7 + fraction[6]),
    Edge(4, 9, 8 + fraction[7]),
    Edge(5, 10, 9 + fraction[8]),
    Edge(5, 11, 10 + fraction[9]),
    Edge(6, 12, 11 + fraction[10]),
    Edge(6, 13, 12 + fraction[11]),
    Edge(7, 11, 13 + fraction[12]),
    Edge(7, 15, 14 + fraction[13]),
    Edge(8, 16, 15 + fraction[14]),
    Edge(9, 17, 16 + fraction[15]),
    Edge(10, 17, 17 + fraction[16]),
    Edge(11, 19, 18 + fraction[17]),
    Edge(12, 2, 19 + fraction[18]),
]
G = nx.Graph()
# Add edges to the graph
for edge in edges:
    # print(edge.u, edge.v, edge.weight)
    # sum_weight = sum_weight + int(edge.weight)
    G.add_edge(edge.u, edge.v, weight=edge.weight)
G = G.to_undirected()
# Preprocess the graph
D = preprocess_graph(G)
# Create a new figure with a larger size
plt.figure(figsize=(10, 10))
# Draw the graph using the spring_layout
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color="yellow", node_size=300, edge_color="gray")
# Draw edge labels
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
plt.show()


class DistanceOracle:
    def __init__(self, distances=None):
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


# Create an instance of the DistanceOracle class
distance_oracle = DistanceOracle(D)
import networkx as nx


def store_shortest_paths(G):
    shortest_paths = {}
    for u in G.nodes:
        for v in G.nodes:
            if u == v:
                shortest_paths[(u, v)] = [u]
            else:
                try:
                    path = nx.dijkstra_path(G, u, v, weight="weight")
                    shortest_paths[(u, v)] = path
                    # shortest_paths[(v, u)] = path[::-1]  # reverse path for (v, u)
                except nx.NetworkXNoPath:
                    shortest_paths[(u, v)] = None
                    # shortest_paths[(v, u)] = None
    return shortest_paths


shortest_paths = store_shortest_paths(G)
# Now you can access the shortest path between any pair of nodes like this:
import math


def nearest_power_of_2(x):
    if x <= 0:
        return 1  # Return 1 for non-positive input
    elif math.isinf(x):
        return float("inf")  # Return infinity for infinite input
    else:
        return 2 ** math.floor(math.log2(x))


def FINDJUMP(P, F):
    X = []  # Initialize X with s
    x = P[0]
    X.append(x)
    # F = list(F)
    # vertices = [F.u , F.v]
    if nearest_power_of_2(distance_oracle.get_distance(x, F[0])) < nearest_power_of_2(
        distance_oracle.get_distance(x, F[1])
    ):
        u = F[0]
    else:
        u = F[1]
    # u = min(F, key=lambda v: nearest_power_of_2(
    #     distance_oracle.get_distance(x, v)))
    # print(u)
    while True:
        # Find y, the first vertex on P[x, t] at distance >= max{1, (xu)^2} from x
        distance = max(1, nearest_power_of_2(distance_oracle.get_distance(x, u)))
        # print(distance)
        y = None
        for vertex in P[P.index(x) + 1 :]:
            if distance_oracle.get_distance(x, vertex) >= distance:
                y = vertex
                break
        if y is not None:
            X.append(y)
            x = y
        else:
            break
    return X


def get_edge_weight(G, u, v):
    if G.has_edge(u, v):
        return G[u][v]["weight"]
    else:
        return float("inf")


def maximizer(x, y, d1, d2):
    max_edges = set()
    edges_set = set()
    max_xy_edge = None
    max_xy_path = None
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
            distance_oracle.get_distance(x, u) >= d1
            and distance_oracle.get_distance(y, v) >= d2
        ) or (
            distance_oracle.get_distance(x, u) >= d2
            and distance_oracle.get_distance(y, v) >= d1
        ):
            max_edge1 = (u, v)
            max_edges.add(max_edge1)
    # print(max_edges)
    for u, v in max_edges:
        max_xy_distance = float("-inf")
        # Remove the (u, v) edge
        original_weight = get_edge_weight(G, u, v)
        # print(original_weight)
        if G.has_edge(u, v):
            G.remove_edge(u, v)
        # print(f"Removed edge: {u} - {v}")
        # Calculate the xy path distance
        D = preprocess_graph(G)
        distance_oracle_new = DistanceOracle(D)
        xy_path = nx.dijkstra_path(G, x, y, weight="weight")
        # print(xy_path)
        max_uv_distance = distance_oracle_new.get_distance(x, y)
        if max_uv_distance > max_xy_distance:
            max_xy_edge = (u, v)
            max_xy_path = xy_path
        # Add the (u, v) edge back to the graph
        # G.add_edge(u, v, weight=original_weight)
        G.add_weighted_edges_from([(u, v, original_weight)])
        D = preprocess_graph(G)
    # print(f"max_xy_path: {max_xy_path}")
    #    distance_oracle=DistanceOracle(D)
    # chandge max_xy_path to 2D-composable form
    if max_xy_path is not None:
        s = 0
        max_xy_path_new = []
        for i in range(len(max_xy_path) - 1):
            u = max_xy_path[s]
            v = max_xy_path[i + 1]
            uv_distance = distance_oracle.get_distance(u, v)
            uv_distance_path = sum(
                (
                    get_edge_weight(G, max_xy_path[j], max_xy_path[j + 1])
                    if get_edge_weight(G, max_xy_path[j], max_xy_path[j + 1])
                    is not None
                    else 0
                )
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
    return max_xy_edge, max_xy_path_new


def find_max_distance(G, distance_oracle):
    max_distance = float("-inf")
    for key1, value1 in distance_oracle.items():
        for key2, value2 in value1.items():
            if value2 > max_distance:
                max_distance = value2
    return max_distance


max_d_value = int(find_max_distance(G, D))
d1_d2_list = []
i = nearest_power_of_2((max_d_value))
while i >= 1:
    d1_d2_list.append(i)
    i //= 2
# print(f"d1_d2_list={d1_d2_list}")
# Initialize a dictionary to store the maximizer output
function_dict = {
    "maximizer": maximizer  # Ensure 'maximizer' is the correct function name
}
maximizer_dict = {}
# Ensure G is a NetworkX graph and d1_d2_list is defined
for x in G.nodes:
    for y in G.nodes:
        # Uncommented to skip pairs of the same node
        for d1 in d1_d2_list:
            for d2 in d1_d2_list:
                # print(f"x={x}, y={y}, d1={d1}, d2={d2}")
                try:
                    result = function_dict["maximizer"](x, y, d1, d2)
                    if result is not None:
                        max_edge, max_path = result
                        maximizer_dict[(x, y, d1, d2)] = (max_edge, max_path)
                except nx.NetworkXNoPath:
                    print(f"No path between {x} and {y}.")
                # Consider adding more specific exception handling here
print(maximizer(1, 16, 8, 2))


def is_valid_path(G, path):
    return all(G.has_edge(path[i], path[i + 1]) for i in range(len(path) - 1)) and (
        len(path) < 2 or G.has_edge(path[-2], path[-1])
    )


def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def QUERY(s, t, e):
    if e == []:
        return shortest_paths[(s, t)]
    # Check if the edge e is in the graph
    if not (G.has_edge(e.u, e.v) or G.has_edge(e.v, e.u)):
        print("The edge to avoid is not in the graph.")
        return [], float("inf")
    if distance_oracle.get_distance(s, e.u) > distance_oracle.get_distance(s, e.v):
        e.u, e.v = e.v, e.u
    # Check if there is a valid path between s and t
    if shortest_paths[(s, t)] is None and shortest_paths[(t, s)] is None:
        print("There is no valid path between the source and destination vertices.")
        return [], float("inf")
    # print(f"shortest_paths[(s, t)]={shortest_paths[(s, t)]}")
    JUMP_st = FINDJUMP(shortest_paths[(s, t)], [e.u, e.v])
    JUMP_ts = FINDJUMP(shortest_paths[(t, s)], [e.u, e.v])
    # print(f"JUMP_st={JUMP_st}, JUMP_ts={JUMP_ts}")
    final_shortest_path = []
    shortest_path_distance = float("inf")
    for x in JUMP_st:
        for y in JUMP_ts:
            # print(f"x={x}, y={y}")
            d1 = nearest_power_of_2(distance_oracle.get_distance(x, e.u))
            d2 = nearest_power_of_2(distance_oracle.get_distance(y, e.v))
            # print(f"d1={d1}, d2={d2}")
            e_star, xy_e_star = maximizer_dict[(x, y, d1, d2)]
            print(f"e_star={e_star}, xy_e_star={xy_e_star}")
            if xy_e_star is None:
                xy_e_star = []
            elif len(xy_e_star) > 2:
                # Flatten the list of lists
                xy_e_star = (
                    shortest_paths[(xy_e_star[0][0], xy_e_star[0][1])]
                    + [xy_e_star[1][0]]
                    + shortest_paths[
                        (
                            # print(f"xy_e_star={xy_e_star}")
                            xy_e_star[2][0],
                            xy_e_star[2][1],
                        )
                    ]
                )
            else:
                xy_e_star = shortest_paths[(xy_e_star[0][0], xy_e_star[0][1])]
            sx_path = shortest_paths[(s, x)]
            yt_path = shortest_paths[(y, t)]
            # print(f"sx_path={sx_path}, yt_path={yt_path}")
            # if sx_path is None or yt_path is None:
            #     continue  # Skip if there is no valid path from s to x or y to t
            if sx_path is None:
                sx_path = [x]
            if yt_path is None:
                yt_path = [y]
            # P = remove_duplicates(sx_path + xy_e_star + yt_path)
            P = sx_path + xy_e_star[1:-1] + yt_path
            # print(f"P={P}")
            # if is_valid_path(G, P):
            p_distance = sum(
                distance_oracle.get_distance(P[i], P[i + 1]) for i in range(len(P) - 1)
            )
            # print(f"p_distance={p_distance}")
            # else:
            # #     p_distance = float('inf')
            # s_u_distance = distance_oracle.get_distance(s, e.u)
            # v_t_distance = distance_oracle.get_distance(e.v, t)
            # u_v_distance = distance_oracle.get_distance(e.u, e.v)
            p_edges = [(P[i], P[i + 1]) for i in range(len(P) - 1)]
            if (
                (e.u, e.v) not in p_edges and (e.v, e.u) not in p_edges
            ) and p_distance < shortest_path_distance:
                final_shortest_path = P
                # print(f"shortest_path1={final_shortest_path}")
                shortest_path_distance = p_distance
    return final_shortest_path


print(QUERY(13, 19, Edge(6, 3, get_edge_weight(G, 6, 3))))
f = 2


def edge_in_path(p, F2):
    p_edges = [(p[i], p[i + 1]) for i in range(len(p) - 1)]
    for edge in F2:
        if (edge.u, edge.v) in p_edges or (edge.v, edge.u) in p_edges:
            return True
    return False


def bfs_tree_from_x_to_u(graph, x, u):
    # Generate BFS tree rooted at x
    bfs_tree_x = nx.bfs_tree(graph, x)
    # Check if u is in the BFS tree rooted at x
    if u in bfs_tree_x.nodes:
        # Generate BFS tree rooted at u from the BFS tree rooted at x
        bfs_tree_u = nx.bfs_tree(bfs_tree_x, u)
        bfs_tree_u_nodes = list(bfs_tree_u.nodes)
        return bfs_tree_u_nodes
    else:
        print(f"Node {u} is not in the BFS tree rooted at {x}")
        return None


def intact_from_failure_path(path, F):
    # Check if any edge in F is in the path
    # print(F)
    for edge in F:
        # print(edge.u, edge.v)
        if edge.u in path and edge.v in path:
            return False
    return True


def intact_from_failure_tree(T, F):
    # Check if any vertex in F is in the tree T
    for edge in F:
        # unpack edge into u and v
        if isinstance(edge, Edge):
            u, v = edge.u, edge.v
        else:
            u, v = edge
        if u in T or v in T:
            return False
    return True


def single_edge_in_path(p, F2):
    if p is not None:
        for edge in F2:
            # unpack edge into u and v
            if isinstance(edge, Edge):
                u, v = edge.u, edge.v
            else:
                u, v = edge
            # check if the edge is in the path
            if (u, v) in zip(p, p[1:]) or (v, u) in zip(p, p[1:]):
                return True
        return False


def maximizer1(x, y, d1, d2):
    max_edges = set()
    edges_set = set()
    max_xy_edge = None
    max_xy_path = None
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
            # or (distance_oracle.get_distance(x, u) >= d2
            #     and distance_oracle.get_distance(y, v) >= d1)
        ):
            max_edge1 = (u, v)
            max_edges.add(max_edge1)
    max_edges = list(max_edges)
    # print(max_edges)
    max_edges_2 = []
    for i in range(len(max_edges)):
        for j in range(i + 1, len(max_edges)):
            max_edges_2.append((max_edges[i], max_edges[j]))
    # print(max_edges_2)
    for e1 in max_edges_2:
        # print(f"e:{e1 , e2}")
        max_xy_distance = float("-inf")
        # Remove the (u, v) edg
        original_weight1 = get_edge_weight(G, e1[0][0], e1[0][1])
        original_weight2 = get_edge_weight(G, e1[1][0], e1[1][1])
        # print(original_weight)
        if G.has_edge(e1[0][0], e1[0][1]) and G.has_edge(e1[1][0], e1[1][1]):
            G.remove_edge(e1[0][0], e1[0][1])
            G.remove_edge(e1[1][0], e1[1][1])
            # print(f"Removed edge: {e1[0][0]} - {e1[0][1]}")
            # print(f"Removed edge: {e1[1][0]} - {e1[1][1]}")
        # Calculate the xy path distance
        D = preprocess_graph(G)
        distance_oracle_new = DistanceOracle(D)
        if nx.has_path(G, x, y):
            xy_path = nx.dijkstra_path(G, x, y, weight="weight")
            # print(xy_path)
            max_uv_distance = distance_oracle_new.get_distance(x, y)
            # print(f"max_uv_distance:{max_uv_distance}")
            if max_uv_distance > max_xy_distance:
                max_xy_edge = (e1[0], e1[1])
                max_xy_path = xy_path
                max_xy_distance = max_uv_distance
            # print(f"max_xy_distance:{max_xy_distance}")
        # Add the (u, v) edge back to the graph
        G.add_edge(e1[0][0], e1[0][1], weight=original_weight1)
        G.add_edge(e1[1][0], e1[1][1], weight=original_weight2)
        D = preprocess_graph(G)
    # print(max_xy_path)
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
    return max_xy_edge, max_xy_path_new


def maximizer2(x, y, d1, v, F):
    max_edges = set()
    edges_set = set()
    max_xy_edge = None
    max_xy_path = None
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
                and intact_from_failure_path(shortest_paths[(y, v)], F)
                and intact_from_failure_tree(bfs_tree_from_x_to_u(G, y, v), F)
            )
        ):
            max_edge1 = (u, v)
            max_edges.add(max_edge1)
        # print(max_edges)
    max_edges = list(max_edges)
    # print(max_edges)
    max_edges_2 = []
    for i in range(len(max_edges)):
        for j in range(i + 1, len(max_edges)):
            max_edges_2.append((max_edges[i], max_edges[j]))
    # print(max_edges_2)
    for e1 in max_edges_2:
        max_xy_distance = float("-inf")
        original_weight1 = get_edge_weight(G, e1[0][0], e1[0][1])
        original_weight2 = get_edge_weight(G, e1[1][0], e1[1][1])
        if G.has_edge(e1[0][0], e1[0][1]) and G.has_edge(e1[1][0], e1[1][1]):
            G.remove_edge(e1[0][0], e1[0][1])
            G.remove_edge(e1[1][0], e1[1][1])
        # Calculate the xy path distance
        D = preprocess_graph(G)
        distance_oracle_new = DistanceOracle(D)
        if nx.has_path(G, x, y):
            xy_path = nx.dijkstra_path(G, x, y, weight="weight")
            # print(xy_path)
            max_uv_distance = distance_oracle_new.get_distance(x, y)
        # print(f"max_uv_distance:{max_uv_distance}")
        if max_uv_distance > max_xy_distance:
            max_xy_edge = (e1[0], e1[1])
            max_xy_path = xy_path
            max_xy_distance = max_uv_distance
        # print(f"max_xy_distance:{max_xy_distance}")
        # Add the (u, v) edge back to the graph
        G.add_edge(e1[0][0], e1[0][1], weight=original_weight1)
        G.add_edge(e1[1][0], e1[1][1], weight=original_weight2)
        D = preprocess_graph(G)
    # print(max_xy_path)
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
    return max_xy_edge, max_xy_path_new


def maximizer21(x, y, u, d2, F):
    max_edges = set()
    edges_set = set()
    max_xy_edge = None
    max_xy_path = None
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
                distance_oracle.get_distance(y, u) >= d2
                and intact_from_failure_path(shortest_paths[(y, u)], F)
                and intact_from_failure_tree(bfs_tree_from_x_to_u(G, y, u), F)
            )
        ):
            max_edge1 = (u, v)
            max_edges.add(max_edge1)
        # print(max_edges)
    max_edges = list(max_edges)
    print(max_edges)
    max_edges_2 = []
    for i in range(len(max_edges)):
        for j in range(i + 1, len(max_edges)):
            max_edges_2.append((max_edges[i], max_edges[j]))
    print(max_edges_2)
    for e1 in max_edges_2:
        max_xy_distance = float("-inf")
        original_weight1 = get_edge_weight(G, e1[0][0], e1[0][1])
        original_weight2 = get_edge_weight(G, e1[1][0], e1[1][1])
        if G.has_edge(e1[0][0], e1[0][1]) and G.has_edge(e1[1][0], e1[1][1]):
            G.remove_edge(e1[0][0], e1[0][1])
            G.remove_edge(e1[1][0], e1[1][1])
        # Calculate the xy path distance
        D = preprocess_graph(G)
        distance_oracle_new = DistanceOracle(D)
        if nx.has_path(G, x, y):
            xy_path = nx.dijkstra_path(G, x, y, weight="weight")
            # print(xy_path)
            max_uv_distance = distance_oracle_new.get_distance(x, y)
        # print(f"max_uv_distance:{max_uv_distance}")
        if max_uv_distance > max_xy_distance:
            max_xy_edge = (e1[0], e1[1])
            max_xy_path = xy_path
            max_xy_distance = max_uv_distance
        # print(f"max_xy_distance:{max_xy_distance}")
        # Add the (u, v) edge back to the graph
        G.add_edge(e1[0][0], e1[0][1], weight=original_weight1)
        G.add_edge(e1[1][0], e1[1][1], weight=original_weight2)
        D = preprocess_graph(G)
    print(max_xy_path)
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
    return max_xy_edge, max_xy_path_new


def maximizer3(x, y, u, v, F_star):
    max_edges = set()
    edges_set = set()
    max_xy_edge = None
    max_xy_path = None
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
            and intact_from_failure_path(shortest_paths[(x, u)], F_star)
            and intact_from_failure_tree(bfs_tree_from_x_to_u(G, y, v), F_star)
            and intact_from_failure_path(shortest_paths[(y, v)], F_star)
            and intact_from_failure_tree(bfs_tree_from_x_to_u(G, y, v), F_star)
        ):
            max_edge1 = (u, v)
            max_edges.add(max_edge1)
    max_edges = list(max_edges)
    # print(max_edges)
    max_edges_2 = []
    for i in range(len(max_edges)):
        for j in range(i + 1, len(max_edges)):
            max_edges_2.append((max_edges[i], max_edges[j]))
    # print(max_edges_2)
    for e1 in max_edges_2:
        max_xy_distance = float("-inf")
        original_weight1 = get_edge_weight(G, e1[0][0], e1[0][1])
        original_weight2 = get_edge_weight(G, e1[1][0], e1[1][1])
        if G.has_edge(e1[0][0], e1[0][1]) and G.has_edge(e1[1][0], e1[1][1]):
            G.remove_edge(e1[0][0], e1[0][1])
            G.remove_edge(e1[1][0], e1[1][1])
        # Calculate the xy path distance
        D = preprocess_graph(G)
        distance_oracle_new = DistanceOracle(D)
        if nx.has_path(G, x, y):
            xy_path = nx.dijkstra_path(G, x, y, weight="weight")
            # print(xy_path)
            max_uv_distance = distance_oracle_new.get_distance(x, y)
        # print(f"max_uv_distance:{max_uv_distance}")
        if max_uv_distance > max_xy_distance:
            max_xy_edge = (e1[0], e1[1])
            max_xy_path = xy_path
            max_xy_distance = max_uv_distance
        # print(f"max_xy_distance:{max_xy_distance}")
        # Add the (u, v) edge back to the graph
        G.add_edge(e1[0][0], e1[0][1], weight=original_weight1)
        G.add_edge(e1[1][0], e1[1][1], weight=original_weight2)
        D = preprocess_graph(G)
    # print(max_xy_path)
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
    return max_xy_edge, max_xy_path_new


import networkx as nx


def maximizer3(x, y, u, v, F_star):
    max_edges = set()
    edges_set = set()
    max_xy_edge = None
    max_xy_path = None
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
            and intact_from_failure_path(shortest_paths[(x, u)], F_star)
            and intact_from_failure_tree(bfs_tree_from_x_to_u(G, y, v), F_star)
            and intact_from_failure_path(shortest_paths[(y, v)], F_star)
            and intact_from_failure_tree(bfs_tree_from_x_to_u(G, y, v), F_star)
        ):
            max_edge1 = (u, v)
            max_edges.add(max_edge1)
    max_edges = list(max_edges)
    # print(max_edges)
    max_edges_2 = []
    for i in range(len(max_edges)):
        for j in range(i + 1, len(max_edges)):
            max_edges_2.append((max_edges[i], max_edges[j]))
    # print(max_edges_2)
    for e1 in max_edges_2:
        max_xy_distance = float("-inf")
        original_weight1 = get_edge_weight(G, e1[0][0], e1[0][1])
        original_weight2 = get_edge_weight(G, e1[1][0], e1[1][1])
        if G.has_edge(e1[0][0], e1[0][1]) and G.has_edge(e1[1][0], e1[1][1]):
            G.remove_edge(e1[0][0], e1[0][1])
            G.remove_edge(e1[1][0], e1[1][1])
        # Calculate the xy path distance
        D = preprocess_graph(G)
        distance_oracle_new = DistanceOracle(D)
        if nx.has_path(G, x, y):
            xy_path = nx.dijkstra_path(G, x, y, weight="weight")
            # print(xy_path)
            max_uv_distance = distance_oracle_new.get_distance(x, y)
        # print(f"max_uv_distance:{max_uv_distance}")
        if max_uv_distance > max_xy_distance:
            max_xy_edge = (e1[0], e1[1])
            max_xy_path = xy_path
            max_xy_distance = max_uv_distance
        # print(f"max_xy_distance:{max_xy_distance}")
        # Add the (u, v) edge back to the graph
        G.add_edge(e1[0][0], e1[0][1], weight=original_weight1)
        G.add_edge(e1[1][0], e1[1][1], weight=original_weight2)
        D = preprocess_graph(G)
    # print(max_xy_path)
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
    return max_xy_edge, max_xy_path_new


def maximizer3(x, y, u, v, F_star):
    max_edges = set()
    edges_set = set()
    max_xy_edge = None
    max_xy_path = None
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
            and intact_from_failure_path(shortest_paths[(x, u)], F_star)
            and intact_from_failure_tree(bfs_tree_from_x_to_u(G, y, v), F_star)
            and intact_from_failure_path(shortest_paths[(y, v)], F_star)
            and intact_from_failure_tree(bfs_tree_from_x_to_u(G, y, v), F_star)
        ):
            max_edge1 = (u, v)
            max_edges.add(max_edge1)
    max_edges = list(max_edges)
    # print(max_edges)
    max_edges_2 = []
    for i in range(len(max_edges)):
        for j in range(i + 1, len(max_edges)):
            max_edges_2.append((max_edges[i], max_edges[j]))
    # print(max_edges_2)
    for e1 in max_edges_2:
        max_xy_distance = float("-inf")
        original_weight1 = get_edge_weight(G, e1[0][0], e1[0][1])
        original_weight2 = get_edge_weight(G, e1[1][0], e1[1][1])
        if G.has_edge(e1[0][0], e1[0][1]) and G.has_edge(e1[1][0], e1[1][1]):
            G.remove_edge(e1[0][0], e1[0][1])
            G.remove_edge(e1[1][0], e1[1][1])
        # Calculate the xy path distance
        D = preprocess_graph(G)
        distance_oracle_new = DistanceOracle(D)
        if nx.has_path(G, x, y):
            xy_path = nx.dijkstra_path(G, x, y, weight="weight")
            # print(xy_path)
            max_uv_distance = distance_oracle_new.get_distance(x, y)
        # print(f"max_uv_distance:{max_uv_distance}")
        if max_uv_distance > max_xy_distance:
            max_xy_edge = (e1[0], e1[1])
            max_xy_path = xy_path
            max_xy_distance = max_uv_distance
        # print(f"max_xy_distance:{max_xy_distance}")
        # Add the (u, v) edge back to the graph
        G.add_edge(e1[0][0], e1[0][1], weight=original_weight1)
        G.add_edge(e1[1][0], e1[1][1], weight=original_weight2)
        D = preprocess_graph(G)
    # print(max_xy_path)
    # chandge max_xy_path to 3D-composable form
    if max_xy_path is not None:
        s = 0
        max_xy_path_new = []
        for i in range(len(max_xy_path) - 1):
            u = max_xy_path[s]
            v = max_xy_path[i + 1]
            uv_distance = distance_oracle.get_distance(u, v)
            uv_distance_path = sum(
                (G, max_xy_path[j], max_xy_path[j + 1]) for j in range(s, i + 1)
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
    return max_xy_edge, max_xy_path_new


def FINDPATHS(s, t, F2):
    F_prime = []
    print(F_prime)
    P = []
    for i in range(1, f + 1):
        flag = 0
        if F_prime:
            u, v, weight = F_prime[i - 2].u, F_prime[i - 2].v, F_prime[i - 2].weight
            P_i = QUERY(s, t, Edge(u, v, weight))
            # print(f"u:{u}, v:{v}, weight:{weight}")
        else:
            P_i = QUERY(s, t, F_prime)
            # print(QUERY(12, 19, Edge(6, 3, get_edge_weight(G, 6, 3))))
            # print(f"P_i:{P_i}")
        P.append(P_i)
        weight_p_i = sum(
            distance_oracle.get_distance(P_i[i], P_i[i + 1])
            for i in range(len(P_i) - 1)
        )
        if F_prime:
            F2.remove(F_prime[i - 2])
        else:
            F2 = F2
        for e1 in F2:
            if F_prime:
                R1 = QUERY(s, e1.u, Edge(u, v, weight))
                R2 = QUERY(e1.v, t, Edge(u, v, weight))
            else:
                R1 = QUERY(s, e1.u, F_prime)
                R2 = QUERY(e1.v, t, F_prime)
            weight_R1 = sum(
                distance_oracle.get_distance(R1[i], R1[i + 1])
                for i in range(len(R1) - 1)
            )
            weight_R2 = sum(
                distance_oracle.get_distance(R2[i], R2[i + 1])
                for i in range(len(R2) - 1)
            )
            weight_e = distance_oracle.get_distance(e1.u, e1.v)
            if weight_R1 + weight_e + weight_R2 == weight_p_i:
                F_prime.append(e1)
                flag = 1
                # print(f"flag: {flag}")
                break
        if flag == 0:
            return P
    return P


def FINDJUMP2(P, F2):
    X = []  # Initialize X
    if not P:
        return []
    x = P[0]
    X.append(x)
    vertices = []
    for edge in F2:
        vertices += [edge.u, edge.v]
    u = min(
        vertices, key=lambda v: nearest_power_of_2(distance_oracle.get_distance(x, v))
    )
    # print(u)
    while True:
        distance = max(1, nearest_power_of_2(distance_oracle.get_distance(x, u)))
        # print(distance)
        y = None
        for vertex in P[P.index(x) + 1 :]:
            if distance_oracle.get_distance(x, vertex) >= distance:
                y = vertex
                break
        # print(y)
        if y is not None:
            X.append(y)
            x = y
        else:
            # Break if no progress can be made (y remains None)
            break
        # print(X)
    return X


def FIND_INTERMEDIATE3(x, y, u, v, F_star, F):
    F_star, xy_F_star = maximizer3(x, y, u, v, F_star)
    INTERMEDIATE = []
    if not any(edge in xy_F_star for edge in xy_F_star):
        PATH = xy_F_star
    else:
        PATH = []
    for z in F_star:
        # z satisfies the conditions of an intermediate vertex
        if edge_in_path(shortest_paths[(x, z)], F) and edge_in_path(
            shortest_paths[(z, y)], F
        ):
            INTERMEDIATE.append(z)
    return PATH, INTERMEDIATE


def FIND_INTERMEDIATE2(x, y, U, V, F_star, F):
    vertices = []
    for edge in F:
        vertices.append(edge.u)
        vertices.append(edge.v)
    d1 = min(nearest_power_of_2(distance_oracle.get_distance(x, a)) for a in vertices)
    d2 = min(nearest_power_of_2(distance_oracle.get_distance(y, b)) for b in vertices)
    if V == None:
        F_star, xy_F_star = maximizer21(x, y, U, d2, F)
    else:
        F_star, xy_F_star = maximizer2(x, y, d1, V, F)
    xy_F_star_path = []
    if xy_F_star is not None:
        if len(xy_F_star) == 1:
            xy_F_star_path = shortest_paths[(xy_F_star[0][0], xy_F_star[0][1])]
        elif len(xy_F_star) == 3:
            xy_F_star_path = (
                shortest_paths[(xy_F_star[0][0], xy_F_star[0][1])]
                + [xy_F_star[1][0]]
                + shortest_paths[(xy_F_star[2][0], xy_F_star[2][1])]
            )
        elif len(xy_F_star) == 5:
            xy_F_star_path = (
                shortest_paths[(xy_F_star[0][0], xy_F_star[0][1])]
                + [xy_F_star[1][0]]
                + shortest_paths[(xy_F_star[2][0], xy_F_star[2][1])]
                + [xy_F_star[3][0]]
                + shortest_paths[(xy_F_star[4][0], xy_F_star[4][1])]
            )
    INTERMEDIATE = []
    if not single_edge_in_path(xy_F_star_path, F):
        PATH = xy_F_star_path
    print(PATH)
    if U is not None:
        for u in F_star:
            # u satisfies the conditions of an x-clean vertex
            if intact_from_failure_path(
                shortest_paths[(x, u)], F
            ) and intact_from_failure_tree(bfs_tree_from_x_to_u(G, x, u), F):
                P, I = FIND_INTERMEDIATE3(x, y, u, V, F_star, F)
                PATH = min(P, PATH, key=len)
                INTERMEDIATE.extend(I)
            # u satisfies the conditions of an intermediate vertex
            if intact_from_failure_path(
                shortest_paths[(x, u)], F
            ) and intact_from_failure_path(shortest_paths[(y, u)], F):
                INTERMEDIATE.append(u)
    return PATH, INTERMEDIATE


def FIND_INTERMEDIATE1(x, y, r, F):
    # Extract vertices
    vertices = []
    for edge in F:
        vertices.append(edge.u)
        vertices.append(edge.v)
    # print(vertices)
    d1 = min(nearest_power_of_2(distance_oracle.get_distance(x, a)) for a in vertices)
    d2 = min(nearest_power_of_2(distance_oracle.get_distance(a, y)) for a in vertices)
    # print(d1, d2)
    F_star, xy_F_star = maximizer1(x, y, d1, d2)
    xy_F_star_path = []
    if xy_F_star is not None:
        # print(xy_F_star)
        # print(len(xy_F_star))
        if len(xy_F_star) == 1:
            xy_F_star_path = shortest_paths[(xy_F_star[0][0], xy_F_star[0][1])]
        elif len(xy_F_star) == 3:
            xy_F_star_path = (
                shortest_paths[(xy_F_star[0][0], xy_F_star[0][1])]
                + [xy_F_star[1][0]]
                + shortest_paths[(xy_F_star[2][0], xy_F_star[2][1])]
            )
        elif len(xy_F_star) == 5:
            xy_F_star_path = (
                shortest_paths[(xy_F_star[0][0], xy_F_star[0][1])]
                + [xy_F_star[1][0]]
                + shortest_paths[(xy_F_star[2][0], xy_F_star[2][1])]
                + [xy_F_star[3][0]]
                + shortest_paths[(xy_F_star[4][0], xy_F_star[4][1])]
            )
    # print(xy_F_star_path)
    # print(F_star, xy_F_star)
    PATH = []
    INTERMEDIATE = []
    path_distance = float("inf")
    if not single_edge_in_path(xy_F_star_path, F2):
        PATH = xy_F_star_path
    # print(PATH)
    if F_star is not None:
        F_star_vertices = [vetex for tuple in F_star for vetex in tuple]
        for u in F_star_vertices:
            # u satisfies the conditions of an x-clean vertex
            if intact_from_failure_path(
                shortest_paths[(x, u)], F
            ) and intact_from_failure_tree(bfs_tree_from_x_to_u(G, x, u), F):
                P1, I1 = FIND_INTERMEDIATE2(x, y, u, None, F_star, F)
                p1_distance = sum(
                    get_edge_weight(G, P1[i], P1[i + 1]) for i in range(len(P1) - 1)
                )
                if p1_distance < path_distance:
                    PATH = P1
                INTERMEDIATE.extend(I1)
            # u satisfies the conditions of a y-clean vertex
            elif intact_from_failure_path(
                shortest_paths[(y, u)], F
            ) and intact_from_failure_tree(bfs_tree_from_x_to_u(G, u, y), F):
                P2, I2 = FIND_INTERMEDIATE2(x, y, None, u, F_star, F)
                p2_distance = sum(
                    get_edge_weight(G, P2[i], P2[i + 1]) for i in range(len(P2) - 1)
                )
                if p2_distance < path_distance:
                    PATH = P2
                INTERMEDIATE.extend(I2)
            # u satisfies the conditions of an intermediate vertex
            elif intact_from_failure_path(
                shortest_paths[(x, u)], F
            ) and intact_from_failure_tree(bfs_tree_from_x_to_u(G, u, x), F):
                # print("else")
                INTERMEDIATE.append(u)
    return PATH, INTERMEDIATE


from itertools import product
from math import inf
from os import remove
from numpy import dtype


def edge_in_path(sx, xt, F):
    for edge in F:
        # print(f"edge:{edge}, sx:{sx}, xt:{xt}")
        if (edge.u and edge.v) in sx and (edge.u and edge.v) in xt:
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


# def QUERY2(s, t, r, F, findpath, findjump, find_intermediate1):
def QUERY2(s, t, r, F):
    # F_copy = F
    if r == 0:
        return float("inf")
    # print(FINDPATHS(s, t, F[:]))
    P = FINDPATHS(s, t, F[:])
    print(P)
    # if len(P) < r-1:
    #     return P
    jumps = []
    for i in range(len(P)):
        print(f"ith_paths={P[i]}")
        # for edge in F[:]:
        # print(f"edges:{edge.u, edge.v} ")
        # print(f"edges:{F[0].u, F[0].v} ")
        jump_st = FINDJUMP2(P[i], F[:])
        # use slicing to reverse the list
        jump_ts = FINDJUMP2(P[i][::-1], F)
        # print(f"jump_st={jump_st}, jump_ts={jump_ts}")
        jumps.append((jump_st, jump_ts))
    print(f"jumps={jumps}")
    PATH = []
    path_distance = float("inf")
    # print(f"r={r}")
    for i in range(len(jumps)):
        for j in range(len(jumps)):
            INTERMEDIATE = []
            for x, y in product(jumps[i][0], jumps[j][1]):
                # print(f"jump_st={jumps[i][0]}, jump_ts={jumps[j][1]}")
                print(f"x={x}, y={y}")
                # x satisfies the condition of an intermediate vertex
                sx = shortest_paths[(s, x)]
                sy = shortest_paths[(s, y)]
                xt = shortest_paths[(x, t)]
                yt = shortest_paths[(y, t)]
                print(f"sx={sx}, sy={sy}, xt={xt}, yt={yt}")
                if edge_in_path(sx, xt, F[:]):
                    #    print(edge in sx and edge in xt )
                    INTERMEDIATE.append(x)
                # y satisfies the condition of an intermediate vertex
                elif edge_in_path(sy, yt, F[:]):
                    INTERMEDIATE.append(y)
                else:
                    P, I = FIND_INTERMEDIATE1(x, y, r, F[:])
                    print(f"P={P}, I={I}")
                    P_prime = sx + P + yt
                    P_prime = remove_duplicates(P_prime)
                    print(f"P_prime={P_prime}")
                    P_prime_distance = sum(
                        distance_oracle.get_distance(P_prime[i], P_prime[i + 1])
                        for i in range(len(P_prime) - 1)
                        # if distance_oracle.get_distance(P_prime[i], P_prime[i + 1]) is not None
                    )
                    if path_distance > P_prime_distance:
                        PATH = P_prime
                        print(f"PATH={PATH}")
                        path_distance = P_prime_distance
                    INTERMEDIATE.extend(I)
                    INTERMEDIATE = remove_duplicates(INTERMEDIATE)
            print(f"INTERMEDIATE={INTERMEDIATE}")
            print(f"PATH={PATH}")
            for u in INTERMEDIATE:
                query_path = []
                print(f"u={u}")
                if (
                    QUERY2(s, u, r - 1, F[:]) != inf
                    and QUERY2(u, t, r - 1, F[:]) != inf
                ):
                    # Ensure QUERY2 returns a tuple, and when concatenating, convert operands to tuples if necessary
                    # Example fix inside QUERY2 or wherever the concatenation occurs
                    query_path_result1 = QUERY2(s, u, r - 1, F[:])
                    query_path_result2 = QUERY2(u, t, r - 1, F[:])
                    print(f"query_path_result1={query_path_result1}")
                    print(f"query_path_result2={query_path_result2}")
                    query_path.extend(query_path_result1)
                    query_path.extend(query_path_result2)
                print(f"query_path={query_path}")
                if query_path != inf:
                    print(f"query_path={query_path}")
                    query_path_distance = sum(
                        distance_oracle.get_distance(query_path[i], query_path[i + 1])
                        for i in range(len(query_path) - 1)
                        # if distance_oracle.get_distance(query_path[i], query_path[i + 1]) is not None
                    )
                if path_distance > query_path_distance:
                    PATH = query_path
                    path_distance = query_path_distance
    return PATH


F2 = [
    Edge(6, 3, get_edge_weight(G, 6, 3)),
    Edge(2, 5, get_edge_weight(G, 2, 5)),
]
# print(F2[0].u, F2[0].v, F2[1].u, F2[1].v)
print(QUERY2(13, 19, 3, F2))
