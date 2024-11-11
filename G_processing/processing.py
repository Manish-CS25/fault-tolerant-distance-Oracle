import networkx as nx
import matplotlib.pyplot as plt
from random import randint
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



# Read the file and create edges with random weights
file_path = 'small_dataset.txt'  # Replace with the actual file path
edges = []
with open(file_path, 'r') as file:
    file_contents = file.readlines()
    num_lines = len(file_contents)
    for i, line in enumerate(file_contents):
        node1, node2 = map(int, line.split())
        weight = float(randint(1, 99)) + (1 / (2 ** (num_lines + i + 1)))
    # Ensure the weight is displayed with full precision
        # print(f"Weight: {weight:.50f}")
        edges.append(Edge(node1, node2, weight))

# Create a NetworkX graph and add edges
G = nx.Graph()
for edge in edges:
    G.add_edge(edge.u, edge.v, weight=edge.weight)

# Preprocess the graph
D = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))

# Create a new figure with a larger size
plt.figure(figsize=(60, 60))

# Draw the graph using the spring_layout
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color="yellow", node_size=300, edge_color="gray")

# Draw edge labels
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

plt.show()

# Store G and D locally using pickle
with open('graph.pkl', 'wb') as f:
    pickle.dump(G, f)

with open('distances.pkl', 'wb') as f:
    pickle.dump(D, f)
    
# store the shortest path in a dictionary
def store_shortest_paths(G):
    # Use NetworkX's built-in function to compute all pairs shortest paths
    all_pairs_paths = dict(nx.all_pairs_dijkstra_path(G, weight='weight'))
    
    # Flatten the dictionary to match the desired output format
    shortest_paths = {}
    for u, paths in all_pairs_paths.items():
        for v, path in paths.items():
            shortest_paths[(u, v)] = path
    
    return shortest_paths

# Example usage
shortest_paths = store_shortest_paths(G)

with open('shortest_paths.pkl', 'wb') as f:
    pickle.dump(shortest_paths, f)
