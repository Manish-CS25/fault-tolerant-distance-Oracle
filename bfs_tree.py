import heapq
import networkx as nx
import pickle

with open('graph.pkl', 'rb') as f:  
    G = pickle.load(f)

def bfs_tree_with_sorting(G, root):
    tree = nx.DiGraph()
    visited = set()
    priority_queue = [(0, root, None)]  # (distance, node, parent)

    while priority_queue:
        distance, node, parent = heapq.heappop(priority_queue)
        if node not in visited:
            visited.add(node)
            if parent is not None:
                tree.add_edge(parent, node)
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    edge_weight = G[node][neighbor].get('weight', 1)  # Default weight is 1 if not specified
                    heapq.heappush(priority_queue, (distance + edge_weight, neighbor, node))
    return tree


def bfs_tree_of_S_rooted_x(graph, s, x):
    # Generate BFS tree rooted at x
    bfs_tree_s = bfs_tree_with_sorting(graph, s)
    # Check if u is in the BFS tree rooted at x
    if x in bfs_tree_s.nodes:
        # BFS tree rooted at u from the BFS tree rooted at s
        bfs_tree_x = bfs_tree_with_sorting(bfs_tree_s, x)   
        bfs_tree_nodes = set(list(bfs_tree_x.nodes))
        return bfs_tree_nodes
    else:
        # print(f"Node {x} is not in the BFS tree rooted at {s}")
        return None
    
txu_dict = {}   
for x in G.nodes:
    for u in G.nodes:
        bfs_tree_nodes = bfs_tree_of_S_rooted_x(G, x, u)  
        if bfs_tree_nodes is not None:
            txu_dict[(x, u)] = bfs_tree_nodes
        else:
            txu_dict[(x, u)] = {}     
        print(f"txu_dict[{x}, {u}] = {txu_dict[(x, u)]}")
# Save the dictionary to a file
with open('txu_dict.pkl', 'wb') as f:
    pickle.dump(txu_dict, f)
    
            


    
