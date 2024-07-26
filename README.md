# Fault-Tolerant Distance Oracle for Weighted Graphs

## Introduction

We present an f-fault tolerant distance oracle for an undirected weighted graph where each edge has an integral weight from [1 . . . W]. Given a set F of f edges, as well as a source node s and a destination node t, our oracle returns the shortest path from s to t avoiding F in O((c f log(nW))^O(f^2)) time, where c > 1 is a constant. The space complexity of our oracle is O(f^4 n^2 log^2(nW)). For a constant f, our oracle is nearly optimal both in terms of space and time (barring some logarithmic factor).

In real-life networks, which can be represented as graphs, determining the distance between any two vertices is often necessary. This problem is commonly known as the “all-pair shortest path” problem in the literature. In the data structure version of this problem, the goal is to preprocess the graph to answer distance queries between any two vertices efficiently.

## Abstract

QUERY(s, t): Find the weight of the shortest path from s to t.
For unweighted graphs, the shortest path between all pairs of vertices can be found using the breadth-first search (BFS) algorithm, and the information can be stored in O(n^2) space. The query time is O(1). In the case of weighted graphs, there is extensive research on all-pair shortest path algorithms [Flo67, War62, Fre76, Dob90, Tak92, Han04, Tak04, Zwi04, Tak05, Cha05, Han06, Cha07, Han08, Cha08, Wil14]. Using the state-of-the-art algorithm, we can construct a data structure of size O(n^2) with O(1) query time.

However, real-life networks are prone to failures. Thus, we need to find the distance between vertices, avoiding certain edges or vertices, known as faults in the literature. This paper focuses on edge faults. Let’s formally define the problem. We are given an undirected weighted graph G. We can preprocess G and construct some suitable data structures. These data structures are used to answer queries of the form:

QUERY(s, t, F): Find the shortest path (and its weight) from s to t avoiding the edge set F.

The algorithm that handles the aforementioned query is known as the query algorithm. It utilises the prepared data structures to respond to the query. This combination of data structures and the query algorithm is referred to as a distance oracle in the literature. Since we are dealing with faults, our oracle is known as f-fault-tolerant distance oracle when |F| ≤ f. A fault-tolerant distance oracle is evaluated based on its size, query time (the time required to answer a query), and the number of faults it can handle, i.e., the maximum size of F. In a naive approach, we could store all distances, avoiding all possible faults and return the distance upon query. However, this would consume significant space, albeit with a small query time. Hence, the primary research challenge in this field is to reduce the space while not significantly increasing the query time.

## Relevant Results

We summarise relevant results in this area in Table 1. When f = 1, the oracle presented in [DTCR08] is optimal. For f = 2, Duan and Pettie [DP09] designed a nearly optimal distance oracle (barring some polylog n factor). However, for f > 2, our understanding of this problem is limited. The results can be categorised into two groups: those with high query time but low space [WY13, vdBS19, KS23], and those with low query time but high space [DR22]. An important question in this field is to develop an oracle that minimises both space (as in [KS23]) and query time (as in [DR22]). In fact, this question is explicitly raised in [DR22], and we quote it here (with minor notation changes).

| Faults | Space | Query time | Remarks | Reference |
|--------|-------|------------|---------|-----------|
| 1      | ˜O(n^2) | O(1) | ˜O hides polylog n factor. | [DTCR08] |
| 2      | ˜O(n^2) | O(log n) | - | [DP09] |
| f      | ˜O(n^3−α) | ˜O(n^2−(1−α)/f) | α ∈ [0, 1] when the preprocessing time is O(W n^3.376−α) and edge weights are integral and in the range [−W . . . W]. | [WY13] |
| f      | ˜O(W n^2+α) | O(W n^2−α f^2 + W n f^ω) | α ∈ [0, 1], edge weights are integral and in the range [−W . . . W] and ω is the matrix multiplication exponent [CW87, Sto10, Wil12, Gal14, AW21] | [vdBS19] |
| f      | O(n^4) | O(f^O(f)) | - | [DR22] |
| f      | O(n^2) | ˜O(n f) | Edge weights are in the range [1 . . . W] | [KS23] |

## Example Usage

Here is an example of how to use the `QUERY2` function:

```python
# Import necessary functions and classes
from your_module import Edge, get_edge_weight, QUERY2

# Define the graph and edges
G = ...  # Your graph definition here
F3 = [Edge(6, 3, get_edge_weight(G, 6, 3)), Edge(4, 9, get_edge_weight(G, 4, 9))]

# Query the graph
ans = QUERY2(6, 9, 3, F3)
print(ans)
