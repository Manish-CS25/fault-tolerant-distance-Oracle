Implementation of Nearly Optimal Fault Tolerant Distance Oracle
This repository contains the implementation of a nearly optimal (f)-fault tolerant distance oracle for undirected weighted graphs, based on the work by Dey and Gupta (2024). The code was developed as part of the thesis "Implementation of Nearly Optimal Fault Tolerant Distance Oracle" by Manish Kumar Bairwa (23210055) at the Indian Institute of Technology Gandhinagar, defended on June 17, 2025.
Overview
The fault-tolerant distance oracle efficiently answers shortest path queries in the presence of up to (f) edge failures. It builds on theoretical guarantees with a space complexity of (O(f^4 n^2 \log^2(n W))) and query time of (O(c^{(f+1)^2} f^{8(f+1)^2} \log^{2(f+1)^2}(n W))), returning paths in ((f+1))-decomposable form. This implementation optimizes preprocessing using multiprocessing and handles edge cases like disconnected graphs.
Features

Implements Dey and Gupta’s (2024) fault-tolerant distance oracle.
Achieves up to 30× preprocessing speedup with 64-thread parallelization.
Supports robust handling of edge cases (e.g., disconnected graphs, critical faults).
Evaluated on diverse graph datasets (Watts-Strogatz, Graph Challenge 2017).
Written in Python 3.10 with NetworkX for graph operations.

Installation

Clone the repository:
git clone https://github.com/yourusername/fault-tolerant-distance-oracle.git
cd fault-tolerant-distance-oracle


Install dependencies:
pip install -r requirements.txt


Ensure you have Python 3.10 and a compatible environment (e.g., Ubuntu 22.04 LTS).
Required packages: networkx, numpy, multiprocessing.


Verify setup:
python -c "import networkx; print(networkx.__version__)"



Usage
Example: Running the Oracle

Prepare a graph file (e.g., graph.edgelist in NetworkX format).
Run the main script:python main.py --graph graph.edgelist --f 2 --source 0 --target 10


--f: Number of edge failures to tolerate.
--source and --target: Vertices for the query.
Output: Shortest path distance avoiding up to (f) failures.



Preprocessing
To precompute the oracle data structures:
python preprocess.py --graph graph.edgelist --f 2 --output oracle_data.pkl


Uses 64-thread parallelization for speedup.
Saves precomputed maximizers and jump sequences to oracle_data.pkl.

Querying
Load precomputed data and query:
python query.py --oracle oracle_data.pkl --source 0 --target 10 --faults edge1,edge2


--faults: List of edges to avoid (format: "u-v" for edge between u and v).

Datasets

Watts-Strogatz: Small-world networks ((n = 47, 100, 150, 200)).
Graph Challenge 2017: Stochastic block partitioning graphs ((n = 100, 500)).
Sample datasets are included in the data/ directory.

Performance

Space Usage: 5.75 MB ((n=47, f=1)) to 675 MB ((n=500, f=1)).
Query Time: 0.19 ms ((n=47, f=1)) to 210.97 ms ((n=100, f=2)).
Preprocessing Time: 300 s ((n=47, f=1)) to 372,240 s ((n=47, f=3)).

Thesis and Documentation

Full details are available in the thesis: Thesis PDF (link to be updated).
Refer to docs/ for additional notes on algorithms and edge case handling.

Contributing
Contributions are welcome! Please open an issue or submit a pull request for bugs, enhancements, or optimizations.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Inspired by Dey, D., and Gupta, M. (2024). Nearly Optimal Fault Tolerant Distance Oracle. arXiv:2402.12832.
Supported by IIT Gandhinagar and the thesis advisor.
