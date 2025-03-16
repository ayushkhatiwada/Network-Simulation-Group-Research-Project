"""
Model simple computer networks with nodes and edges
Each edge contains a normal distribution (mean, std), which represents the delay between the two nodes.
Delay measured in milliseconds.

Each edge is defined as (node1, node2, {"mean": x, "std": y}).
"""


"""
Topology:
1 -> 2
"""
one_edge_normal_params = [
    (1, 2, {"mean": 0.8, "std": 0.15}),
]


"""
Topology:
1 -> 2
1 -> 2

Two nodes (1 and 2) and two edges connecting them
"""
two_edges_normal_params = [
    (1, 2, {"mean": 0.8, "std": 0.15}), 
    (1, 2, {"mean": 0.5, "std": 0.1})
]
