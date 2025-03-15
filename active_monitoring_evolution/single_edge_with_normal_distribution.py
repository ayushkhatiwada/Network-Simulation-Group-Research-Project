"""
Model a very simple computer network.
Two nodes, one edge connecting them.
Edge contains a normal distribution (mean, std), which represents the delay between the two nodes.
Delay measured in milliseconds.

Topology:
1 -> 2

Each edge is defined as (node1, node2, {"mean": x, "std": y}).
"""

edge_with_normal_params = [
    (1, 2, {"mean": 0.8, "std": 0.15}),
]
