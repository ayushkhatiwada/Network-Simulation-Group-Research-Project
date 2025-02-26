"""
Define network edges, where each edge connects two nodes with a delay.
The delay follows a gamma distribution (shape, scale).

Topology:
- Long path: 
1 -> 2 -> 3 -> 4 -> 5

- Shorter Paths: 
1 -> 3 -> 5
1 -> 2 -> 4 -> 5
1 -> 3 -> 4 -> 5 
1 -> 2 -> 3 -> 5

Each edge is defined as (node1, node2, {"shape": x, "scale": y}).
"""

edges_with_gamma_params = [
    (1, 2, {"shape": 5.0, "scale": 2.0}),   # mean = 10.0
    (2, 3, {"shape": 6.0, "scale": 1.5}),   # mean = 9.0
    (3, 4, {"shape": 4.0, "scale": 2.5}),   # mean = 10.0
    (4, 5, {"shape": 5.0, "scale": 2.0}),   # mean = 10.0

    # Shortcuts
    # Keeping 1 path for now for simplicity
    # (1, 3, {"shape": 3.0, "scale": 3.5}),   # mean = 10.5
    # (2, 4, {"shape": 3.5, "scale": 3.0}),   # mean = 10.5
    # (3, 5, {"shape": 4.0, "scale": 2.0}),   # mean = 8.0
]


# TODO: Need to understand better what a gamma distribution is + what "shape" and "scale" are
# IDK if shape and scale values are appropriate here
