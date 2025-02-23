"""
Define network edges, where each edge connects two nodes with a delay.
The delay follows a normal distribution (mean, std).

Topology:
- Long path: 
1 -> 2 -> 3 -> 4 -> 5

- Shorter Paths: 
1 -> 3 -> 5
1 -> 2 -> 4 -> 5
1 -> 3 -> 4 -> 5 
1 -> 2 -> 3 -> 5

Each edge is defined as (node1, node2, {"mean": x, "std": y}).
"""

edges_with_normal_params = [
    (1, 2, {"mean": 10, "std": 2}),
    (2, 3, {"mean": 5, "std": 1}),
    (3, 4, {"mean": 10, "std": 2}),
    (4, 5, {"mean": 15, "std": 3}),
    
    # Shortcuts
    # Keeping 1 path for now for simplicity
    # (1, 3, {"mean": 15, "std": 3}),
    # (2, 4, {"mean": 20, "std": 4}),
    # (3, 5, {"mean": 25, "std": 5}),
]
