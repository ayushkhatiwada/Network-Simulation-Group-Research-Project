"""
Define network edges, where each edge connects two nodes with a delay.
The delay follows a log-normal distribution (mu, sigma).

Topology:
- Long path: 
1 -> 2 -> 3 -> 4 -> 5

- Shorter Paths: 
1 -> 3 -> 5
1 -> 2 -> 4 -> 5
1 -> 3 -> 4 -> 5 
1 -> 2 -> 3 -> 5

Each edge is defined as (node1, node2, {"mu": x, "sigma": y}).
"""

edges_with_log_normal_params = [
    (1, 2, {"mu": 2.0, "sigma": 0.5}),   # median = e^mu ≈ 7.39, variability controlled by sigma
    (2, 3, {"mu": 1.8, "sigma": 0.4}),   # median ≈ 6.05
    (3, 4, {"mu": 2.2, "sigma": 0.6}),   # median ≈ 9.03
    (4, 5, {"mu": 2.1, "sigma": 0.5}),   # median ≈ 8.17
    
    # Shortcuts - commented out for simplicity
    # (1, 3, {"mu": 2.5, "sigma": 0.7}),
    # (2, 4, {"mu": 2.6, "sigma": 0.6}),
    # (3, 5, {"mu": 2.7, "sigma": 0.4}),
]