"""
Model a very simple computer network.
Two nodes, one edge connecting them.
Edge contains a lognormal distribution (mu, sigma).
Lognormal distribution represents the delay between the two nodes.
Delay is measured in milliseconds.

Topology:
1 -> 2

Each edge is defined as (node1, node2, {"mu": x, "sigma": y}).
Note:   mu is the mean of the log of the random variable
        sigma is the standard deviation of the log of the random variable
"""

edge_with_lognormal_params = [
    (1, 2, {"mu": -0.22, "sigma": 0.15}),
]

# Note: These parameters (mu=-0.22, sigma=0.15) give approximately:
# - median delay of exp(mu) = exp(-0.22) ≈ 0.8 ms
# - mean delay of exp(mu + sigma²/2) ≈ 0.81 ms
