import networkx as nx
import random
from numpy.random import gamma


class TrueDelayNetwork:
    def __init__(self):
        self.graph = nx.Graph()
        self.SOURCE = 1
        self.DESTINATION = 5
        self.graph.add_nodes_from(range(1, 6))

        edges_with_gamma_params = [
            (1, 2, {"shape": 5.0, "scale": 2.0}),   # mean = 10.0
            (2, 3, {"shape": 6.0, "scale": 1.5}),   # mean = 9.0
            (3, 4, {"shape": 4.0, "scale": 2.5}),   # mean = 10.0
            (4, 5, {"shape": 5.0, "scale": 2.0}),   # mean = 10.0

            # Shortcut edges with slightly different characteristics
            (1, 3, {"shape": 3.0, "scale": 3.5}),   # mean = 10.5
            (2, 4, {"shape": 3.5, "scale": 3.0}),   # mean = 10.5
            (3, 5, {"shape": 4.0, "scale": 2.0}),   # mean = 8.0
        ]

        # use gamma distribution for delay at each edge
        self.is_distribution_gamma = True
        for u, v, params in edges_with_gamma_params:
            self.graph.add_edge(u, v, **params)

        # use normal distribution for delay at each edge
        # self.is_distribution_gamma = False
        # for u, v, params in edges_with_normal_params:
        #     self.graph.add_edge(u, v, **params)

    # Yes it's bad code, but it's quicky and dirty
    def sample_edge_delay(self, u, v):
        if self.is_distribution_gamma:
            shape = self.graph[u][v]["shape"]
            scale = self.graph[u][v]["scale"]
            return gamma(shape, scale)
        else:
            mean = self.graph[u][v]['mean']
            std = self.graph[u][v]['std']
            delay = random.gauss(mean, std)
            return max(delay, 0.0)
