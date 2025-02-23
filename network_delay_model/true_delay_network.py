import networkx as nx
import random
from numpy.random import gamma

from edges_with_gamma_params import edges_with_gamma_params
from edges_with_normal_params import edges_with_normal_params


class TrueDelayNetwork:
    def __init__(self):
        self.graph = nx.Graph()
        self.SOURCE = 1
        self.DESTINATION = 5
        self.graph.add_nodes_from(range(1, 6))

        # use gamma distribution for delay at each edge
        self.is_distribution_gamma = True
        for u, v, params in edges_with_gamma_params:
            self.graph.add_edge(u, v, **params)

        # use normal distribution for delay at each edge
        # self.is_distribution_gamma = False
        # for u, v, params in edges_with_normal_params:
        #     self.graph.add_edge(u, v, **params)

    # Yes it's bad code, but it's quick and dirty
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
