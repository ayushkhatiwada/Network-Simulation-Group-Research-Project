import networkx as nx
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
        for u, v, params in edges_with_gamma_params:
            self.graph.add_edge(u, v, **params)

    # sample edge delay for edges with gamma distribution
    # if we want our graph to work with normally distributed edges
    # we need to create a new "Edges" class to make things cleaner
    def sample_edge_delay(self, u, v):
        shape = self.graph[u][v]["shape"]
        scale = self.graph[u][v]["scale"]
        return gamma(shape, scale)
