import random

import networkx as nx

from single_edge_with_normal_distribution import edge_with_normal_params


class GroundTruthNetwork:
    def __init__(self):
        """
        Initialize a network with normal delay distribution.
        """
        self.graph = nx.Graph()
        self.SOURCE = 1
        self.DESTINATION = 2
        self.graph.add_nodes_from(range(1, 3))
        
        for u, v, params in edge_with_normal_params:
            self.graph.add_edge(u, v, **params)

    def sample_edge_delay(self, u, v):
        """
        Sample a delay value for the edge between nodes u and v
        based on normal distribution.
        
        Returns:
        --------
        float: A positive delay value in milliseconds
        """
        mean = self.graph[u][v]['mean']
        std = self.graph[u][v]['std']
        delay = random.gauss(mean, std)
        
        # Clamp values to be positive since negative delays don't make sense
        return max(delay, 0.0)

    def get_distribution_parameters(self, u, v):
        """
        Get the distribution parameters for the edge between nodes u and v.
        
        Returns:
        --------
        dict: A dictionary of distribution parameters
        """
        return {
            "mean": self.graph[u][v]['mean'],
            "std": self.graph[u][v]['std']
        }
