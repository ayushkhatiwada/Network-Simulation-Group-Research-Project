import networkx as nx
import random
import numpy as np

from single_edge_with_normal_distribution import edge_with_normal_params
from single_edge_with_log_normal_distribution import edge_with_lognormal_params


class GroundTruthNetwork:
    def __init__(self, distribution_type="normal"):
        """
        Initialize a network with either normal or log-normal delay distribution.
        
        Parameters:
        -----------
        distribution_type : str
            Either "normal" or "lognormal" to specify the delay distribution
        """
        self.graph = nx.Graph()
        self.SOURCE = 1
        self.DESTINATION = 2
        self.graph.add_nodes_from(range(1, 3))
        self.distribution_type = distribution_type.lower()
        
        # Add edges based on selected distribution type
        if self.distribution_type == "normal":
            for u, v, params in edge_with_normal_params:
                self.graph.add_edge(u, v, **params)
        elif self.distribution_type == "lognormal":
            for u, v, params in edge_with_lognormal_params:
                self.graph.add_edge(u, v, **params)
        else:
            raise ValueError("Distribution type must be 'normal' or 'lognormal'")

    def sample_edge_delay(self, u, v):
        """
        Sample a delay value for the edge between nodes u and v
        based on the selected distribution type.
        
        Returns:
        --------
        float: A positive delay value in milliseconds
        """
        if self.distribution_type == "normal":
            mean = self.graph[u][v]['mean']
            std = self.graph[u][v]['std']
            delay = random.gauss(mean, std)
            
            # Clamp values to be positive since negative delays don't make sense
            return max(delay, 0.0)
        
        elif self.distribution_type == "lognormal":
            mu = self.graph[u][v]['mu']
            sigma = self.graph[u][v]['sigma']
            delay = np.random.lognormal(mean=mu, sigma=sigma)
            
            return delay

    def get_distribution_parameters(self, u, v):
        """
        Get the distribution parameters for the edge between nodes u and v.
        
        Returns:
        --------
        dict: A dictionary of distribution parameters
        """
        if self.distribution_type == "normal":
            return {
                "type": "normal",
                "mean": self.graph[u][v]['mean'],
                "std": self.graph[u][v]['std']
            }
        elif self.distribution_type == "lognormal":
            mu = self.graph[u][v]['mu']
            sigma = self.graph[u][v]['sigma']
            median = np.exp(mu)
            mean = np.exp(mu + (sigma**2)/2)

            return {
                "type": "lognormal",
                "mu": mu,
                "sigma": sigma,
                "median": median,
                "mean": mean
            }
