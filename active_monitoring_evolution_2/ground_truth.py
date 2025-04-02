import random
import networkx as nx

from edges_with_normal_distribution import one_edge_normal_params, two_edges_normal_params


class GroundTruthNetwork:
    def __init__(self, paths="1"):
        """
        Initialise a network with a normal delay distribution.
        """
        self.graph = nx.MultiGraph()
        self.SOURCE = 1
        self.DESTINATION = 2
        self.graph.add_nodes_from([1, 2])
        
        path_params = {
            1: one_edge_normal_params,
            "1": one_edge_normal_params,
            2: two_edges_normal_params,
            "2": two_edges_normal_params
        }
        
        self.selected_params = path_params.get(paths, one_edge_normal_params)
        
        # Add the edges based on the selected parameters.
        for u, v, params in self.selected_params:
            self.graph.add_edge(u, v, **params)

    def sample_edge_delay(self, u, v):
        """
        Sample a delay value for the edge between nodes u and v based on its normal distribution.
        If multiple edges exist between u and v, one is selected at random with equal probability.
        
        Returns:
            float: A positive delay value in milliseconds.
        """
        edges = self.graph[u][v]
        selected_key = random.choice(list(edges.keys()))
        edge_data = edges[selected_key]
        
        mean = edge_data['mean']
        std = edge_data['std']
        delay = random.gauss(mean, std)
        return max(delay, 0.0)

    def get_single_edge_distribution_parameters(self, u, v):
        """
        Return the mean and std for the only edge between u and v.
        """
        edges = self.graph[u][v]

        assert len(edges) == 1, "Expected only one edge between nodes."

        edge_data = next(iter(edges.values()))
        return {
            "mean": edge_data["mean"],
            "std": edge_data["std"]
        }

    def get_multi_edge_distribution_parameters(self, u, v):
        """
        Return a list of dictionaries with mean and std for all edges between u and v.
        """
        edges = self.graph[u][v]
        return [
            {"mean": edge["mean"], "std": edge["std"]}
            for edge in edges.values()
        ]
