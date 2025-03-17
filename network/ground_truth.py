import networkx as nx
import random
from numpy.random import gamma
import time
import numpy as np

from network.edges_with_gamma_params import edges_with_gamma_params
from network.edges_with_normal_params import edges_with_normal_params
from network.edges_with_log_normal_params import edges_with_log_normal_params


class GroundTruthNetwork:
    def __init__(self, distribution_type="gamma", edge_params=None):
        self.graph = nx.Graph()
        self.SOURCE = 1
        self.DESTINATION = 5
        self.graph.add_nodes_from(range(1, 6))

        self.distribution_type = distribution_type

        if distribution_type == "gamma":
            for u, v, params in edges_with_gamma_params:
                self.graph.add_edge(u, v, **params)
        elif distribution_type == "normal":
            for u, v, params in edges_with_normal_params:
                self.graph.add_edge(u, v, **params)
        elif distribution_type == "lognormal":
            for u, v, params in edges_with_log_normal_params:
                self.graph.add_edge(u, v, **params)
    

    # Yes it's bad code, but it's quick and dirty
    def sample_edge_delay(self, u, v):
        if self.distribution_type == "gamma":
            shape = self.graph[u][v]["shape"]
            scale = self.graph[u][v]["scale"]
            return gamma(shape, scale)
        elif self.distribution_type == "normal":
            mean = self.graph[u][v]['mean']
            std = self.graph[u][v]['std']
            delay = random.gauss(mean, std)
            return max(delay, 0.0)
        elif self.distribution_type == "lognormal":
            mu = self.graph[u][v]['mu']
            sigma = self.graph[u][v]['sigma']
            return np.random.lognormal(mu, sigma)
        else:
            raise ValueError(f"Unsupported distribution type: {self.distribution_type}")

    def simulate_traffic(self, num_flows, switches):
        for i in range(num_flows):
            try:
                path = nx.shortest_path(self.graph, source=self.SOURCE, target=self.DESTINATION)
            except nx.NetworkXNoPath:
                print(f"No path found for flow {i+1}.")
                continue

            packet = {
                'src_ip': f"10.0.0.{self.SOURCE}",
                'dst_ip': f"10.0.0.{self.DESTINATION}",
                'src_port': random.randint(1024, 65535),
                'dst_port': random.choice([80, 443]),
                'protocol': 'TCP'
            }
            print(f"\nFlow {i+1}: Path from node {self.SOURCE} to node {self.DESTINATION} via {path}")

            for idx, node in enumerate(path):
                if idx == 0:
                    role = "ingress"
                elif idx == len(path) - 1:
                    role = "egress"
                else:
                    role = "forward"

                switches[node].process_packet(packet, role)

                if idx < len(path) - 1:
                    next_node = path[idx + 1]
                    delay = self.sample_edge_delay(node, next_node)
                    print(f"  Link from {node} to {next_node} delay: {delay:.2f} ms")
                    time.sleep(delay / 1000.0)  

    def get_distribution_parameters(self, source, destination):
        """Get parameters with connection validation"""
        try:
            edge_data = self.graph[source][destination]
        except KeyError:
            raise ValueError(f"No connection between {source} and {destination}") from None
        
        return {
            'distribution_type': self.distribution_type,
            'mean': edge_data.get('mean', 0),
            'std': edge_data.get('std', 0),
            'shape': edge_data.get('shape', 0),
            'scale': edge_data.get('scale', 0),
            'mu': edge_data.get('mu', 0),
            'sigma': edge_data.get('sigma', 0)
        }  
