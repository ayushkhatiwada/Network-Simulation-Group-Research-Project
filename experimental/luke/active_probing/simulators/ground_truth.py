import random
import time
import numpy as np
import networkx as nx
from numpy.random import gamma

from .edges_with_normal_distribution import one_edge_normal_params, two_edges_normal_params
from passive_monitoring_evolution.switch_and_packet import Packet, Switch

class GroundTruthNetwork:
    def __init__(self, paths="1", distribution_type="normal"):
        """
        Initialise a network with a configurable delay distribution.
        """
        self.graph = nx.MultiGraph()
        self.SOURCE = 1
        self.DESTINATION = 2
        self.graph.add_nodes_from([1, 2])
        
        self.source_switch = Switch(self.SOURCE)
        self.destination_switch = Switch(self.DESTINATION)
        
        self.distribution_type = distribution_type
        
        path_params = {
            1: one_edge_normal_params,
            "1": one_edge_normal_params,
            2: two_edges_normal_params,
            "2": two_edges_normal_params
        }
        
        selected_params = path_params.get(paths, one_edge_normal_params)
        
        # Add the edges based on the selected parameters
        for u, v, params in selected_params:
            self.graph.add_edge(u, v, **params)

    def sample_edge_delay(self, u, v):
        """
        Sample a delay value for the edge between nodes u and v based on its distribution.

        If multiple edges exist between u and v, one is selected at random with equal probability.
        
        Returns:
        --------
        float: A positive delay value in milliseconds
        """
        # Retrieve the dictionary of edges between u and v
        edges = self.graph[u][v]
        
        # Randomly choose one edge key from the dictionary
        selected_key = random.choice(list(edges.keys()))
        edge_data = edges[selected_key]
        
        if self.distribution_type == "gamma":
            shape = edge_data.get("shape")
            scale = edge_data.get("scale")
            return gamma(shape, scale)
        elif self.distribution_type == "normal":
            mean = edge_data.get('mean')
            std = edge_data.get('std')
            delay = random.gauss(mean, std)
            return max(delay, 0.0)
        elif self.distribution_type == "lognormal":
            mu = edge_data.get('mu')
            sigma = edge_data.get('sigma')
            return np.random.lognormal(mu, sigma)
        else:
            raise ValueError(f"Unsupported distribution type: {self.distribution_type}")

    def simulate_traffic(self, num_flows, switches):
        """
        Simulate traffic flows through the network.
        """
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
                    
    def transmit_packet(self, packet, virtual_time):
        """
        Simulate the transmission of a packet through the network using virtual time.
        
        Parameters:
        -----------
        packet: Packet
            The packet to be transmitted
        virtual_time: float
            The current virtual time
        
        Returns:
        --------
        float: The updated virtual time after transmission
        """
        # Sample delay for the packet's path
        delay = self.sample_edge_delay(packet.source, packet.destination)
        
        # Calculate new virtual time
        new_virtual_time = virtual_time + delay / 1000.0
        
        # Simulate packet reception at the destination switch
        self.destination_switch.receive(packet, new_virtual_time)
        
        # Return the new virtual time so the simulator can update its own copy
        return new_virtual_time

    def get_distribution_parameters(self, source, destination):
        """
        Get parameters with connection validation.
        For MultiGraph, this returns parameters for the first edge found.
        """
        try:
            # For MultiGraph, we need to get the first edge's data (i.e when theres 2 edges between the same 2 nodes)
            # idk what the point of using multipgraph here is tbh
            edges = self.graph[source][destination]
            if not edges:
                raise ValueError(f"No connection between {source} and {destination}")
            
            first_key = list(edges.keys())[0]
            edge_data = edges[first_key]
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