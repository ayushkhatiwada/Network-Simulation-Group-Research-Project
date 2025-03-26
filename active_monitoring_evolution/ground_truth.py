import time
import random
import threading
import networkx as nx
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from edges_with_normal_distribution import one_edge_normal_params, two_edges_normal_params
from passive_monitoring.passive_monitoring_interface.switch_and_packet import Packet, Switch

class GroundTruthNetwork:
    def __init__(self, paths="1"):
        """
        Initialise a network with a normal delay distribution.
        """
        self.graph = nx.MultiGraph()
        self.SOURCE = 1
        self.DESTINATION = 2
        self.graph.add_nodes_from([1, 2])
        self.source_switch = Switch(1)
        self.destination_switch = Switch(2)
        
        path_params = {
            1: one_edge_normal_params,
            "1": one_edge_normal_params,
            2: two_edges_normal_params,
            "2": two_edges_normal_params
        }
        
        selected_params = path_params.get(paths, one_edge_normal_params)
        
        # Add the edges based on the selected parameters.
        for u, v, params in selected_params:
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

    def transmit_packet(self, packet):
        delay = self.sample_edge_delay(packet.source, packet.destination)
        # Simulate transit delay (delay is in milliseconds, so we sleep for delay/1000 seconds).
        time.sleep(delay / 1000.0)
        self.destination_switch.receive(packet)

    def simulate_traffic(self, duration_seconds=10, avg_interarrival_ms=50):
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            packet = Packet(self.source_switch.switch_id, self.destination_switch.switch_id)
            threading.Thread(target=self.transmit_packet, args=(packet,)).start()
            interarrival = random.expovariate(1.0 / avg_interarrival_ms)
            time.sleep(interarrival / 1000.0)

    def get_distribution_parameters(self, u, v):
        """
        Compute and return the average delay parameters (mean and std) for edges connecting u and v.
        This function aggregates the delay parameters from all edges between the given nodes.
        
        Returns:
            dict: A dictionary with keys "mean" and "std".
        """
        edges = self.graph[u][v]
        means = []
        stds = []
        for key in edges:
            edge_data = edges[key]
            means.append(edge_data.get('mean', 0))
            stds.append(edge_data.get('std', 0))
        if means:
            avg_mean = sum(means) / len(means)
            avg_std = sum(stds) / len(stds)
        else:
            avg_mean, avg_std = 0, 0
        return {"mean": avg_mean, "std": avg_std}
