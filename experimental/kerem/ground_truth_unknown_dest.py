import networkx as nx
import random
import time
from numpy.random import gamma

class SimulatedNetwork:
    def __init__(self):
        """
        Create a small graph and pick a random hidden destination
        not revealed to the 'prober'.
        """
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(1, 7))
        
        self.graph.add_edge(1, 2, shape=2.0, scale=2.0)
        self.graph.add_edge(2, 3, shape=2.0, scale=1.0)
        self.graph.add_edge(2, 4, shape=3.0, scale=2.0)
        self.graph.add_edge(3, 5, shape=2.0, scale=2.0)
        self.graph.add_edge(4, 6, shape=3.0, scale=1.5)
        self.graph.add_edge(5, 6, shape=2.0, scale=2.5)
        
        self.source = 1
        possible_destinations = [n for n in self.graph.nodes if n != self.source]
        self.hidden_destination = random.choice(possible_destinations)
        
    def sample_edge_delay(self, u, v):
        edge_data = self.graph[u][v]
        shape = edge_data['shape']
        scale = edge_data['scale']
        return gamma(shape, scale)
    
    def get_next_hop(self, ttl):
        try:
            path = nx.shortest_path(self.graph, self.source, self.hidden_destination)
        except nx.NetworkXNoPath:
            return None 
        if ttl >= len(path):
            return path[-1]
        else:
            return path[ttl] 
    
    def measure_path_delay(self, path):
        total_delay = 0.0
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i+1]
            delay = self.sample_edge_delay(u, v) 
            total_delay += delay
        return total_delay


def incremental_ttl_discovery(network, max_ttl=10):
    discovered_path = []
    reached_destination = False
    for ttl in range(1, max_ttl + 1):
        hop_node = network.get_next_hop(ttl)
        if hop_node is None:
            print(f"TTL={ttl}: No path (dead end).")
            break
        if not discovered_path or discovered_path[-1] != hop_node:
            discovered_path.append(hop_node)
        print(f"TTL={ttl}: Reached node {hop_node}.")
        if hop_node == network.hidden_destination:
            reached_destination = True
            break
    if reached_destination:
        discovered_path.insert(0, network.source)
        return discovered_path
    else:
        return None

def multi_probe_ttl_discovery(network, max_ttl=10, probes_per_ttl=3):
    discovered_path = []
    for ttl in range(1, max_ttl + 1):
        hop_set = set()
        for _ in range(probes_per_ttl):
            hop = network.get_next_hop(ttl)
            if hop:
                hop_set.add(hop)
        if not hop_set:
            print(f"TTL={ttl}: No path.")
            break
        print(f"TTL={ttl}: Reached {hop_set}.")
        if network.hidden_destination in hop_set:
            discovered_path = list(nx.shortest_path(network.graph, network.source, network.hidden_destination))
            return discovered_path
    return None

def main():
    net = SimulatedNetwork()
    print(f"Source is node {net.source}, hidden destination is unknown.\n")
    
    path = multi_probe_ttl_discovery(net, max_ttl=10, probes_per_ttl=3)
    
    if path is not None:
        print(f"\nDiscovered Path: {path}")
        delay = net.measure_path_delay(path)
        print(f"Measured end-to-end delay = {delay:.2f} ms\n")
    else:
        print("\nCould not discover the destination within max TTL.")

if __name__ == "__main__":
    main()
