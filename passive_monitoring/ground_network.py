import random
import networkx as nx
import matplotlib.pyplot as plt

class Network:
    def __init__(self, num_nodes, num_edges, min_delay):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.min_delay = min_delay
        self.gamma_shape = 2 # this is so that delay can be modelled as a gamma distribution which is closer to reality than a normal/uniform distribution
        self.gamma_scale = 1 # TODO: these are currently default values, see if this can be changed 
        self.network = self.create_network()

    def create_network(self):
        graph = nx.gnm_random_graph(self.num_nodes, self.num_edges, directed=False)
        for u, v in graph.edges():
            delay = self.min_delay + random.gammavariate(self.gamma_shape, self.gamma_scale)
            graph[u][v]['weight'] = delay
        return graph

    def show_network(self):
        pos = nx.spring_layout(self.network)
        nx.draw(self.network, pos, with_labels=True, node_color='lightblue', node_size=500)
        labels = nx.get_edge_attributes(self.network, 'weight')
        formatted_labels = {k: f"{v:.1f}" for k, v in labels.items()}
        nx.draw_networkx_edge_labels(self.network, pos, edge_labels=formatted_labels)
        plt.title("Network Graph (Switches and Link Delays)")
        plt.show()

    def simulate_traffic(self, num_flows, switch):
        pass