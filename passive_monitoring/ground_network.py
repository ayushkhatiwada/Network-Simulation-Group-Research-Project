import random
import networkx as nx
import matplotlib.pyplot as plt
from switch import Switch
import time

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

    def simulate_traffic(self, num_flows, switches):
        nodes = list(self.network.nodes())
        for i in range(num_flows):
            src, dst = random.sample(nodes, 2)
            try:
                path = nx.shortest_path(self.network, source=src, target=dst, weight='weight')
            except nx.NetworkXNoPath:
                continue  
            packet = {
                'src_ip': f"10.0.0.{src+1}",
                'dst_ip': f"10.0.0.{dst+1}",
                'src_port': random.randint(1024, 65535),
                'dst_port': random.choice([80, 443]),
                'protocol': 'TCP'
            }

            print(f"\nFlow {i+1}: Path from node {src} to node {dst} via {path}")

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
                    link_delay = self.network[node][next_node]['weight']
                    print(f"  Link from {node} to {next_node} delay: {link_delay:.2f} ms")
                    time.sleep(link_delay / 1000.0) 

if __name__ == '__main__':
    net = Network(num_nodes=10, num_edges=20, min_delay=1.0)
    net.show_network()

    switches = {node: Switch(width=50, depth=5, seed=42) for node in net.network.nodes()}

    net.simulate_traffic(num_flows=5, switches=switches)
