import networkx as nx
import random
from numpy.random import gamma

class AsymmetricNetwork:
    def __init__(self):
        self.graph = nx.DiGraph()
        nodes = range(1, 6)
        self.graph.add_nodes_from(nodes)
        self.source = 1
        possible_dest = [n for n in nodes if n != self.source]
        self.hidden_destination = random.choice(possible_dest)

        edges = [
            (1,2), (2,3), (3,4), (4,5),
            (2,1), (3,2), (4,3), (5,4),
            (1,3), (3,1), (2,4), (4,2),
            (3,5), (5,3)
        ]
        for (u,v) in edges:
            shape = random.uniform(1.5,3.0)
            scale = random.uniform(1.0,2.0)
            self.graph.add_edge(u, v, shape=shape, scale=scale)

    def sample_edge_delay(self, u, v):
        d = self.graph[u][v]
        return gamma(d['shape'], d['scale'])
    
    def get_path_and_delay(self, src, dst):
        try:
            path = nx.shortest_path(self.graph, src, dst)
            delay = 0.0
            for i in range(len(path)-1):
                delay += self.sample_edge_delay(path[i], path[i+1])
            return path, delay
        except nx.NetworkXNoPath:
            return None, None

def detect_asymmetry(net):
    fwd_path, fwd_delay = net.get_path_and_delay(net.source, net.hidden_destination)
    rev_path, rev_delay = net.get_path_and_delay(net.hidden_destination, net.source)
    if not fwd_path or not rev_path:
        print("No valid forward or reverse path found.")
        return
    print(f"Forward path: {fwd_path} (Delay={fwd_delay:.2f})")
    print(f"Reverse path: {rev_path} (Delay={rev_delay:.2f})")
    if fwd_path != list(reversed(rev_path)) or abs(fwd_delay - rev_delay) > 1e-3:
        print("Asymmetry detected.")
    else:
        print("Paths appear symmetric.")

def main():
    net = AsymmetricNetwork()
    print(f"Source={net.source}, hidden_destination={net.hidden_destination}")
    detect_asymmetry(net)

if __name__ == "__main__":
    main()
