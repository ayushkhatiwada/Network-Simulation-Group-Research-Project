from network_delay_model.simulator import Simulator
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Set


class PathProber:
    def __init__(self):
        self.simulator = Simulator()
        self.network = self.simulator.network
        self.paths = {}  # Store different paths in the network
        self._discover_paths()
        
    def _discover_paths(self):
        G = self.network.graph
        # build all paths
        for source in G.nodes():
            for target in G.nodes():
                if source != target:
                    paths = list(nx.all_simple_paths(G, source, target))
                    if paths:
                        if (source, target) not in self.paths:
                            self.paths[(source, target)] = []
                        self.paths[(source, target)].extend(paths)
    
    def probe_path(self, source, target, num_probes = 100):
        # probe frm src to dest, record delays
        custom_sim = Simulator()
        custom_sim.network.SOURCE = source
        custom_sim.network.DESTINATION = target
        
        # send at "times"
        departure_times = [float(i) for i in range(num_probes)]
        results = custom_sim.send_multiple_probes(departure_times)
        
        # get the delays
        delays = np.array([result[2] for result in results])
        return delays
    
    def probe_all_paths(self, num_probes = 100):
        results = {}
        for (source, target), paths in self.paths.items():
            results[(source, target)] = self.probe_path(source, target, num_probes)
        return results