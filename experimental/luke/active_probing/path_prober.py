import sys
import os

# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from network.ground_truth import GroundTruthNetwork
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple

class PathProber:
    def __init__(self):
        self.network = GroundTruthNetwork()
        # not path dict, since rn we just have straight line topology


    def probe_path(self, source: int, target: int, num_probes: int) -> List[float]:
        #probe each link in the path num_probes times
        delays = []
        for _ in range(num_probes):
            total_delay = 0
            current = source
            while current < target:
                delay = self.network.sample_edge_delay(current, current + 1)
                total_delay += delay
                current += 1
            delays.append(total_delay)
            
        return delays

    # def probe_all_paths(self, num_probes = 100):
    #     results = {}
    #     for (source, target), paths in self.paths.items():
    #         results[(source, target)] = self.probe_path(source, target, num_probes)
    #     return results