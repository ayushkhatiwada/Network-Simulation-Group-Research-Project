import networkx as nx
import numpy as np
from collections import defaultdict
import random
import matplotlib
import matplotlib.pyplot as plt
import time

"""
- Represent network as a graph, where each edge is assigned a baseline delay drawn uniformly.
- Add issue; make weight of an edge much higher than others
- send "probes" through graph, and find delays between src and dst
"""

#The actual network topology that our monitoring system shouldn't be able to see
class GroundTruth:
    def __init__(self, seed=42):
        self.graph = nx.Graph()
        random.seed(seed)
        np.random.seed(seed)

    #create topology, assign delays to edges
    def create_topology(self, num_nodes: int, connectivity: float = 0.3):
        self.graph = nx.erdos_renyi_graph(num_nodes, connectivity)
        
        # Assign true delays to edges; base delay is drawn uniformly
        for (u, v) in self.graph.edges():
            base_delay = random.uniform(1, 10)
            self.graph[u][v]['true_delay'] = base_delay
            self.graph[u][v]['current_delay'] = base_delay
            
    def add_issue(self, node: int, delay_range=(20, 50)):
        print(f"increasing delay/weight at {node}")
        affected_edges = list(self.graph.edges(node))
        for (u, v) in affected_edges:
            additional_delay = random.uniform(delay_range[0], delay_range[1])
            self.graph[u][v]['current_delay'] += additional_delay
            print(f"Edge {u}->{v}: Added {additional_delay:.2f}ms delay")
    
    def visualize_path(self, source: int, target: int):
        path = nx.shortest_path(self.graph, source, target)
        plt.figure(figsize=(12, 6))
        pos = nx.spring_layout(self.graph)
        
        nx.draw_networkx_edges(self.graph, pos, alpha=0.2, edge_color='gray')
        
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color='r')
        
        nx.draw_networkx_nodes(self.graph, pos)
        nx.draw_networkx_labels(self.graph, pos)
        
        edge_labels = {(u, v): f"{self.graph[u][v]['current_delay']:.1f}ms" for (u, v) in path_edges}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels)
        
        plt.title(f"Path from {source} to {target}")
        plt.show()
    
    def get_true_delay(self, path):
        total_delay = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            delay = self.graph[u][v]['current_delay']
            # random jitter/extra delay
            jitter = np.random.normal(0, delay * 0.1)
            total_delay += delay + jitter
        return total_delay

class MonitoringSystem:
    def __init__(self, ground_truth):
        self.ground_truth = ground_truth
        self.observed_delays = defaultdict(list)
        self.baseline_delays = {}
        
    def send_probe(self, source, target):
        
        if not nx.has_path(self.ground_truth.graph, source, target):
            return float('inf')
        
        path = nx.shortest_path(self.ground_truth.graph, source, target)
        delay = self.ground_truth.get_true_delay(path)
        additional_jitter = np.random.normal(0, 1)
        delay += additional_jitter
        return delay
        
    def establish_baseline(self, source, target, num_probes=100):
        delays = []
        for _ in range(num_probes):
            delay = self.send_probe(source, target)
            delays.append(delay)
            
        self.baseline_delays[(source, target)] = {
            'mean': np.mean(delays),
            'std': np.std(delays)
        }
        
    def monitor_path(self, source, target, num_probes=10):
        if (source, target) not in self.baseline_delays:
            self.establish_baseline(source, target)
            
        baseline = self.baseline_delays[(source, target)]        
        delays = []
        anomalies = []
        
        for i in range(num_probes):
            delay = self.send_probe(source, target)
            delays.append(delay)
            
            # arbitray 'anomaly'
            if delay > baseline['mean'] + 2 * baseline['std']:
                anomalies.append(i)
        
        mean_delay = np.mean(delays)
        delay_increase = (mean_delay - baseline['mean']) / baseline['mean'] * 100
            
        return {
            'delays': delays,
            'anomalies': anomalies,
            'mean_delay': mean_delay,
            'delay_increase': delay_increase,
            'min_delay': min(delays),
            'max_delay': max(delays)
        }
    
    # How much time we wait between probes wont actually matter in a graph implementation like we have
    # how can we make this more realistic?
    def adaptive_monitor_path(self, source, target, initial_probe_interval=1.0, max_duration=60.0):
        start_time = time.time()
        probe_interval = initial_probe_interval
        delays = []
        
        while (time.time() - start_time) < max_duration:
            delay = self.send_probe(source, target)
            delays.append(delay)
            print(f"Probe delay: {delay:.2f} ms (current interval: {probe_interval:.2f}s)")
            
            if len(delays) >= 5:
                recent_mean = np.mean(delays[-5:])
                # If curr delay is significantly higher than recent probes, reduce the interval
                if delay > recent_mean * 1.5:
                    probe_interval = max(0.1, probe_interval / 2)
                else:
                    probe_interval = min(5.0, probe_interval * 1.05)
            
            time.sleep(probe_interval)
        
        return delays

if __name__ == "__main__":
    ground_truth = GroundTruth(seed=20)
    ground_truth.create_topology(num_nodes=10)
    monitor = MonitoringSystem(ground_truth)
    
    source, target = 0, 5
    
    # static
    results = monitor.monitor_path(source, target)
    print(f"Initial delays: min={results['min_delay']:.2f}ms, mean={results['mean_delay']:.2f}ms, max={results['max_delay']:.2f}ms")
    
    # determine a node in the middle of the route and add an issue there
    path = nx.shortest_path(ground_truth.graph, source, target)
    issue_node = path[len(path) // 2]
    print(f"\nAdding network issue at node {issue_node}...")
    ground_truth.add_issue(node=issue_node)
    
    # static monitoring after adding the issue
    results = monitor.monitor_path(source, target)
    print(f"\nNew delays (static probing): min={results['min_delay']:.2f}ms, mean={results['mean_delay']:.2f}ms, max={results['max_delay']:.2f}ms")
    print(f"Delay increase: {results['delay_increase']:.1f}%")
    
    # adaptive monitoring ... this is kinda useless
    print("\nAdaptive monitoring (dynamic probe intervals)...")
    adaptive_delays = monitor.adaptive_monitor_path(source, target, initial_probe_interval=1.0, max_duration=30.0)
    adaptive_mean = np.mean(adaptive_delays)
    print(f"Adaptive monitoring mean delay: {adaptive_mean:.2f}ms")
