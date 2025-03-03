import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# forcing paths because I want to put all in my folder but keep Ayush's interface where it is
from network_delay_model.true_delay_network import TrueDelayNetwork
from luke.network_inference.path_prober import PathProber
from luke.network_inference.distribution_estimator import DistributionEstimator
import networkx as nx
import numpy as np
from typing import Dict, Tuple, List, Optional


class InferenceEngine:
    def __init__(self):
        # graph interface
        self.prober = PathProber()
        # methods to estimate distribution params from delay data
        self.estimator = DistributionEstimator()
        self.edge_delay_samples = {}  # store delay samples for each edge
        self.edge_distributions = {}  # store inferred distributions
        
    # probe paths, get delays
    def collect_path_data(self, num_probes = 1000):
        # Just collect data for the single path from source to destination
        source = self.prober.network.SOURCE
        destination = self.prober.network.DESTINATION
        
        print(f"Probing path from {source} to {destination} with {num_probes} probes...")
        
        path_delays = self.prober.probe_path(source, destination, num_probes)
        
        print(f"Collected {len(path_delays)} delay samples for path ({source}, {destination})")
        
        self.path_delays = {(source, destination): path_delays}
    
    # just take direct path delays for now (since its single path)
    # in our case, this is not yet needed since we only have one path
    # (this just rebuild the exact dict we get from the prober)
    def decompose_path_delays(self):
        G = self.prober.network.graph
        # how many samples we have for each edge
        edge_counts = {}  
        
        # Initialize edge delay samples for edges in both directions
        for u, v in G.edges():
            self.edge_delay_samples[(u, v)] = []
            self.edge_delay_samples[(v, u)] = []
            edge_counts[(u, v)] = 0
            edge_counts[(v, u)] = 0
        
        for (source, target), delays in self.path_delays.items():
            if G.has_edge(source, target): 
                self.edge_delay_samples[(source, target)].extend(delays)
                edge_counts[(source, target)] += len(delays)
            else:
                # For multi-hop paths, get the actual path
                try:
                    path = nx.shortest_path(G, source, target)
                    # Divide delay equally among edges in the path
                    # This is a simplified approach - in reality, delays won't be equal
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        # Distribute each delay to the edges
                        for delay in delays:
                            # Simple approximation: divide delay equally among edges
                            edge_delay = delay / (len(path) - 1)
                            self.edge_delay_samples[(u, v)].append(edge_delay)
                            edge_counts[(u, v)] += 1
                except nx.NetworkXNoPath:
                    print(f"No path found between {source} and {target}")

        print("Edge delay samples collected:")
        for edge, count in edge_counts.items():
            print(f"Edge {edge}: {count} samples")

        #------------------------------------------------------------------
        # if we extend multi-hop paths, we need more sophisticated methods
        # like in reality, we would need to solve a system of equations
        # would put this below 
        #------------------------------------------------------------------
    
    def infer_edge_distributions(self):
        results = {}
        
        for (u, v), samples in self.edge_delay_samples.items():
            if not samples:
                continue

            # two possible distribution types
            data = np.array(samples)
            distribution_type = self.estimator.test_distribution_fit(data)
            
            if distribution_type == 'gamma':
                shape, scale = self.estimator.fit_gamma(data)
                results[(u, v)] = {
                    'distribution': 'gamma',
                    'shape': shape,
                    'scale': scale
                }
            else:
                mean, std = self.estimator.fit_normal(data)
                results[(u, v)] = {
                    'distribution': 'normal',
                    'mean': mean,
                    'std': std
                }
        
        self.edge_distributions = results
        return results
    
    def run_inference(self, num_probes=100, calculate_confidence=False):
        
        if calculate_confidence:
            # define probe batch sizes to evaluate, get confidence for each
            probe_batches = [10, 20, 50, 100, 200, 500]
            # capped at num_probes
            probe_batches = [b for b in probe_batches if b <= num_probes]
            if num_probes not in probe_batches:
                probe_batches.append(num_probes)
            
            source = self.prober.network.SOURCE
            destination = self.prober.network.DESTINATION
            confidences = {}
            
            print("\n===== CONFIDENCE EVALUATION =====")
            print("Probes | Confidence")
            print("-" * 25)
            
            for batch_size in probe_batches:
                # Collect delay samples
                delays = self.prober.probe_path(source, destination, batch_size)
                
                # Prepare data for decomposition
                self.path_delays = {(source, destination): delays}
                
                # Process samples
                self.decompose_path_delays()
                self.infer_edge_distributions()
                
                # Calculate confidence for each edge
                edge_confidence = {}
                for edge, samples in self.edge_delay_samples.items():
                    if len(samples) >= 5:
                        confidence = self._calculate_confidence(samples)
                        edge_confidence[edge] = confidence
                
                confidences[batch_size] = edge_confidence
                
                # Calculate average confidence across all edges
                if edge_confidence:
                    avg_confidence = sum(edge_confidence.values()) / len(edge_confidence)
                    print(f"{batch_size} | {avg_confidence:.4f}")
            
            # Use the final batch size for the actual inference
            self.path_delays = {(source, destination): self.prober.probe_path(source, destination, num_probes)}
            
        else:
            self.collect_path_data(num_probes)
        
        print("Getting edge delays...")
        self.decompose_path_delays()
        
        print("Inferring edge distributions...")
        return self.infer_edge_distributions()
    
    def _calculate_confidence(self, samples, bootstrap_iterations=1000):
        if len(samples) < 5:  
            return 0.0
        
        samples = np.array(samples)
        distribution_type = self.estimator.test_distribution_fit(samples)
        
        param_variations = []
        
        for _ in range(bootstrap_iterations):
            # Sample with replacement
            bootstrap_sample = np.random.choice(samples, size=len(samples), replace=True)
            
            if distribution_type == 'gamma':
                shape, scale = self.estimator.fit_gamma(bootstrap_sample)
                param_variations.append((shape, scale))
            else:
                mean, std = self.estimator.fit_normal(bootstrap_sample)
                param_variations.append((mean, std))
        
        # Calculate coefficient of variation for each parameter
        params_array = np.array(param_variations)
        # coefficient of variation is std / mean
        cv = np.std(params_array, axis=0) / np.mean(params_array, axis=0)
        
        # Average CV across distribution parameters
        avg_cv = np.mean(cv)
        
        # Convert to confidence score (1 = perfect confidence)
        confidence = max(0, min(1, 1 - avg_cv))
        
        return confidence
    
    def report_results(self):
        print("\n===== NETWORK EDGE DISTRIBUTION INFERENCE REPORT =====")
        
        for (u, v), params in self.edge_distributions.items():
            print(f"\nEdge ({u}, {v}):")
            print(f"  Distribution: {params['distribution']}")
            
            if params['distribution'] == 'gamma':
                print(f"  Shape: {params['shape']:.4f}")
                print(f"  Scale: {params['scale']:.4f}")
            else:
                print(f"  Mean: {params['mean']:.4f}")
                print(f"  Std: {params['std']:.4f}")
        
        print("\n========================================================")

    # compare results with actual values
    def validate_inference(self) -> None:
        from network_delay_model.edges_with_gamma_params import edges_with_gamma_params
        
        actual_params = {}
        for u, v, params in edges_with_gamma_params:
            actual_params[(u, v)] = params
        
        print("\n===== INFERENCE ACCURACY REPORT =====")
        print("Edge | Actual | Inferred | % Error (shape, scale)")
        print("-" * 60)
        
        for (u, v), inferred in self.edge_distributions.items():
            if (u, v) not in actual_params:
                if (u, v) not in [(v, u) for u, v, _ in edges_with_gamma_params]:
                    continue
                actual = actual_params.get((v, u), None)
            else:
                actual = actual_params[(u, v)]
            
            if not actual or inferred['distribution'] != 'gamma':
                continue
            
            shape_error = ((inferred['shape'] - actual['shape']) / actual['shape']) * 100
            scale_error = ((inferred['scale'] - actual['scale']) / actual['scale']) * 100
            
            print(f"({u},{v}) | shape={actual['shape']}, scale={actual['scale']} | " +
                  f"shape={inferred['shape']:.4f}, scale={inferred['scale']:.4f} | " +
                  f"{shape_error:.2f}%, {scale_error:.2f}%")
        
        print("=" * 60)

if __name__ == "__main__":
    engine = InferenceEngine()
    engine.run_inference(num_probes=500, calculate_confidence=True)
    engine.report_results()
    engine.validate_inference()

    prober = PathProber()
    source = prober.network.SOURCE
    destination = prober.network.DESTINATION
    print(f"Testing path from {source} to {destination}")
    delays = prober.probe_path(source, destination, 10)
    print(f"Sample delays: {delays}")