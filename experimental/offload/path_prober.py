import sys
import os
import networkx as nx
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from network.ground_truth import GroundTruthNetwork
from experimental.luke.active_probing.offload.congestion_network import CongestionNetwork

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

class PathProber:
    def __init__(self, use_congestion=False, initial_congestion=0.0):
        self.network = CongestionNetwork() if use_congestion else GroundTruthNetwork()
        if use_congestion:
            self.network.set_congestion(True, level=initial_congestion)
        self.use_congestion = use_congestion
        self.probe_stats = {
            'total': 0, 'dropped': 0, 'successful': 0, 'last_batch_drop_rate': 0.0
        }
        self.path_cache = {}
        self.path_metadata = {}

    def set_congestion_level(self, level):
        if self.use_congestion:
            self.network.set_congestion(True, level)
            
    def get_congestion_level(self):
        return self.network.congestion_level if self.use_congestion else 0.0

    def probe_path(self, source: int, target: int, num_probes: int = 1) -> Tuple[List[float], Dict]:
        delay_samples = []
        stats = {
            'total_probes': 0, 'dropped_probes': 0,
            'total_edges_traversed': 0, 'dropped_edges': 0
        }

        for _ in range(num_probes):
            stats['total_probes'] += 1
            path = self._get_cached_path(source, target)
            total_delay, edge_count, dropped = 0, 0, False

            for u, v in zip(path, path[1:]):
                edge_delay = self.network.sample_edge_delay(u, v)
                stats['total_edges_traversed'] += 1
                
                if edge_delay is None:
                    stats['dropped_edges'] += 1
                    dropped = True
                    break
                
                total_delay += edge_delay
                edge_count += 1

            if not dropped:
                delay_samples.append(total_delay)
                self._store_path_metadata(len(path)-1)
            else:
                stats['dropped_probes'] += 1

        return delay_samples, stats

    def adaptive_probe_path(self, source: int, target: int, 
                          min_probes=20, max_probes=1000, 
                          batch_size=20, confidence_threshold=0.9):
        from experimental.luke.active_probing.inference_engine import InferenceEngine
        
        all_delays = []
        total_sent = 0
        confidence = 0.0
        current_batch = batch_size
        inference_engine = InferenceEngine()
        
        while total_sent < max_probes:
            if self.use_congestion:
                congestion = self.network.congestion_level
                drop_rate = self.probe_stats['last_batch_drop_rate']
                
                if congestion > 0.7 or drop_rate > 0.3:
                    current_batch = max(5, int(batch_size * 0.5))
                elif congestion > 0.4 or drop_rate > 0.1:
                    current_batch = max(10, int(batch_size * 0.75))
                else:
                    current_batch = batch_size
            
            new_delays, new_stats = self.probe_path(source, target, current_batch)
            all_delays.extend(new_delays)
            total_sent += current_batch
            
            if len(all_delays) >= min_probes:
                confidence = inference_engine._calculate_distribution_confidence(all_delays)
                if confidence >= confidence_threshold:
                    break
                    
        return all_delays, confidence, total_sent

    def _get_cached_path(self, source: int, target: int) -> List[int]:
        if (source, target) not in self.path_cache:
            self.path_cache[(source, target)] = self.network.get_path(source, target)
        return self.path_cache[(source, target)]

    def _store_path_metadata(self, path_length: int):
        self.path_metadata['last_path_length'] = path_length

    def get_probe_statistics(self):
        stats = self.probe_stats.copy()
        if stats['total'] > 0:
            stats['overall_drop_rate'] = stats['dropped'] / stats['total']
        else:
            stats['overall_drop_rate'] = 0.0
            
        if self.use_congestion:
            stats['congestion'] = self.network.get_congestion_stats()
            
        return stats

    def cleanup(self):
        if self.use_congestion:
            self.network.set_congestion(False)

    def send_probes(self, source: int, target: int, num_probes: int = 1) -> Tuple[List[float], Dict]:
        return self.probe_path(source, target, num_probes)

    def estimate_parameters(self, samples: List[float]) -> Dict:
        """Estimate distribution parameters from delay samples."""
        # Implementation would interface with DistributionEstimator
        return {}

    def test_congestion_detection_speed(self):
        # Ramp congestion from 0.0 → 0.8 over 5 minutes
        for t in range(300):
            current_congestion = min(0.8, t/300 * 0.8)
            self.network.set_congestion(current_congestion)
            
            # Take measurement every 15 seconds
            if t % 15 == 0:
                estimates = self.estimate_parameters()
                kl = self.calculate_kl(true_params, estimates)
                
                # Track time-to-detection metrics
                if kl < threshold:
                    detection_time = t

    def get_distribution_parameters(self, path):
        # For lognormal paths, use multiplicative combination
        total_mu = 0
        total_sigma_sq = 0
        for u, v in zip(path, path[1:]):
            total_mu += self.network.graph[u][v]['mu']
            total_sigma_sq += self.network.graph[u][v]['sigma']**2
        return {
            'mu': total_mu,
            'sigma': np.sqrt(total_sigma_sq)
        }

    def calculate_kl_divergence(self, true_params, estimated_params):
        if distribution_type == "lognormal":
            if any(p < 0 for p in [true_params.get('mu',0), estimated_params.get('mu',0)]):
                return float('nan')