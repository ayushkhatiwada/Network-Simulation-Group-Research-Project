"""
Standardized testing framework for network probing algorithms
"""
from typing import Dict, List, Callable, Type
import time
import numpy as np
from . import PathProber, InferenceEngine
from .congestion_network import CongestionNetwork
from .evaluation import calculate_kl_divergence

class ProbingExperiment:
    def __init__(self, 
                 network_type: Type = CongestionNetwork,
                 probe_strategies: List[Callable] = None,
                 metrics: List[str] = ['kl_divergence', 'probe_count', 'time']):
        """
        Initialize experiment with:
        - network_type: Network implementation class
        - probe_strategies: List of probing strategy functions
        - metrics: List of metrics to collect
        """
        self.network = network_type()
        self.strategies = probe_strategies or [self.baseline_strategy]
        self.metrics = metrics
        self.results = {}
        
    @staticmethod
    def baseline_strategy(prober: PathProber, target_confidence: float) -> dict:
        """Default uniform probing strategy"""
        return prober.adaptive_probe_path(
            prober.network.SOURCE,
            prober.network.DESTINATION,
            confidence_threshold=target_confidence
        )

    def run(self, 
            congestion_levels: List[float] = [0.0, 0.3, 0.6, 0.9],
            trials: int = 5,
            target_confidence: float = 0.9) -> Dict:
        """
        Run full experiment suite:
        - Vary congestion levels
        - Test all probe strategies
        - Collect specified metrics
        """
        results = {}
        
        for level in congestion_levels:
            self.network.set_congestion_level(level)
            true_params = self.network.get_distribution_parameters()
            
            for strategy in self.strategies:
                key = f"L{level}_S{strategy.__name__}"
                results[key] = {
                    'kl': [],
                    'probes': [],
                    'time': [],
                    'drops': []
                }
                
                for _ in range(trials):
                    prober = PathProber(use_congestion=True)
                    prober.network = self.network
                    
                    start_time = time.time()
                    delays, _ = strategy(prober, target_confidence)
                    elapsed = time.time() - start_time
                    
                    if len(delays) < 10:  # Handle failed runs
                        continue
                        
                    est_params = InferenceEngine().estimate_parameters(delays)
                    kl = calculate_kl_divergence(delays, est_params, true_params)
                    
                    results[key]['kl'].append(kl)
                    results[key]['probes'].append(len(delays))
                    results[key]['time'].append(elapsed)
                    results[key]['drops'].append(prober.get_probe_statistics()['overall_drop_rate'])
        
        self.results = results
        return results

    def generate_report(self, output_dir: str = "results"):
        """Generate comparative analysis plots and metrics"""
        if not self.results:
            raise ValueError("Run experiments first with .run()")
            
        # Implementation would create plots comparing:
        # - Accuracy vs probe count for each strategy
        # - Time to convergence across congestion levels
        # - Resource efficiency metrics
        # - Congestion detection capabilities
        pass