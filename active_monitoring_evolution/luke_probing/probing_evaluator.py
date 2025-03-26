import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from typing import Dict, List, Tuple, Any
from scipy import stats
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import copy

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from active_simulator_v3 import ActiveSimulator_v3
from base_prober import BaseProber
from adaptive_prober import AdaptiveProber


# TO-DO COMPARE WIHT KL DIVERGENCE
class ProbingEvaluator:
    def __init__(self, max_departure_time=100):
        self.max_departure_time = max_departure_time
        self.results = {}
        
    def run_single_evaluation(self, probes_per_second, intensity):
        # 2 sims so rate limitng doesnt stop us 
        base_simulator = ActiveSimulator_v3(self.max_departure_time, max_intensity=intensity)
        adaptive_simulator = copy.deepcopy(base_simulator)
        
        seed = 42
        base_simulator.seed = seed
        adaptive_simulator.seed = seed
        
        base_simulator.max_probes_per_second = probes_per_second
        base_simulator.probe_count_per_second = {}
        
        adaptive_simulator.max_probes_per_second = probes_per_second
        adaptive_simulator.probe_count_per_second = {}
        
        base_prober = BaseProber(base_simulator, probes_per_second)
        adaptive_prober = AdaptiveProber(adaptive_simulator, probes_per_second)
        
        base_prober.probe()
        adaptive_prober.probe()
        
        ground_truth_params = base_simulator.network.get_distribution_parameters(1, 2)
        
        return {
            'base_prober': base_prober.get_metrics(),
            'adaptive_prober': adaptive_prober.get_metrics(),
            'congestion_intervals': base_simulator.congestion_intervals,
            'ground_truth_params': ground_truth_params
        }
    
    def calculate_accuracy(self, estimated_params, ground_truth_params):
        mse = np.mean((np.array(estimated_params) - np.array(list(ground_truth_params.values())))**2)
        return 1 / (1 + mse)  
    
    def plot_results(self, results, probes_per_second, intensity):
        base_metrics = results['base_prober']['metrics_per_timeslot']
        adaptive_metrics = results['adaptive_prober']['metrics_per_timeslot']
        congestion_intervals = results['congestion_intervals']
        ground_truth_params = results['ground_truth_params']
        
        # get timestamps
        times = sorted(set([m[0] for m in base_metrics]))
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot 1: Probes per second
        base_probes = [m[3] for m in base_metrics] 
        adaptive_probes = [m[3] for m in adaptive_metrics]
        
        ax1.plot(times, base_probes, label='Base Prober', alpha=0.5)
        ax1.plot(times, adaptive_probes, label='Adaptive Prober', alpha=0.5)
        
        # shade congestion intervals
        for start, end in congestion_intervals:
            ax1.axvspan(start, end, color='red', alpha=0.1)
        
        ax1.set_title('Probes per Second')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Probes')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Cumulative probes
        base_cum_probes = np.cumsum(base_probes)
        adaptive_cum_probes = np.cumsum(adaptive_probes)
        
        ax2.plot(times, base_cum_probes, label='Base Prober', alpha=0.5)
        ax2.plot(times, adaptive_cum_probes, label='Adaptive Prober', alpha=0.5)
        
        ax2.set_title('Cumulative Probes')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Total Probes')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Accuracy over cumulative probes
        # Calculate accuracy for each prober
        base_accuracy = [self.calculate_accuracy(m[1:3], ground_truth_params) for m in base_metrics]
        adaptive_accuracy = [self.calculate_accuracy(m[1:3], ground_truth_params) for m in adaptive_metrics]
        
        ax3.plot(base_cum_probes, base_accuracy, label='Base Prober', alpha=0.5)
        ax3.plot(adaptive_cum_probes, adaptive_accuracy, label='Adaptive Prober', alpha=0.5)
        
        ax3.set_title('Accuracy over Cumulative Probes')
        ax3.set_xlabel('Cumulative Probes')
        ax3.set_ylabel('Accuracy')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        
        # Save plot; hardcoded for now
        plot_dir = '/Users/lukemciver/Network-Simulation-Group-Research-Project/active_monitoring_evolution/luke_probing/plots'
        os.makedirs(plot_dir, exist_ok=True)
        plot_filename = f'probes_rate_{probes_per_second}_intensity_{intensity}.png'
        plt.savefig(os.path.join(plot_dir, plot_filename))
        plt.close(fig)

if __name__ == "__main__":
    evaluator = ProbingEvaluator()
    
    parameter_sets = [
        (5, 1.0), 
        (10, 1.5),
        (20, 2.0),
        (50, 3.0)
    ]
    
    for probes_per_second, intensity in parameter_sets:
        results = evaluator.run_single_evaluation(probes_per_second, intensity)
        evaluator.plot_results(results, probes_per_second, intensity)