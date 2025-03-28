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
from active_simulator_v2 import ActiveSimulator_v2
from base_prober import BaseProber
from adaptive_prober import AdaptiveProber


# TO-DO COMPARE WIHT KL DIVERGENCE
class ProbingEvaluator:
    def __init__(self, max_departure_time=100):
        self.max_departure_time = max_departure_time
        self.results = {}
        
    def run_single_evaluation(self, probes_per_second, intensity, simulator_version=3):
        # 2 sims so rate limiting doesn't stop us 
        if simulator_version == 3:
            base_simulator = ActiveSimulator_v3(self.max_departure_time, max_intensity=intensity)
        else:
            base_simulator = ActiveSimulator_v2(self.max_departure_time)
            
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
        
        # Get ground truth parameters
        ground_truth_params = base_simulator.network.get_distribution_parameters(1, 2)
        
        return {
            'base_prober': base_prober.get_metrics(),
            'adaptive_prober': adaptive_prober.get_metrics(),
            'congestion_intervals': base_simulator.congestion_intervals,
            'ground_truth_params': ground_truth_params
        }
    
    def calculate_accuracy(self, estimated_params, ground_truth_params):
        gt_values = [ground_truth_params['mean'], ground_truth_params['std']]
        mse = np.mean((np.array(estimated_params) - np.array(gt_values))**2)
        return 1 / (1 + mse)  
    
    def calculate_kl_divergence(self, estimated_mean, estimated_std, true_mean, true_std):
        # Add epsilon to avoid division by zero
        eps = 1e-6
        sigma_p = max(estimated_std, eps)
        sigma_q = max(true_std, eps)
        
        # KL(P||Q) formula for normal distributions
        kl = np.log(sigma_q/sigma_p) + (sigma_p**2 + (estimated_mean - true_mean)**2)/(2*sigma_q**2) - 0.5
        return kl

    def plot_results(self, results, probes_per_second, intensity, simulator_version=3):
        base_metrics = results['base_prober']['metrics_per_timeslot']
        adaptive_metrics = results['adaptive_prober']['metrics_per_timeslot']
        congestion_intervals = results['congestion_intervals']
        ground_truth_params = results['ground_truth_params']
        
        # Get timestamps and ground truth values
        times = sorted(set([m[0] for m in base_metrics]))
        true_mean = ground_truth_params['mean']
        true_std = ground_truth_params['std']
        
        # Create plots with 4 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
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
        
        # Plot 3: KL Divergence over time
        base_kl = [self.calculate_kl_divergence(m[1], m[2], true_mean, true_std) for m in base_metrics]
        adaptive_kl = [self.calculate_kl_divergence(m[1], m[2], true_mean, true_std) for m in adaptive_metrics]
        
        ax3.plot(times, base_kl, label='Base Prober', alpha=0.5)
        ax3.plot(times, adaptive_kl, label='Adaptive Prober', alpha=0.5)
        
        ax3.set_title('KL Divergence over Time')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('KL Divergence')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Accuracy over cumulative probes
        base_accuracy = [self.calculate_accuracy(m[1:3], ground_truth_params) for m in base_metrics]
        adaptive_accuracy = [self.calculate_accuracy(m[1:3], ground_truth_params) for m in adaptive_metrics]
        
        ax4.plot(base_cum_probes, base_accuracy, label='Base Prober', alpha=0.5)
        ax4.plot(adaptive_cum_probes, adaptive_accuracy, label='Adaptive Prober', alpha=0.5)
        
        ax4.set_title('Accuracy over Cumulative Probes')
        ax4.set_xlabel('Cumulative Probes')
        ax4.set_ylabel('Accuracy')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        
        # save plot in appropriate directory
        base_dir = '/Users/lukemciver/Network-Simulation-Group-Research-Project/active_monitoring_evolution/luke_probing/plots'
        if simulator_version == 3:
            plot_dir = os.path.join(base_dir, 'v3')
        else:
            plot_dir = os.path.join(base_dir, 'v2')
            
        os.makedirs(plot_dir, exist_ok=True)
        plot_filename = f'probes_rate_{probes_per_second}_intensity_{intensity}.png'
        plt.savefig(os.path.join(plot_dir, plot_filename))
        plt.close(fig)


    # 2 runs wiht same seed, check for equality over their results
    def test_seeding_consistency(self, simulator_version=2, seed=42, probes_per_second=5):
        print(f"Testing seeding consistency with seed {seed}...")
        
        # Create two identical simulators with the same seed
        if simulator_version == 3:
            simulator1 = ActiveSimulator_v3(self.max_departure_time, max_intensity=1.5, seed=seed)
            simulator2 = ActiveSimulator_v3(self.max_departure_time, max_intensity=1.5, seed=seed)
        else:
            simulator1 = ActiveSimulator_v2(paths="1", seed=seed)
            simulator2 = ActiveSimulator_v2(paths="1", seed=seed)
        
        # Set rate limits
        simulator1.max_probes_per_second = probes_per_second
        simulator2.max_probes_per_second = probes_per_second
        
        # Run the same probing pattern on both simulators
        probe_times = [0.5 + i for i in range(20)]  # 20 probes at regular intervals
        
        results1 = []
        results2 = []
        
        # Run probes and collect results
        for t in probe_times:
            delay1 = simulator1.send_probe_at(t)
            results1.append((t, delay1))
        
        for t in probe_times:
            delay2 = simulator2.send_probe_at(t)
            results2.append((t, delay2))
        
        # Compare congestion intervals
        intervals_match = simulator1.congestion_intervals == simulator2.congestion_intervals
        
        # Compare probe results
        results_match = results1 == results2
        
        # Compare event logs
        logs_match = simulator1.event_log == simulator2.event_log
        
        print(f"Congestion intervals match: {intervals_match}")
        print(f"Probe results match: {results_match}")
        print(f"Event logs match: {logs_match}")
        
        if intervals_match and results_match and logs_match:
            print("Seeding is consistent! Both runs produced identical results.")
            return True
        else:
            print("Seeding inconsistency detected! Results differ between runs.")
            return False

if __name__ == "__main__":
    evaluator = ProbingEvaluator()
    
    # Test seeding consistency
    evaluator.test_seeding_consistency(simulator_version=3, seed=42)
    evaluator.test_seeding_consistency(simulator_version=3, seed=123)
    
    # Then run your regular evaluations...
    #parameter_sets = [
      #  (5, 1.0), 
       # (10, 1.5),
       # (20, 2.0),
       # (50, 3.0)
   # ]
    
    # Test with v3 simulator
   # for probes_per_second, intensity in parameter_sets:
        #results = evaluator.run_single_evaluation(probes_per_second, intensity, simulator_version=3)
        #evaluator.plot_results(results, probes_per_second, intensity, simulator_version=3)
    
    # Test with v2 simulator
    #for probes_per_second, intensity in parameter_sets:
       # results = evaluator.run_single_evaluation(probes_per_second, intensity, simulator_version=2)
        #evaluator.plot_results(results, probes_per_second, intensity, simulator_version=2)