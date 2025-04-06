import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from typing import Dict, List, Tuple, Any
import pandas as pd
import json
import random
from concurrent.futures import ProcessPoolExecutor
import math
from tqdm import tqdm
import copy
from collections import defaultdict
import pickle
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from active_simulator_v2 import ActiveSimulator_v2
from active_simulator_v1 import ActiveSimulator_v1
from active_simulator_v0 import ActiveSimulator_v0

from base_prober import BaseProber
from adaptive_prober import AdaptiveProber as CongestionAwareOutlierFilterProber
from SamirAdaptive import AdaptiveActiveProber as BinaryRateAdaptiveProber

from probing_evaluator import ProbingEvaluator


class ActiveProberComparison:
    def __init__(self, num_seeds=50, simulator_version=0):
        self.num_seeds = num_seeds
        self.simulator_version = simulator_version
        self.evaluator = ProbingEvaluator(max_departure_time=100)
        self.results = {}
        
        # Load optimized parameters if available
        self.tuned_params = self._load_tuned_parameters()
        
    def _load_tuned_parameters(self):
        """Load tuned parameters for different durations."""
        tuned_params = {}
        tuned_params_dir = 'tuned_params'
        
        # First try to load the all_best_params file
        all_params_file = f'{tuned_params_dir}/all_best_params_v{self.simulator_version}.json'
        if os.path.exists(all_params_file):
            try:
                with open(all_params_file, 'r') as f:
                    all_params = json.load(f)
                    # Convert string keys (JSON) to integers
                    for duration_str, params in all_params.items():
                        tuned_params[int(duration_str)] = params
                    print(f"Loaded optimized parameters for {len(tuned_params)} durations from {all_params_file}")
                    return tuned_params
            except Exception as e:
                print(f"Error loading all parameters: {e}")
        
        # If all_best_params.json doesn't exist, try to load individual files
        durations = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for duration in durations:
            param_file = f'{tuned_params_dir}/best_params_v{self.simulator_version}_d{duration}.json'
            if os.path.exists(param_file):
                try:
                    with open(param_file, 'r') as f:
                        params = json.load(f)
                        tuned_params[duration] = params
                except Exception as e:
                    print(f"Error loading parameters for duration {duration}: {e}")
        
        if tuned_params:
            print(f"Loaded optimized parameters for {len(tuned_params)} durations")
        else:
            print("No optimized parameters found, using default values")
            
        return tuned_params
    
    def _get_parameters_for_duration(self, duration):
        """Get the optimized parameters for a specific duration."""
        if duration in self.tuned_params:
            return self.tuned_params[duration]
        
        # If exact duration not found, find the closest available duration
        if self.tuned_params:
            durations = sorted(self.tuned_params.keys())
            closest = min(durations, key=lambda x: abs(x - duration))
            print(f"No parameters for duration={duration}s, using closest duration={closest}s")
            return self.tuned_params[closest]
        
        # Default parameters if nothing found
        return {
            'congestion_z': 1.5, 
            'outlier_z': 2.5,
            'duration': duration,
            'probes_per_second': 100
        }
    
    def calculate_kl_divergence(self, pred_mean, pred_std, true_mean, true_std):
        """Calculate KL divergence between two normal distributions."""
        return self.evaluator.calculate_kl_divergence(pred_mean, pred_std, true_mean, true_std)
    

    def run_single_comparison(self, duration, probes_per_second, seed):
        """
        Run a single comparison of all probers with the given parameters,
        while caching results to avoid re-running expensive simulations.
        """
        # Create a unique filename for caching based on the input parameters
        cache_key = f"v{self.simulator_version}_d{duration}_r{probes_per_second}_seed{seed}.pkl"
        cache_dir = "cache"
        cache_path = os.path.join(cache_dir, cache_key)

        # Check for an existing cache file
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        # Otherwise, run the simulation
        if self.simulator_version == 2:
            simulator = ActiveSimulator_v2(paths="1", simulation_duration=duration, random_seed=seed)
        elif self.simulator_version == 1:
            simulator = ActiveSimulator_v1(paths="1", simulation_duration=duration, random_seed=seed)
        else:
            simulator = ActiveSimulator_v0(paths="1", simulation_duration=duration, random_seed=seed)

        # Create fresh simulator instances for each prober
        if self.simulator_version == 2:
            adaptive_simulator = ActiveSimulator_v2(paths="1", simulation_duration=duration, random_seed=seed)
            samir_simulator = ActiveSimulator_v2(paths="1", simulation_duration=duration, random_seed=seed)
        elif self.simulator_version == 1:
            adaptive_simulator = ActiveSimulator_v1(paths="1", simulation_duration=duration, random_seed=seed)
            samir_simulator = ActiveSimulator_v1(paths="1", simulation_duration=duration, random_seed=seed)
        else:
            adaptive_simulator = ActiveSimulator_v0(paths="1", simulation_duration=duration, random_seed=seed)
            samir_simulator = ActiveSimulator_v0(paths="1", simulation_duration=duration, random_seed=seed)

        # Set probe rate limits on all simulators
        simulator.max_probes_per_second = probes_per_second
        adaptive_simulator.max_probes_per_second = probes_per_second
        samir_simulator.max_probes_per_second = probes_per_second

        # Get optimized parameters for this duration
        params = self._get_parameters_for_duration(duration)
        congestion_z = params.get('congestion_z', 1.5)
        outlier_z = params.get('outlier_z', 2.5)

        # Initialize probers
        base_prober = BaseProber(simulator, probes_per_second)
        adaptive_prober = CongestionAwareOutlierFilterProber(
            adaptive_simulator,
            probes_per_second,
            congestion_z=congestion_z,
            outlier_z=outlier_z
        )
        samir_prober = BinaryRateAdaptiveProber(
            samir_simulator, 
            probes_per_second, 
            time_limit=duration
        )

        # Acquire ground truth parameters from the simulator
        ground_truth_params = simulator.network.get_single_edge_distribution_parameters(
            simulator.network.SOURCE,
            simulator.network.DESTINATION
        )

        # Run each prober and measure CPU time
        start_time = time.process_time()
        base_prober.probe()
        base_cpu_time = time.process_time() - start_time

        start_time = time.process_time()
        adaptive_prober.probe()
        adaptive_cpu_time = time.process_time() - start_time

        start_time = time.process_time()
        samir_prober.probe()
        samir_cpu_time = time.process_time() - start_time

        # Compute final metrics
        base_metrics = base_prober.get_metrics()
        adaptive_metrics = adaptive_prober.get_metrics()
        samir_metrics = samir_prober.get_metrics()

        # Extract final estimates
        final_base_mean = base_metrics['metrics_per_timeslot'][-1][1]
        final_base_std = base_metrics['metrics_per_timeslot'][-1][2]
        final_adaptive_mean = adaptive_metrics['metrics_per_timeslot'][-1][1]
        final_adaptive_std = adaptive_metrics['metrics_per_timeslot'][-1][2]
        final_samir_mean = samir_metrics['metrics_per_timeslot'][-1][1]
        final_samir_std = samir_metrics['metrics_per_timeslot'][-1][2]

        # Calculate KL divergence
        base_kl = self.calculate_kl_divergence(
            final_base_mean, final_base_std,
            ground_truth_params['mean'], ground_truth_params['std']
        )
        adaptive_kl = self.calculate_kl_divergence(
            final_adaptive_mean, final_adaptive_std,
            ground_truth_params['mean'], ground_truth_params['std']
        )
        samir_kl = self.calculate_kl_divergence(
            final_samir_mean, final_samir_std,
            ground_truth_params['mean'], ground_truth_params['std']
        )

        # Gather total probes
        base_total_probes = sum(m[3] for m in base_metrics['metrics_per_timeslot'])
        adaptive_total_probes = sum(m[3] for m in adaptive_metrics['metrics_per_timeslot'])
        samir_total_probes = sum(m[3] for m in samir_metrics['metrics_per_timeslot'])

        # Create a results dictionary
        result = {
            'seed': seed,
            'duration': duration,
            'probes_per_second': probes_per_second,
            'ground_truth': ground_truth_params,
            'base_prober': {
                'kl_divergence': base_kl,
                'cpu_time': base_cpu_time,
                'total_probes': base_total_probes,
                'kl_per_cpu': base_kl / base_cpu_time if base_cpu_time > 0 else float('inf'),
                'kl_per_probe_rate': base_kl / probes_per_second
            },
            'congestion_aware_prober': {
                'kl_divergence': adaptive_kl,
                'cpu_time': adaptive_cpu_time,
                'total_probes': adaptive_total_probes,
                'kl_per_cpu': adaptive_kl / adaptive_cpu_time if adaptive_cpu_time > 0 else float('inf'),
                'kl_per_probe_rate': adaptive_kl / probes_per_second
            },
            'binary_rate_prober': {
                'kl_divergence': samir_kl,
                'cpu_time': samir_cpu_time,
                'total_probes': samir_total_probes,
                'kl_per_cpu': samir_kl / samir_cpu_time if samir_cpu_time > 0 else float('inf'),
                'kl_per_probe_rate': samir_kl / probes_per_second
            }
        }

        # Save results to the cache directory
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)

        return result
    
    def run_comparison_over_durations(self, durations, probes_per_second=100):
        """Run comparison across different simulation durations."""
        results = []
        
        for duration in durations:
            print(f"Running comparison for duration={duration}s, probes_per_second={probes_per_second}")
            
            # Run with multiple seeds in parallel
            seed_results = []
            for seed in tqdm(range(self.num_seeds)):
                seed_result = self.run_single_comparison(duration, probes_per_second, seed)
                seed_results.append(seed_result)
            
            # Aggregate results across seeds
            avg_results = {
                'duration': duration,
                'probes_per_second': probes_per_second,
                'base_prober': {
                    'kl_divergence': np.mean([r['base_prober']['kl_divergence'] for r in seed_results]),
                    'kl_divergence_std': np.std([r['base_prober']['kl_divergence'] for r in seed_results]),
                    'cpu_time': np.mean([r['base_prober']['cpu_time'] for r in seed_results]),
                    'kl_per_cpu': np.mean([r['base_prober']['kl_per_cpu'] for r in seed_results]),
                    'kl_per_probe_rate': np.mean([r['base_prober']['kl_per_probe_rate'] for r in seed_results]),
                    # Add total_probes:
                    'total_probes': np.mean([r['base_prober']['total_probes'] for r in seed_results])
                },
                'congestion_aware_prober': {
                    'kl_divergence': np.mean([r['congestion_aware_prober']['kl_divergence'] for r in seed_results]),
                    'kl_divergence_std': np.std([r['congestion_aware_prober']['kl_divergence'] for r in seed_results]),
                    'cpu_time': np.mean([r['congestion_aware_prober']['cpu_time'] for r in seed_results]),
                    'kl_per_cpu': np.mean([r['congestion_aware_prober']['kl_per_cpu'] for r in seed_results]),
                    'kl_per_probe_rate': np.mean([r['congestion_aware_prober']['kl_per_probe_rate'] for r in seed_results]),
                    'total_probes': np.mean([r['congestion_aware_prober']['total_probes'] for r in seed_results])
                },
                'binary_rate_prober': {
                    'kl_divergence': np.mean([r['binary_rate_prober']['kl_divergence'] for r in seed_results]),
                    'kl_divergence_std': np.std([r['binary_rate_prober']['kl_divergence'] for r in seed_results]),
                    'cpu_time': np.mean([r['binary_rate_prober']['cpu_time'] for r in seed_results]),
                    'kl_per_cpu': np.mean([r['binary_rate_prober']['kl_per_cpu'] for r in seed_results]),
                    'kl_per_probe_rate': np.mean([r['binary_rate_prober']['kl_per_probe_rate'] for r in seed_results]),
                    'total_probes': np.mean([r['binary_rate_prober']['total_probes'] for r in seed_results])
                },
                'raw_results': seed_results
            }
            
            results.append(avg_results)
        
        self.results['durations'] = results
        self._save_results('durations')
        self._plot_kl_vs_duration(results)
        return results
        
    def run_comparison_over_probe_rates(self, probe_rates, duration=100):
        """Run comparison across different probe rates."""
        results = []
        
        for rate in probe_rates:
            print(f"Running comparison for duration={duration}s, probes_per_second={rate}")
            
            # Run with multiple seeds in parallel
            seed_results = []
            for seed in tqdm(range(self.num_seeds)):
                seed_result = self.run_single_comparison(duration, rate, seed)
                seed_results.append(seed_result)
            
            # Aggregate results across seeds
            avg_results = {
                'duration': duration,
                'probes_per_second': rate,
                'base_prober': {
                    'kl_divergence': np.mean([r['base_prober']['kl_divergence'] for r in seed_results]),
                    'kl_divergence_std': np.std([r['base_prober']['kl_divergence'] for r in seed_results]),
                    'cpu_time': np.mean([r['base_prober']['cpu_time'] for r in seed_results]),
                    'kl_per_cpu': np.mean([r['base_prober']['kl_per_cpu'] for r in seed_results]),
                    'kl_per_probe_rate': np.mean([r['base_prober']['kl_per_probe_rate'] for r in seed_results]),
                    'total_probes': np.mean([r['base_prober']['total_probes'] for r in seed_results])
                },
                'congestion_aware_prober': {
                    'kl_divergence': np.mean([r['congestion_aware_prober']['kl_divergence'] for r in seed_results]),
                    'kl_divergence_std': np.std([r['congestion_aware_prober']['kl_divergence'] for r in seed_results]),
                    'cpu_time': np.mean([r['congestion_aware_prober']['cpu_time'] for r in seed_results]),
                    'kl_per_cpu': np.mean([r['congestion_aware_prober']['kl_per_cpu'] for r in seed_results]),
                    'kl_per_probe_rate': np.mean([r['congestion_aware_prober']['kl_per_probe_rate'] for r in seed_results]),
                    'total_probes': np.mean([r['congestion_aware_prober']['total_probes'] for r in seed_results])
                },
                'binary_rate_prober': {
                    'kl_divergence': np.mean([r['binary_rate_prober']['kl_divergence'] for r in seed_results]),
                    'kl_divergence_std': np.std([r['binary_rate_prober']['kl_divergence'] for r in seed_results]),
                    'cpu_time': np.mean([r['binary_rate_prober']['cpu_time'] for r in seed_results]),
                    'kl_per_cpu': np.mean([r['binary_rate_prober']['kl_per_cpu'] for r in seed_results]),
                    'kl_per_probe_rate': np.mean([r['binary_rate_prober']['kl_per_probe_rate'] for r in seed_results]),
                    'total_probes': np.mean([r['binary_rate_prober']['total_probes'] for r in seed_results])
                },
                'raw_results': seed_results
            }

            
            results.append(avg_results)
        
        self.results['probe_rates'] = results
        self._save_results('probe_rates')
        self._plot_kl_vs_probe_rate(results)
        return results
    
    def _save_results(self, result_type):
        """Save results to a text file and JSON."""
        # Create output directory based on simulator version
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(script_dir, f'comparison_results/v{self.simulator_version}')
        os.makedirs(base_dir, exist_ok=True)
        
        # Save detailed JSON results
        with open(f'{base_dir}/{result_type}_results_detailed.json', 'w') as f:
            # Filter out potentially non-serializable objects
            filtered_results = []
            for r in self.results[result_type]:
                r_copy = r.copy()
                if 'raw_results' in r_copy:
                    del r_copy['raw_results']
                filtered_results.append(r_copy)
            
            json.dump(filtered_results, f, indent=4)
        
        # Create and save summary table
        with open(f'{base_dir}/{result_type}_summary.txt', 'w') as f:
            if result_type == 'durations':
                f.write(f"KL Divergence vs. Simulation Duration (averaged over {self.num_seeds} seeds)\n")
                f.write("="*80 + "\n")
                f.write(f"{'Duration (s)':<15}{'Base Prober':<25}{'Adaptive Prober':<25}{'Samir Prober':<25}\n")
                f.write("-"*80 + "\n")
                
                for result in self.results[result_type]:
                    duration = result['duration']
                    base_kl = result['base_prober']['kl_divergence']
                    adaptive_kl = result['congestion_aware_prober']['kl_divergence']
                    samir_kl = result['binary_rate_prober']['kl_divergence']
                    
                    f.write(f"{duration:<15}{base_kl:<25.4f}{adaptive_kl:<25.4f}{samir_kl:<25.4f}\n")
                
                f.write("\n\nKL Divergence / CPU Time\n")
                f.write("="*80 + "\n")
                f.write(f"{'Duration (s)':<15}{'Base Prober':<25}{'Adaptive Prober':<25}{'Samir Prober':<25}\n")
                f.write("-"*80 + "\n")
                
                for result in self.results[result_type]:
                    duration = result['duration']
                    base_kl_cpu = result['base_prober']['kl_per_cpu']
                    adaptive_kl_cpu = result['congestion_aware_prober']['kl_per_cpu']
                    samir_kl_cpu = result['binary_rate_prober']['kl_per_cpu']
                    
                    f.write(f"{duration:<15}{base_kl_cpu:<25.4f}{adaptive_kl_cpu:<25.4f}{samir_kl_cpu:<25.4f}\n")
                
            elif result_type == 'probe_rates':
                f.write(f"KL Divergence vs. Probe Rate (averaged over {self.num_seeds} seeds)\n")
                f.write("="*80 + "\n")
                f.write(f"{'Probe Rate':<15}{'Base Prober':<25}{'Adaptive Prober':<25}{'Samir Prober':<25}\n")
                f.write("-"*80 + "\n")
                
                for result in self.results[result_type]:
                    rate = result['probes_per_second']
                    base_kl = result['base_prober']['kl_divergence']
                    adaptive_kl = result['congestion_aware_prober']['kl_divergence']
                    samir_kl = result['binary_rate_prober']['kl_divergence']
                    
                    f.write(f"{rate:<15}{base_kl:<25.4f}{adaptive_kl:<25.4f}{samir_kl:<25.4f}\n")
                
                f.write("\n\nKL Divergence / Probe Rate\n")
                f.write("="*80 + "\n")
                f.write(f"{'Probe Rate':<15}{'Base Prober':<25}{'Adaptive Prober':<25}{'Samir Prober':<25}\n")
                f.write("-"*80 + "\n")
                
                for result in self.results[result_type]:
                    rate = result['probes_per_second']
                    base_kl_rate = result['base_prober']['kl_per_probe_rate']
                    adaptive_kl_rate = result['congestion_aware_prober']['kl_per_probe_rate']
                    samir_kl_rate = result['binary_rate_prober']['kl_per_probe_rate']
                    
                    f.write(f"{rate:<15}{base_kl_rate:<25.4f}{adaptive_kl_rate:<25.4f}{samir_kl_rate:<25.4f}\n")
    
    def _plot_kl_vs_duration(self, results):
        """Plot KL divergence and CPU time vs. simulation duration."""
        durations = [r['duration'] for r in results]
        probes_per_second = results[0]['probes_per_second']
        
        # Create output directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(script_dir, f'comparison_results/v{self.simulator_version}')
        os.makedirs(base_dir, exist_ok=True)
        
        # 1. Plot KL divergence vs duration
        plt.figure(figsize=(10, 6))
        plt.errorbar(durations, 
                    [r['base_prober']['kl_divergence'] for r in results],
                    yerr=[r['base_prober']['kl_divergence_std'] for r in results], 
                    marker='o', label='Uniform Rate Prober')
        plt.errorbar(durations, 
                    [r['congestion_aware_prober']['kl_divergence'] for r in results],
                    yerr=[r['congestion_aware_prober']['kl_divergence_std'] for r in results], 
                    marker='s', label='Congestion-Aware Outlier Filter Prober')
        plt.errorbar(durations, 
                    [r['binary_rate_prober']['kl_divergence'] for r in results],
                    yerr=[r['binary_rate_prober']['kl_divergence_std'] for r in results], 
                    marker='^', label='Binary Rate Adaptive Prober')
        plt.xlabel('Simulation Duration (seconds)')
        plt.ylabel('KL Divergence')
        plt.title(f'KL Divergence vs. Simulation Duration (v{self.simulator_version}, {probes_per_second} probes/sec)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{base_dir}/kl_vs_duration_{probes_per_second}pps.png')
        
        # 2. Plot CPU time vs duration (direct measurement)
        plt.figure(figsize=(10, 6))
        plt.plot(durations, [r['base_prober']['cpu_time'] for r in results], marker='o', label='Uniform Rate Prober')
        plt.plot(durations, [r['congestion_aware_prober']['cpu_time'] for r in results], marker='s', label='Congestion-Aware Outlier Filter Prober')
        plt.plot(durations, [r['binary_rate_prober']['cpu_time'] for r in results], marker='^', label='Binary Rate Adaptive Prober')
        plt.xlabel('Simulation Duration (seconds)')
        plt.ylabel('CPU Time (seconds)')
        plt.title(f'CPU Time vs. Simulation Duration (v{self.simulator_version}, {probes_per_second} probes/sec)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{base_dir}/cpu_time_vs_duration_{probes_per_second}pps.png')
        
        # 3. Plot total probes vs duration
        plt.figure(figsize=(10, 6))
        plt.plot(durations, [r['base_prober']['total_probes'] for r in results], marker='o', label='Uniform Rate Prober')
        plt.plot(durations, [r['congestion_aware_prober']['total_probes'] for r in results], marker='s', label='Congestion-Aware Outlier Filter Prober')
        plt.plot(durations, [r['binary_rate_prober']['total_probes'] for r in results], marker='^', label='Binary Rate Adaptive Prober')
        plt.xlabel('Simulation Duration (seconds)')
        plt.ylabel('Total Probes Sent')
        plt.title(f'Total Probes vs. Simulation Duration (v{self.simulator_version}, {probes_per_second} probes/sec)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{base_dir}/total_probes_vs_duration_{probes_per_second}pps.png')
    
    def _plot_kl_vs_probe_rate(self, results):
        """Plot KL divergence and CPU time vs. probe rate."""
        rates = [r['probes_per_second'] for r in results]
        duration = results[0]['duration']
        
        # Create output directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(script_dir, f'comparison_results/v{self.simulator_version}')
        os.makedirs(base_dir, exist_ok=True)
        
        # 1. Plot KL divergence vs probe rate
        plt.figure(figsize=(10, 6))
        plt.errorbar(rates, 
                    [r['base_prober']['kl_divergence'] for r in results],
                    yerr=[r['base_prober']['kl_divergence_std'] for r in results], 
                    marker='o', label='Uniform Rate Prober')
        plt.errorbar(rates, 
                    [r['congestion_aware_prober']['kl_divergence'] for r in results],
                    yerr=[r['congestion_aware_prober']['kl_divergence_std'] for r in results], 
                    marker='s', label='Congestion-Aware Outlier Filter Prober')
        plt.errorbar(rates, 
                    [r['binary_rate_prober']['kl_divergence'] for r in results],
                    yerr=[r['binary_rate_prober']['kl_divergence_std'] for r in results], 
                    marker='^', label='Binary Rate Adaptive Prober')
        plt.xlabel('Probes per Second')
        plt.ylabel('KL Divergence')
        plt.title(f'KL Divergence vs. Probe Rate (v{self.simulator_version}, {duration}s duration)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{base_dir}/kl_vs_probe_rate_{duration}s.png')
        
        # 2. Plot CPU time vs probe rate
        plt.figure(figsize=(10, 6))
        plt.plot(rates, [r['base_prober']['cpu_time'] for r in results], marker='o', label='Uniform Rate Prober')
        plt.plot(rates, [r['congestion_aware_prober']['cpu_time'] for r in results], marker='s', label='Congestion-Aware Outlier Filter Prober')
        plt.plot(rates, [r['binary_rate_prober']['cpu_time'] for r in results], marker='^', label='Binary Rate Adaptive Prober')
        plt.xlabel('Probes per Second')
        plt.ylabel('CPU Time (seconds)')
        plt.title(f'CPU Time vs. Probe Rate (v{self.simulator_version}, {duration}s duration)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{base_dir}/cpu_time_vs_probe_rate_{duration}s.png')
        
        # 3. Plot total probes vs probe rate
        plt.figure(figsize=(10, 6))
        plt.plot(rates, [r['base_prober']['total_probes'] for r in results], marker='o', label='Uniform Rate Prober')
        plt.plot(rates, [r['congestion_aware_prober']['total_probes'] for r in results], marker='s', label='Congestion-Aware Outlier Filter Prober')
        plt.plot(rates, [r['binary_rate_prober']['total_probes'] for r in results], marker='^', label='Binary Rate Adaptive Prober')
        plt.xlabel('Probes per Second')
        plt.ylabel('Total Probes Sent')
        plt.title(f'Total Probes vs. Probe Rate (v{self.simulator_version}, {duration}s duration)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{base_dir}/total_probes_vs_probe_rate_{duration}s.png')


if __name__ == "__main__":
    # Define test parameters
    durations = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    probe_rates = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  
    
    # During development, use fewer seeds for testing
    num_seeds = 50  # Set back to 50 for final run
    
    # Create comparisons for each simulator version
    for version in [0, 1, 2]:
        print(f"\n=== Running comparisons for simulator version {version} ===\n")
        comparison = ActiveProberComparison(num_seeds=num_seeds, simulator_version=version)
        
        # Run duration comparison with a fixed probe rate of 100
        comparison.run_comparison_over_durations(durations, probes_per_second=100)
        
        # Run probe rate comparison with a fixed duration of 100s
        comparison.run_comparison_over_probe_rates(probe_rates, duration=100)