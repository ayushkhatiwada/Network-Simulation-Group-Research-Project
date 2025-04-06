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
import math
import json

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from active_simulator_v3 import ActiveSimulator_v3
from active_simulator_v2 import ActiveSimulator_v2
from active_simulator_v1 import ActiveSimulator_v1

from base_prober import BaseProber
from adaptive_prober import AdaptiveProber


# TO-DO COMPARE WIHT KL DIVERGENCE
class ProbingEvaluator:
    def __init__(self, max_departure_time=100):
        self.max_departure_time = max_departure_time
        self.results = {}
        
    def run_single_evaluation(self, probes_per_second, simulator_version=3, random_seed=42):
        # Check if tuned parameters exist
        tuned_params_dir = 'tuned_params'
        tuned_params_file = f'{tuned_params_dir}/best_params_v{simulator_version}.json'
        adaptive_params = None
        
        if os.path.exists(tuned_params_file):
            try:
                with open(tuned_params_file, 'r') as f:
                    best_params = json.load(f)
                    adaptive_params = {
                        'congestion_z': best_params['congestion_z'],
                        'outlier_z': best_params['outlier_z']
                    }
                    print(f"Using tuned parameters: congestion_z={adaptive_params['congestion_z']}, "
                          f"outlier_z={adaptive_params['outlier_z']}")
            except:
                print("Error loading tuned parameters, using defaults")
        
        # 2 sims so rate limiting doesn't stop us 
        if simulator_version == 2:
            base_simulator = ActiveSimulator_v2(
                paths="1",
                random_seed=42,
                simulation_duration=self.max_departure_time
            )
            
        adaptive_simulator = copy.deepcopy(base_simulator)
        
        random_seed = 42
        base_simulator.random_seed = random_seed
        adaptive_simulator.random_seed = random_seed
        
        base_simulator.max_probes_per_second = probes_per_second
        base_simulator.probe_count_per_second = {}
        
        adaptive_simulator.max_probes_per_second = probes_per_second
        adaptive_simulator.probe_count_per_second = {}
        
        base_prober = BaseProber(base_simulator, probes_per_second)
        
        # Use tuned parameters if available
        if adaptive_params:
            adaptive_prober = AdaptiveProber(
                adaptive_simulator, 
                max_probes_per_second=probes_per_second,
                **adaptive_params
            )
        else:
            adaptive_prober = AdaptiveProber(adaptive_simulator, probes_per_second)
        
        base_prober.probe()
        adaptive_prober.probe()
        
        # Get ground truth parameters using the correct method
        ground_truth_params = base_simulator.network.get_single_edge_distribution_parameters(
            base_simulator.network.SOURCE, 
            base_simulator.network.DESTINATION
        )
        base_metrics = base_prober.get_metrics()
        adaptive_metrics = adaptive_prober.get_metrics()
        
        # Get metrics
        base_metrics = base_prober.get_metrics()
        adaptive_metrics = adaptive_prober.get_metrics()
        
        # Instead of using all samples, use the final timestep's estimates
        # This will make it consistent with what's shown in the graph
        final_base_mean = base_metrics['metrics_per_timeslot'][-1][1]  # final estimated mean
        final_base_std = base_metrics['metrics_per_timeslot'][-1][2]   # final estimated std
        final_adaptive_mean = adaptive_metrics['metrics_per_timeslot'][-1][1]
        final_adaptive_std = adaptive_metrics['metrics_per_timeslot'][-1][2]
        
        # Calculate KL divergence using final estimates
        base_kl = self.calculate_kl_divergence(
            final_base_mean, 
            final_base_std, 
            ground_truth_params['mean'], 
            ground_truth_params['std']
        )
        
        adaptive_kl = self.calculate_kl_divergence(
            final_adaptive_mean, 
            final_adaptive_std, 
            ground_truth_params['mean'], 
            ground_truth_params['std']
        )
        
        # Count dropped probes
        base_dropped = sum(1 for _, d in base_prober.probe_history if d is None)
        adaptive_dropped = sum(1 for _, d in adaptive_prober.probe_history if d is None)
        base_total = len(base_prober.probe_history)
        adaptive_total = len(adaptive_prober.probe_history)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Metric': [
                'Mean Delay (ms)', 
                'Std Dev (ms)',
                'KL Divergence',
                'Total Probes',
                'Dropped Probes',
                'Drop Rate (%)',
                'Ground Truth Mean',
                'Ground Truth Std'
            ],
            'Base Prober': [
                f"{final_base_mean:.3f}",
                f"{final_base_std:.3f}",
                f"{base_kl:.4f} {'✅' if base_kl <= 0.05 else '❌'}",
                base_total,
                base_dropped,
                f"{(base_dropped/base_total)*100:.1f}%",
                f"{ground_truth_params['mean']:.3f}",
                f"{ground_truth_params['std']:.3f}"
            ],
            'Adaptive Prober': [
                f"{final_adaptive_mean:.3f}",
                f"{final_adaptive_std:.3f}",
                f"{adaptive_kl:.4f} {'✅' if adaptive_kl <= 0.05 else '❌'}",
                adaptive_total,
                adaptive_dropped,
                f"{(adaptive_dropped/adaptive_total)*100:.1f}%",
                f"{ground_truth_params['mean']:.3f}",
                f"{ground_truth_params['std']:.3f}"
            ]
        })
        
        # Print congestion intervals
        print("\nCongestion Intervals:")
        for start, end in base_simulator.congestion_intervals:
            print(f"  {start:.2f}s to {end:.2f}s")
            
        # Print results table
        print("\nProbing Results:")
        print(results_df.to_string(index=False))
        
        # If using tuned parameters, print them
        if adaptive_params:
            print(f"\nUsing tuned parameters: congestion_z={adaptive_params['congestion_z']}, "
                  f"outlier_z={adaptive_params['outlier_z']}")
        
        return {
            'base_prober': base_metrics,
            'adaptive_prober': adaptive_metrics,
            'congestion_intervals': base_simulator.congestion_intervals,
            'ground_truth_params': ground_truth_params,
            'results_df': results_df,
            'base_kl': base_kl,
            'adaptive_kl': adaptive_kl
        }
    
    def calculate_accuracy(self, estimated_params, ground_truth_params):
        gt_values = [ground_truth_params['mean'], ground_truth_params['std']]
        mse = np.mean((np.array(estimated_params) - np.array(gt_values))**2)
        return 1 / (1 + mse)  

    def calculate_kl_divergence(self, pred_mean, pred_std, true_mean, true_std):

        return math.log(pred_std / true_std) + \
               ((true_std**2 + (true_mean - pred_mean)**2) / \
               (2 * pred_std**2)) - 0.5

    def plot_results(self, results, probes_per_second, simulator_version=2, simulation_duration=100):
        """
        Plot the metrics for BaseProber and AdaptiveProber over 'time frames'.
        Also export CPU time, total probes, etc., to a CSV file.

        A 'time frame' is one discrete iteration in the simulator, 
        encompassing any packet departures/arrivals plus an 
        opportunity to send/receive probes.
        """
        print(f"Plotting results for probes_per_second={probes_per_second}, "
              f"simulator_version={simulator_version}, simulation_duration={simulation_duration}")
        
        base_metrics = results['base_prober']['metrics_per_timeslot']
        adaptive_metrics = results['adaptive_prober']['metrics_per_timeslot']
        congestion_intervals = results['congestion_intervals']
        ground_truth_params = results['ground_truth_params']

        # Extract the simulation frames from the metrics:
        # Each entry in metrics_per_timeslot is typically:
        # (time_slot, estimated_mean, estimated_std, probes_sent, cpu_time)
        times = sorted({m[0] for m in base_metrics})

        # For analyzing CPU usage and probes:
        # index 3 => probes_sent, index 4 => cpu_time
        base_probes = [m[3] for m in base_metrics]
        adaptive_probes = [m[3] for m in adaptive_metrics]
        base_cpu_times = [m[4] for m in base_metrics]
        adaptive_cpu_times = [m[4] for m in adaptive_metrics]

        # Cumulative sums
        base_cum_probes = np.cumsum(base_probes)
        adaptive_cum_probes = np.cumsum(adaptive_probes)
        base_cum_cpu = np.cumsum(base_cpu_times)
        adaptive_cum_cpu = np.cumsum(adaptive_cpu_times)

        # Extract ground-truth distribution parameters
        true_mean = ground_truth_params['mean']
        true_std = ground_truth_params['std']

        # Prepare subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # (1) Plot: Probes per time frame
        ax1.plot(times, base_probes, label='Base Prober', alpha=0.5)
        ax1.plot(times, adaptive_probes, label='Adaptive Prober', alpha=0.5)
        # Highlight congestion intervals with shading
        for start, end in congestion_intervals:
            ax1.axvspan(start, end, color='red', alpha=0.1)

        ax1.set_title('Probes per Time Frame')
        ax1.set_xlabel('Time (frames)')  # renamed axis
        ax1.set_ylabel('Probes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # (2) Plot: Cumulative probes
        ax2.plot(times, base_cum_probes, label='Base Prober', alpha=0.5)
        ax2.plot(times, adaptive_cum_probes, label='Adaptive Prober', alpha=0.5)

        ax2.set_title('Cumulative Probes')
        ax2.set_xlabel('Time (frames)')
        ax2.set_ylabel('Total Probes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # (3) Plot: KL Divergence over time
        base_kl = [
            self.calculate_kl_divergence(m[1], m[2], true_mean, true_std)
            for m in base_metrics
        ]
        adaptive_kl = [
            self.calculate_kl_divergence(m[1], m[2], true_mean, true_std)
            for m in adaptive_metrics
        ]

        ax3.plot(times, base_kl, label='Base Prober', alpha=0.5)
        ax3.plot(times, adaptive_kl, label='Adaptive Prober', alpha=0.5)

        ax3.set_title('KL Divergence over Time')
        ax3.set_xlabel('Time (frames)')
        ax3.set_ylabel('KL Divergence')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # (4) Plot: CPU Time per frame (optional example)
        ax4.plot(times, base_cpu_times, label='Base CPU/frame', alpha=0.5)
        ax4.plot(times, adaptive_cpu_times, label='Adaptive CPU/frame', alpha=0.5)

        ax4.set_title('CPU Time per Frame')
        ax4.set_xlabel('Time (frames)')
        ax4.set_ylabel('CPU Time (arbitrary units)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Adjust x limits if desired
        ax1.set_xlim(0, simulation_duration)
        ax2.set_xlim(0, simulation_duration)
        ax3.set_xlim(0, simulation_duration)
        ax4.set_xlim(0, simulation_duration)

        plt.tight_layout()

        # Base directory for saving plots and CSV
        base_dir = '/Users/lukemciver/Network-Simulation-Group-Research-Project/active_monitoring_evolution/luke_probing/plots'
        if simulator_version == 3:
            plot_dir = os.path.join(base_dir, 'v3')
        else:
            plot_dir = os.path.join(base_dir, 'v2')

        os.makedirs(plot_dir, exist_ok=True)
        plot_filename = f'duration_{simulation_duration}_probes_rate_{probes_per_second}.png'
        plt.savefig(os.path.join(plot_dir, plot_filename))
        plt.close(fig)

        # ---------------------------------------------------------------
        # Export the data series used for these plots to a CSV file
        # ---------------------------------------------------------------
        export_data = {
            'time_frames': times,
            'base_probes': base_probes,
            'adaptive_probes': adaptive_probes,
            'base_cpu_times': base_cpu_times,
            'adaptive_cpu_times': adaptive_cpu_times,
            'base_cum_probes': base_cum_probes,
            'adaptive_cum_probes': adaptive_cum_probes,
            'base_cum_cpu': base_cum_cpu,
            'adaptive_cum_cpu': adaptive_cum_cpu,
            'base_kl': base_kl,
            'adaptive_kl': adaptive_kl,
        }

        # Optionally, if you have an accuracy measure:
        # base_accuracy = [self.calculate_accuracy((m[1], m[2]), ground_truth_params) for m in base_metrics]
        # adaptive_accuracy = [self.calculate_accuracy((m[1], m[2]), ground_truth_params) for m in adaptive_metrics]
        #
        # export_data['base_accuracy'] = base_accuracy
        # export_data['adaptive_accuracy'] = adaptive_accuracy

        df_export = pd.DataFrame(export_data)
        csv_filename = f'duration_{simulation_duration}_probes_rate_{probes_per_second}.csv'
        df_export.to_csv(os.path.join(plot_dir, csv_filename), index=False)

        # (Optional) Print final sums
        final_base_cpu = float(base_cum_cpu[-1]) if len(base_cum_cpu) else 0
        final_adaptive_cpu = float(adaptive_cum_cpu[-1]) if len(adaptive_cum_cpu) else 0
        final_base_probes = int(base_cum_probes[-1]) if len(base_cum_probes) else 0
        final_adaptive_probes = int(adaptive_cum_probes[-1]) if len(adaptive_cum_probes) else 0
        
        print(f"---> Exported CSV to: {os.path.join(plot_dir, csv_filename)}")
        print(f"Final BaseProber CPU: {final_base_cpu:.2f}, Probes: {final_base_probes}")
        print(f"Final AdaptiveProber CPU: {final_adaptive_cpu:.2f}, Probes: {final_adaptive_probes}")

    # 2 runs wiht same seed, check for equality over their results
    def test_seeding_consistency(self, simulator_version=2, random_seed=42, probes_per_second=5):
        print(f"Testing seeding consistency with random seed {random_seed}...")
        
        # Create two identical simulators with the same seed
        if simulator_version == 3:
            simulator1 = ActiveSimulator_v3(self.max_departure_time, max_intensity=1.5, random_seed=random_seed)
            simulator2 = ActiveSimulator_v3(self.max_departure_time, max_intensity=1.5, random_seed=random_seed)
        elif simulator_version == 2:
            simulator1 = ActiveSimulator_v2(paths="1", random_seed=random_seed)
            simulator2 = ActiveSimulator_v2(paths="1", random_seed=random_seed)
        elif simulator_version == 1:
            simulator1 = ActiveSimulator_v1(paths="1", random_seed=random_seed)
            simulator2 = ActiveSimulator_v1(paths="1", random_seed=random_seed)
        
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
    # We'll run 3 durations and 3 probe rates
    durations = [10, 50,100]
    probes_per_second_list = [10, 20, 50, 100]
    
    # Define fixed seeds for each duration
    # This ensures reproducibility while still having different patterns for each duration
    duration_seeds = {
        10: 42,   
        50: 123,    
        100: 456   
    }

    for duration in durations:
        # Create a ProbingEvaluator with max_departure_time = duration
        evaluator = ProbingEvaluator(max_departure_time=duration)
        
        # Get the fixed seed for this duration
        fixed_seed = duration_seeds[duration]
        
        for pps in probes_per_second_list:
            print(f"\n=== Testing duration={duration}s, probes_per_second={pps}, seed={fixed_seed} ===")
            
            results = evaluator.run_single_evaluation(
                probes_per_second=pps,
                simulator_version=2,
                random_seed=fixed_seed  
            )
            
            evaluator.plot_results(
                results, 
                pps, 
                simulator_version=2, 
                simulation_duration=duration
            )
