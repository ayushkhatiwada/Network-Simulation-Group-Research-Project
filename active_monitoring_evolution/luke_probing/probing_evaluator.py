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
from kl_prober import KLDivergenceProber
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
        
        # Create simulators - need three now
        if simulator_version == 3:
             # Assuming ActiveSimulator_v3 exists and works similarly
             base_simulator = ActiveSimulator_v3(
                 simulation_duration=self.max_departure_time,
                 random_seed=random_seed # Pass seed here
             )
        elif simulator_version == 2:
            base_simulator = ActiveSimulator_v2(
                paths="1",
                random_seed=random_seed, # Pass seed here
                simulation_duration=self.max_departure_time
            )
        elif simulator_version == 1:
             # Assuming ActiveSimulator_v1 exists and works similarly
             base_simulator = ActiveSimulator_v1(
                 paths="1",
                 random_seed=random_seed, # Pass seed here
                 simulation_duration=self.max_departure_time
             )
        else:
            raise ValueError(f"Unsupported simulator version: {simulator_version}")

        # Deep copy simulators to ensure independent state
        adaptive_simulator = copy.deepcopy(base_simulator)
        kl_simulator = copy.deepcopy(base_simulator) # Create simulator for KL Prober

        # Set max probes per second for each simulator instance
        base_simulator.max_probes_per_second = probes_per_second
        adaptive_simulator.max_probes_per_second = probes_per_second
        kl_simulator.max_probes_per_second = probes_per_second

        # Reset probe counts (important if simulator tracks this internally)
        base_simulator.probe_count_per_second = {}
        adaptive_simulator.probe_count_per_second = {}
        kl_simulator.probe_count_per_second = {}

        # Instantiate Probers
        base_prober = BaseProber(base_simulator, probes_per_second)

        if adaptive_params:
            adaptive_prober = AdaptiveProber(
                adaptive_simulator,
                max_probes_per_second=probes_per_second,
                debug=False, # Disable debug logging during evaluation runs
                **adaptive_params
            )
        else:
            adaptive_prober = AdaptiveProber(
                adaptive_simulator,
                max_probes_per_second=probes_per_second,
                debug=False # Disable debug logging
            )

        # Instantiate KL Divergence Prober (using defaults for now)
        kl_prober = KLDivergenceProber(
            kl_simulator,
            max_probes_per_second=probes_per_second,
            debug=False # Disable debug logging
            # Add specific KL params here if needed, e.g., kl_threshold=0.6
        )

        # Run probing
        print("Running Base Prober...")
        base_prober.probe()
        print("Running Adaptive Prober...")
        adaptive_prober.probe()
        print("Running KL Divergence Prober...")
        kl_prober.probe() # Run the KL prober

        # Get ground truth parameters (use any simulator instance, they share the network initially)
        ground_truth_params_list = base_simulator.network.get_distribution_parameters()
        # Assuming ground_truth_params_list is like [(src, dest, {'mean': m, 'std': s})]
        # Extract the dictionary from the first tuple in the list
        ground_truth_dict = ground_truth_params_list[0][2]
        gt_mean = ground_truth_dict['mean']
        gt_std = ground_truth_dict['std']
        print(f"Ground truth parameters extracted: mean={gt_mean}, std={gt_std}") # Added print for verification

        # Get metrics
        base_metrics = base_prober.get_metrics()
        adaptive_metrics = adaptive_prober.get_metrics()
        kl_metrics = kl_prober.get_metrics() # Get metrics for KL prober

        # Use final timestep's estimates for comparison
        final_base_mean = base_metrics['metrics_per_timeslot'][-1][1] if base_metrics['metrics_per_timeslot'] else 0.0
        final_base_std = base_metrics['metrics_per_timeslot'][-1][2] if base_metrics['metrics_per_timeslot'] else kl_prober.min_std # Use min_std if no data
        final_adaptive_mean = adaptive_metrics['metrics_per_timeslot'][-1][1] if adaptive_metrics['metrics_per_timeslot'] else 0.0
        final_adaptive_std = adaptive_metrics['metrics_per_timeslot'][-1][2] if adaptive_metrics['metrics_per_timeslot'] else kl_prober.min_std
        final_kl_mean = kl_metrics['metrics_per_timeslot'][-1][1] if kl_metrics['metrics_per_timeslot'] else 0.0 # Get KL final mean
        final_kl_std = kl_metrics['metrics_per_timeslot'][-1][2] if kl_metrics['metrics_per_timeslot'] else kl_prober.min_std # Get KL final std

        # Calculate KL divergence against ground truth
        base_kl_vs_gt = self.calculate_kl_divergence(
            final_base_mean, final_base_std,
            gt_mean, gt_std # Use extracted values
        )
        adaptive_kl_vs_gt = self.calculate_kl_divergence(
            final_adaptive_mean, final_adaptive_std,
            gt_mean, gt_std # Use extracted values
        )
        kl_kl_vs_gt = self.calculate_kl_divergence( # Calculate for KL prober
            final_kl_mean, final_kl_std,
            gt_mean, gt_std # Use extracted values
        )

        # Count dropped probes
        base_dropped = sum(1 for _, d in base_prober.probe_history if d is None)
        adaptive_dropped = sum(1 for _, d in adaptive_prober.probe_history if d is None)
        kl_dropped = sum(1 for _, d in kl_prober.probe_history if d is None) # Count for KL prober

        base_total = len(base_prober.probe_history)
        adaptive_total = len(adaptive_prober.probe_history)
        kl_total = len(kl_prober.probe_history) # Total for KL prober

        # Calculate drop rates safely
        base_drop_rate = (base_dropped / base_total * 100) if base_total > 0 else 0
        adaptive_drop_rate = (adaptive_dropped / adaptive_total * 100) if adaptive_total > 0 else 0
        kl_drop_rate = (kl_dropped / kl_total * 100) if kl_total > 0 else 0 # Rate for KL prober

        # Create results dataframe including KL Prober
        results_df = pd.DataFrame({
            'Metric': [
                'Mean Delay (ms)',
                'Std Dev (ms)',
                'KL Divergence (vs GT)', # Clarified metric name
                'Total Probes',
                'Dropped Probes',
                'Drop Rate (%)',
                'Ground Truth Mean',
                'Ground Truth Std'
            ],
            'Base Prober': [
                f"{final_base_mean:.3f}",
                f"{final_base_std:.3f}",
                f"{base_kl_vs_gt:.4f}",
                base_total,
                base_dropped,
                f"{base_drop_rate:.1f}%",
                f"{gt_mean:.3f}", # Use extracted value
                f"{gt_std:.3f}"  # Use extracted value
            ],
            'Adaptive Prober': [
                f"{final_adaptive_mean:.3f}",
                f"{final_adaptive_std:.3f}",
                f"{adaptive_kl_vs_gt:.4f}",
                adaptive_total,
                adaptive_dropped,
                f"{adaptive_drop_rate:.1f}%",
                f"{gt_mean:.3f}", # Use extracted value
                f"{gt_std:.3f}"  # Use extracted value
            ],
            'KL Prober': [ # Added column for KL Prober
                f"{final_kl_mean:.3f}",
                f"{final_kl_std:.3f}",
                f"{kl_kl_vs_gt:.4f}",
                kl_total,
                kl_dropped,
                f"{kl_drop_rate:.1f}%",
                f"{gt_mean:.3f}", # Use extracted value
                f"{gt_std:.3f}"  # Use extracted value
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
            'kl_prober': kl_metrics,
            'congestion_intervals': base_simulator.congestion_intervals,
            'ground_truth_params': {'mean': gt_mean, 'std': gt_std},
            'results_df': results_df,
            'base_kl_vs_gt': base_kl_vs_gt,
            'adaptive_kl_vs_gt': adaptive_kl_vs_gt,
            'kl_kl_vs_gt': kl_kl_vs_gt
        }
    
    def calculate_accuracy(self, estimated_params, ground_truth_params):
        gt_values = [ground_truth_params['mean'], ground_truth_params['std']]
        mse = np.mean((np.array(estimated_params) - np.array(gt_values))**2)
        return 1 / (1 + mse)  

    def calculate_kl_divergence(self, pred_mean, pred_std, true_mean, true_std):
        """Calculates KL(N(true||pred)) - how well pred approximates true."""
        # Ensure std deviations are positive and non-zero
        pred_std = max(pred_std, 1e-6) # Use a small epsilon instead of min_std
        true_std = max(true_std, 1e-6)

        # Formula for KL(N(true||pred))
        # log(pred_std / true_std) + (true_std^2 + (true_mean - pred_mean)^2) / (2 * pred_std^2) - 0.5
        term1 = np.log(pred_std / true_std)
        term2 = (true_std**2 + (true_mean - pred_mean)**2) / (2 * pred_std**2)
        kl = term1 + term2 - 0.5

        # Return non-negative KL divergence
        return max(0, kl) # KL divergence should be >= 0

    def plot_results(self, results, probes_per_second, simulator_version=2, simulation_duration=100):
        """
        Plot the metrics for Base and Adaptive Probers over time in separate figures.
        Also export CPU time, total probes, etc., to a CSV file.
        """
        print(f"Plotting results for probes_per_second={probes_per_second}, "
              f"simulator_version={simulator_version}, simulation_duration={simulation_duration}")

        base_metrics_ts = results['base_prober']['metrics_per_timeslot']
        adaptive_metrics_ts = results['adaptive_prober']['metrics_per_timeslot']
        # Keep KL metrics for CSV export and final stats
        kl_metrics_ts = results['kl_prober']['metrics_per_timeslot']
        congestion_intervals = results['congestion_intervals']
        ground_truth_params = results['ground_truth_params']

        # Extract time slots (assuming they are consistent across probers)
        # Use the longest timeseries in case one prober finishes early or has no data
        all_times = set(m[0] for m in base_metrics_ts) | \
                    set(m[0] for m in adaptive_metrics_ts) | \
                    set(m[0] for m in kl_metrics_ts) # Include KL times for full range
        times = sorted(list(all_times))

        # Helper to get metric value for a specific time, defaulting to 0 or NaN
        def get_metric_at_time(metric_list, time, index, default=0.0):
             for m in metric_list:
                 if m[0] == time:
                     # Ensure the metric exists at this index
                     if index < len(m):
                         return m[index]
                     else:
                         # Handle cases where metric tuple might be shorter than expected
                         print(f"Warning: Metric tuple {m} at time {time} is shorter than expected index {index}. Returning default.")
                         return default
             return default

        # Extract data series, aligning by time
        base_probes = [get_metric_at_time(base_metrics_ts, t, 3) for t in times]
        adaptive_probes = [get_metric_at_time(adaptive_metrics_ts, t, 3) for t in times]
        # Keep KL probes for CSV/stats
        kl_probes = [get_metric_at_time(kl_metrics_ts, t, 3) for t in times]

        # Keep CPU times for CSV/stats
        base_cpu_times = [get_metric_at_time(base_metrics_ts, t, 4) for t in times]
        adaptive_cpu_times = [get_metric_at_time(adaptive_metrics_ts, t, 4) for t in times]
        kl_cpu_times = [get_metric_at_time(kl_metrics_ts, t, 4) for t in times]

        # Cumulative sums (keep all for CSV/stats)
        base_cum_probes = np.cumsum(base_probes)
        adaptive_cum_probes = np.cumsum(adaptive_probes)
        kl_cum_probes = np.cumsum(kl_probes)
        base_cum_cpu = np.cumsum(base_cpu_times)
        adaptive_cum_cpu = np.cumsum(adaptive_cpu_times)
        kl_cum_cpu = np.cumsum(kl_cpu_times)

        # Keep KL divergence calculations for CSV/stats
        true_mean = ground_truth_params['mean']
        true_std = ground_truth_params['std']
        def safe_kl_calc(metrics, t, true_mean, true_std):
            mean = get_metric_at_time(metrics, t, 1, default=None)
            std = get_metric_at_time(metrics, t, 2, default=None)
            # Use a small default std dev if missing or zero for KL calculation robustness
            if std is None or std <= 1e-6:
                 std = 0.01 # Or use self.min_std if accessible
            if mean is None:
                return np.nan # Return NaN if mean is missing
            # Ensure true_std is also positive
            safe_true_std = max(true_std, 1e-6)
            return self.calculate_kl_divergence(mean, std, true_mean, safe_true_std)

        base_kl_vs_gt = [safe_kl_calc(base_metrics_ts, t, true_mean, true_std) for t in times]
        adaptive_kl_vs_gt = [safe_kl_calc(adaptive_metrics_ts, t, true_mean, true_std) for t in times]
        kl_kl_vs_gt = [safe_kl_calc(kl_metrics_ts, t, true_mean, true_std) for t in times]


        # --- Plot 1: Probes per Simulation Frame ---
        fig1, ax1 = plt.subplots(figsize=(12, 6)) # Single plot figure

        ax1.plot(times, base_probes, label='Base', alpha=0.8, linestyle='-')
        ax1.plot(times, adaptive_probes, label='Adaptive', alpha=0.8, linestyle='--')
        # Highlight congestion intervals
        for start, end in congestion_intervals:
            ax1.axvspan(start, end, color='red', alpha=0.1, label='Congestion' if start == congestion_intervals[0][0] else "") # Label only once

        # ax1.set_title('Probes per Time Unit') # Removed axes title
        ax1.set_xlabel('Simulation Frames')
        ax1.set_ylabel('Probes Sent per Simulation Frame') # Updated Y label
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, simulation_duration)

        fig1.suptitle(f'Probes per Simulation Frame (Sim v{simulator_version}, Rate Limit={probes_per_second}pps)', fontsize=14) # Updated figure title
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to prevent suptitle overlap if needed

        # Saving Plot 1
        base_dir = '/Users/lukemciver/Network-Simulation-Group-Research-Project/active_monitoring_evolution/luke_probing/plots'
        plot_dir = os.path.join(base_dir, f'v{simulator_version}')
        os.makedirs(plot_dir, exist_ok=True)
        plot1_filename = f'probes_per_frame_duration_{simulation_duration}_rate_{probes_per_second}.png' # Updated filename
        plt.savefig(os.path.join(plot_dir, plot1_filename))
        print(f"---> Saved plot 1 to: {os.path.join(plot_dir, plot1_filename)}")
        plt.close(fig1) # Close the figure


        # --- Plot 2: Cumulative Probes Sent ---
        fig2, ax2 = plt.subplots(figsize=(12, 6)) # Second figure

        ax2.plot(times, base_cum_probes, label='Base', alpha=0.8, linestyle='-')
        ax2.plot(times, adaptive_cum_probes, label='Adaptive', alpha=0.8, linestyle='--')
        # Highlight congestion intervals
        for start, end in congestion_intervals:
            ax2.axvspan(start, end, color='red', alpha=0.1)

        # ax2.set_title('Cumulative Probes Sent per Time Unit') # Removed axes title
        ax2.set_xlabel('Simulation Frames')
        ax2.set_ylabel('Cumulative Probes Sent') # Updated Y label
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, simulation_duration)

        fig2.suptitle(f'Cumulative Probes Sent (Sim v{simulator_version}, Rate Limit={probes_per_second}pps)', fontsize=14) # Updated figure title
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to prevent suptitle overlap if needed

        # Saving Plot 2
        # (Use the same plot_dir as defined above)
        plot2_filename = f'cumulative_probes_duration_{simulation_duration}_rate_{probes_per_second}.png'
        plt.savefig(os.path.join(plot_dir, plot2_filename))
        print(f"---> Saved plot 2 to: {os.path.join(plot_dir, plot2_filename)}")
        plt.close(fig2) # Close the second figure


        # --- Export data series to CSV (Kept original data including KL) ---
        export_data = {
            'time_seconds': times,
            'base_probes_per_sec': base_probes,
            'adaptive_probes_per_sec': adaptive_probes,
            'kl_probes_per_sec': kl_probes,
            'base_cpu_time_per_sec': base_cpu_times,
            'adaptive_cpu_time_per_sec': adaptive_cpu_times,
            'kl_cpu_time_per_sec': kl_cpu_times,
            'base_cum_probes': base_cum_probes,
            'adaptive_cum_probes': adaptive_cum_probes,
            'kl_cum_probes': kl_cum_probes,
            'base_cum_cpu': base_cum_cpu,
            'adaptive_cum_cpu': adaptive_cum_cpu,
            'kl_cum_cpu': kl_cum_cpu,
            'base_kl_vs_gt': base_kl_vs_gt,
            'adaptive_kl_vs_gt': adaptive_kl_vs_gt,
            'kl_kl_vs_gt': kl_kl_vs_gt,
        }

        df_export = pd.DataFrame(export_data)
        # Ensure directory exists (redundant if plots saved, but safe)
        os.makedirs(plot_dir, exist_ok=True)
        csv_filename = f'metrics_duration_{simulation_duration}_probes_rate_{probes_per_second}.csv'
        df_export.to_csv(os.path.join(plot_dir, csv_filename), index=False)
        print(f"---> Exported CSV to: {os.path.join(plot_dir, csv_filename)}")

        # --- Print final sums (Kept original data including KL) ---
        final_base_cpu = float(base_cum_cpu[-1]) if len(base_cum_cpu) > 0 else 0
        final_adaptive_cpu = float(adaptive_cum_cpu[-1]) if len(adaptive_cum_cpu) > 0 else 0
        final_kl_cpu = float(kl_cum_cpu[-1]) if len(kl_cum_cpu) > 0 else 0
        final_base_probes = int(base_cum_probes[-1]) if len(base_cum_probes) > 0 else 0
        final_adaptive_probes = int(adaptive_cum_probes[-1]) if len(adaptive_cum_probes) > 0 else 0
        final_kl_probes = int(kl_cum_probes[-1]) if len(kl_cum_probes) > 0 else 0

        print(f"Final BaseProber CPU: {final_base_cpu:.4f}s, Probes: {final_base_probes}")
        print(f"Final AdaptiveProber CPU: {final_adaptive_cpu:.4f}s, Probes: {final_adaptive_probes}")
        print(f"Final KLProber CPU: {final_kl_cpu:.4f}s, Probes: {final_kl_probes}")

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
