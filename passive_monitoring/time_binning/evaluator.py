#!/usr/bin/env python3
# Network Delay Estimator Evaluation Script
# This script runs tests with varying simulation duration and packet rates,
# collects metrics, and saves the results to CSV files.
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
import random
import sys
import pandas as pd
from datetime import datetime

# Import all the necessary modules from the original script
# These need to be the same imports as in your original script
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy import signal
import random

# Import your custom modules exactly as they're imported in the original script
from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
from active_monitoring_evolution.ground_truth import GroundTruthNetwork
from passive_monitoring.time_binning.time_bin_monitoring import TimeBinMonitor
from passive_monitoring.time_binning.delay_estimator import DelayDistributionEstimator, compute_global_offset

# Import the functions directly from your paste.txt
# Make sure paste.txt is in the same directory
with open('passive_monitoring/time_binning/time_binning_v2.py', 'r') as f:
    exec(f.read())

# This will make extract_delays_with_early_exit and get_dropout_stats available in the global scope

def run_simulation(simulation_duration, avg_interarrival_ms, kl_threshold=0.05):
    """
    Run a single simulation with the specified parameters.
    
    Args:
        simulation_duration: Duration of the simulation in seconds
        avg_interarrival_ms: Average packet interarrival time in ms
        kl_threshold: Target KL divergence threshold
        
    Returns:
        Dictionary containing all measured metrics
    """
    # Set random seed for reproducibility (always use 42 as requested)
    np.random.seed(42)
    random.seed(42)
    
    # Calculate packets per second for logging
    packets_per_second = 1000 / avg_interarrival_ms
    
    print(f"\n--- Running simulation: duration={simulation_duration}s, "
          f"interarrival={avg_interarrival_ms}ms ({packets_per_second:.1f} pps) ---")
    
    # Initialize network and simulator
    network = GroundTruthNetwork(paths="1")
    passive = PassiveSimulator(network)
    
    # Configure simulation parameters
    bin_size = 0.0001  # 0.1ms bin size
    
    # Start timing the entire experiment
    experiment_start_time = time.time()
    
    # Enable congestion simulation
    passive.enable_congestion_simulation(network.DESTINATION, simulation_duration)
    
    # Start monitoring
    start_time = time.time()
    tb_monitor = TimeBinMonitor(passive, bin_size, start_time)
    tb_monitor.enable_monitoring()
    
    # Simulate traffic
    passive.simulate_traffic(duration_seconds=simulation_duration, avg_interarrival_ms=avg_interarrival_ms)
    
    # Measure simulation time
    simulation_time = time.time() - start_time
    
    # Retrieve histograms
    source_hist = tb_monitor.get_source_histogram()
    dest_hist = tb_monitor.get_destination_histogram()
    source_sliding_hist = tb_monitor.get_source_sliding_histogram()
    dest_sliding_hist = tb_monitor.get_destination_sliding_histogram()
    
    # Calculate dropout statistics
    measured_drop_rate, affected_bins = get_dropout_stats(source_hist, dest_hist)
    
    # Count total packets
    total_source_packets = sum(source_hist.values())
    total_dest_packets = sum(dest_hist.values())
    
    # Start timing the estimation process
    estimation_start_time = time.time()
    
    # Extract delays with early exit when threshold is met
    delays_ms, best_params = extract_delays_with_early_exit(
        source_hist, dest_hist, 
        source_sliding_hist, dest_sliding_hist, 
        bin_size,
        kl_threshold
    )
    
    # Measure estimation time
    estimation_time = time.time() - estimation_start_time
    
    # Initialize results dictionary with basic info
    results = {
        'simulation_duration': simulation_duration,
        'avg_interarrival_ms': avg_interarrival_ms,
        'packets_per_second': packets_per_second,
        'kl_threshold': kl_threshold,
        'simulation_time': simulation_time,
        'estimation_time': estimation_time,
        'total_processing_time': time.time() - experiment_start_time,
        'total_source_packets': total_source_packets,
        'total_dest_packets': total_dest_packets,
        'measured_drop_rate': measured_drop_rate,
        'affected_bins': affected_bins,
        'affected_bins_percentage': affected_bins / len(source_hist) if source_hist else 0,
        'num_valid_delay_samples': len(delays_ms) if delays_ms else 0
    }
    
    # Get the underlying true distribution parameters
    true_params = network.get_distribution_parameters()[0][2]
    true_mu = true_params["mean"]
    true_std = true_params["std"]
    
    results['true_mean'] = true_mu
    results['true_std'] = true_std
    
    # If no delays were extracted, mark remaining metrics as N/A
    if not delays_ms or len(delays_ms) < 5:
        results['est_mean'] = None
        results['est_std'] = None
        results['mean_error'] = None
        results['std_error'] = None
        results['mean_error_percentage'] = None
        results['std_error_percentage'] = None
        results['kl_divergence'] = None
        results['target_reached'] = False
        
        # Add best parameter info if available
        if best_params:
            for key, value in best_params.items():
                if key not in ['mean', 'std', 'kl', 'source']:
                    results[f'best_{key}'] = value
        
        print("WARNING: No valid delay data was computed.")
        return results
    
    # Fit the distribution to the raw delays
    est_mu, est_std = norm.fit(delays_ms)
    
    # Calculate KL divergence
    kl_div = passive.compare_distribution_parameters(est_mu, est_std)
    
    # Add estimation metrics to results
    results['est_mean'] = est_mu
    results['est_std'] = est_std
    results['mean_error'] = abs(est_mu - true_mu)
    results['std_error'] = abs(est_std - true_std)
    results['mean_error_percentage'] = abs(est_mu - true_mu) / true_mu * 100
    results['std_error_percentage'] = abs(est_std - true_std) / true_std * 100
    results['kl_divergence'] = kl_div
    results['target_reached'] = kl_div <= kl_threshold
    
    # Add best parameter info
    for key, value in best_params.items():
        if key not in ['mean', 'std', 'kl', 'source']:
            results[f'best_{key}'] = value
    
    # Print summary
    print(f"Results summary:")
    print(f"  Simulation: {simulation_duration}s, {packets_per_second:.1f} pps")
    print(f"  Processing time: {results['total_processing_time']:.2f}s")
    print(f"  Drop rate: {measured_drop_rate:.2%}")
    print(f"  Valid delay samples: {len(delays_ms)}")
    print(f"  True μ/σ: {true_mu:.2f}ms/{true_std:.2f}ms")
    print(f"  Est μ/σ: {est_mu:.2f}ms/{est_std:.2f}ms")
    print(f"  Mean error: {results['mean_error']:.2f}ms ({results['mean_error_percentage']:.2f}%)")
    print(f"  KL divergence: {kl_div:.6f}" + 
          (" ✓" if results['target_reached'] else " ✗"))
    
    return results

def run_duration_tests(output_file, durations, avg_interarrival_ms=20, kl_threshold=0.05):
    """
    Run tests with varying simulation durations.
    
    Args:
        output_file: Path to the output CSV file
        durations: List of simulation durations to test (in seconds)
        avg_interarrival_ms: Fixed packet interarrival time to use
        kl_threshold: Target KL divergence threshold
    """
    print(f"\n=== RUNNING DURATION TESTS (fixed {avg_interarrival_ms}ms interarrival) ===\n")
    
    all_results = []
    
    for duration in durations:
        results = run_simulation(duration, avg_interarrival_ms, kl_threshold)
        all_results.append(results)
    
    # Save results to CSV
    if all_results:
        keys = all_results[0].keys()
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\nDuration test results saved to: {output_file}")
    else:
        print("No results to save.")
    
    return all_results

def run_pps_tests(output_file, packets_per_second_list, simulation_duration=10, kl_threshold=0.05):
    """
    Run tests with varying packet rates.
    
    Args:
        output_file: Path to the output CSV file
        packets_per_second_list: List of packet rates to test (in packets per second)
        simulation_duration: Fixed simulation duration to use
        kl_threshold: Target KL divergence threshold
    """
    print(f"\n=== RUNNING PACKET RATE TESTS (fixed {simulation_duration}s duration) ===\n")
    
    all_results = []
    
    for pps in packets_per_second_list:
        # Convert PPS to interarrival time in milliseconds
        avg_interarrival_ms = 1000 / pps
        
        results = run_simulation(simulation_duration, avg_interarrival_ms, kl_threshold)
        all_results.append(results)
    
    # Save results to CSV
    if all_results:
        keys = all_results[0].keys()
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\nPacket rate test results saved to: {output_file}")
    else:
        print("No results to save.")
    
    return all_results

def create_plots(duration_results, pps_results, output_dir):
    """
    Create and save plots from the test results.
    
    Args:
        duration_results: List of result dictionaries from duration tests
        pps_results: List of result dictionaries from packet rate tests
        output_dir: Directory to save plots in
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to DataFrames for easier plotting
    if duration_results:
        df_duration = pd.DataFrame(duration_results)
        
        # Plot KL divergence vs duration
        plt.figure(figsize=(10, 6))
        plt.plot(df_duration['simulation_duration'], df_duration['kl_divergence'], 'o-', linewidth=2)
        plt.axhline(y=df_duration['kl_threshold'].iloc[0], color='r', linestyle='--', 
                   label=f"Target threshold: {df_duration['kl_threshold'].iloc[0]}")
        plt.xlabel('Simulation Duration (s)')
        plt.ylabel('KL Divergence')
        plt.title('KL Divergence vs. Simulation Duration')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'kl_vs_duration.png'))
        
        # Plot estimation error vs duration
        plt.figure(figsize=(10, 6))
        plt.plot(df_duration['simulation_duration'], df_duration['mean_error_percentage'], 'o-', 
                linewidth=2, label='Mean Error (%)')
        plt.plot(df_duration['simulation_duration'], df_duration['std_error_percentage'], 's-', 
                linewidth=2, label='Std Dev Error (%)')
        plt.xlabel('Simulation Duration (s)')
        plt.ylabel('Error (%)')
        plt.title('Estimation Error vs. Simulation Duration')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_vs_duration.png'))
        
        # Plot number of valid samples vs duration
        plt.figure(figsize=(10, 6))
        plt.plot(df_duration['simulation_duration'], df_duration['num_valid_delay_samples'], 'o-', linewidth=2)
        plt.xlabel('Simulation Duration (s)')
        plt.ylabel('Number of Valid Delay Samples')
        plt.title('Valid Samples vs. Simulation Duration')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'samples_vs_duration.png'))
    
    if pps_results:
        df_pps = pd.DataFrame(pps_results)
        
        # Plot KL divergence vs packet rate
        plt.figure(figsize=(10, 6))
        plt.plot(df_pps['packets_per_second'], df_pps['kl_divergence'], 'o-', linewidth=2)
        plt.axhline(y=df_pps['kl_threshold'].iloc[0], color='r', linestyle='--', 
                   label=f"Target threshold: {df_pps['kl_threshold'].iloc[0]}")
        plt.xlabel('Packets per Second')
        plt.ylabel('KL Divergence')
        plt.title('KL Divergence vs. Packet Rate')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'kl_vs_pps.png'))
        
        # Plot drop rate vs packet rate
        plt.figure(figsize=(10, 6))
        plt.plot(df_pps['packets_per_second'], df_pps['measured_drop_rate'] * 100, 'o-', linewidth=2)
        plt.xlabel('Packets per Second')
        plt.ylabel('Drop Rate (%)')
        plt.title('Drop Rate vs. Packet Rate')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'drop_rate_vs_pps.png'))
        
        # Plot estimation error vs packet rate
        plt.figure(figsize=(10, 6))
        plt.plot(df_pps['packets_per_second'], df_pps['mean_error_percentage'], 'o-', 
                linewidth=2, label='Mean Error (%)')
        plt.plot(df_pps['packets_per_second'], df_pps['std_error_percentage'], 's-', 
                linewidth=2, label='Std Dev Error (%)')
        plt.xlabel('Packets per Second')
        plt.ylabel('Error (%)')
        plt.title('Estimation Error vs. Packet Rate')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_vs_pps.png'))
    
    print(f"Plots saved to directory: {output_dir}")

def main():
    # Create timestamp for unique output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"delay_estimation_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define test parameters
    durations = [1, 10, 15, 20, 30]  # Simulation durations in seconds
    packet_rates = [10, 25, 50, 75, 100, 150, 200]  # Packets per second
    
    # Run duration tests
    duration_output_file = os.path.join(output_dir, "duration_test_results.csv")
    duration_results = run_duration_tests(duration_output_file, durations)
    
    # Run packet rate tests
    pps_output_file = os.path.join(output_dir, "packet_rate_test_results.csv")
    pps_results = run_pps_tests(pps_output_file, packet_rates)
    
    # Create and save plots
    create_plots(duration_results, pps_results, output_dir)
    
    print(f"\nAll tests completed. Results saved to directory: {output_dir}")

if __name__ == "__main__":
    main()