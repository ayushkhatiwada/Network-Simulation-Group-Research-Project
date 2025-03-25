import time
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from passive_monitoring.time_binning.time_bin_sketch import TimeBinSketch
from passive_monitoring.time_binning.time_bin_monitoring import TimeBinMonitor
from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
from active_monitoring_evolution.ground_truth import GroundTruthNetwork

def compute_delay_stats_from_crosscorr(lag_times, corr):
    """
    Given lag_times (in seconds) and their corresponding cross-correlation values (weights),
    compute a weighted mean and standard deviation.
    """
    total_weight = np.sum(corr)
    if total_weight == 0:
        return 0.0, 0.0
    estimated_mean = np.sum(lag_times * corr) / total_weight
    estimated_variance = np.sum(corr * (lag_times - estimated_mean)**2) / total_weight
    estimated_std = math.sqrt(estimated_variance)
    return estimated_mean, estimated_std

if __name__ == '__main__':
    # Create the ground truth network.
    network = GroundTruthNetwork(paths="1")
    # Instantiate the passive simulator.
    passive = PassiveSimulator(network)
    
    # Enable congestion simulation on the destination switch.
    # This will cause packets to be dropped and extra delay added during congestion intervals.
    passive.enable_congestion_simulation(network.DESTINATION)
    
    # Create the time bin monitor with a bin size of 0.1 seconds.
    tb_monitor = TimeBinMonitor(passive, bin_size=0.1)
    tb_monitor.enable_monitoring()
    
    # Run the traffic simulation: simulate for 10 seconds with an average interarrival time of 100 ms.
    passive.simulate_traffic(duration_seconds=10, avg_interarrival_ms=100)
    
    # Retrieve the histograms.
    source_hist = tb_monitor.get_source_histogram()
    dest_hist = tb_monitor.get_destination_histogram()
    print("Source (sent) histogram (bin -> count):", source_hist)
    print("Destination (received) histogram (bin -> count):", dest_hist)
    
    # --- Compute Cross-correlation ---
    # Convert the histograms (dicts) into arrays over a common range of bins.
    all_bins = set(source_hist.keys()) | set(dest_hist.keys())
    max_bin = max(all_bins)
    source_arr = np.zeros(max_bin + 1)
    dest_arr = np.zeros(max_bin + 1)
    for b, count in source_hist.items():
        source_arr[b] = count
    for b, count in dest_hist.items():
        dest_arr[b] = count

    # Compute full cross-correlation between destination and source arrays.
    corr = np.correlate(dest_arr, source_arr, mode='full')
    lags = np.arange(-len(source_arr) + 1, len(source_arr))
    lag_times = lags * tb_monitor.bin_size  # convert lag (in bins) to seconds

    # Compute estimated delay stats from the cross-correlation curve.
    est_mean_delay, est_std_delay = compute_delay_stats_from_crosscorr(lag_times, corr)
    print("Estimated mean delay from cross-correlation (seconds):", est_mean_delay)
    print("Estimated std delay from cross-correlation (seconds):", est_std_delay)
    
    # --- Compare with the Underlying Normal Distribution ---
    # Retrieve the actual distribution parameters from the ground truth network.
    true_params = network.get_distribution_parameters(network.SOURCE, network.DESTINATION)
    print("Underlying delay parameters:", true_params)
    # Compare the estimated parameters (predicted) to the actual ones using KL divergence.
    kl_div = passive.compare_distribution_parameters(est_mean_delay, est_std_delay)
    print("KL divergence:", kl_div)
    
    # --- Graphing ---
    # Convert histogram bin indices to time values (in seconds).
    source_bins = sorted(source_hist.keys())
    source_time_bins = [b * tb_monitor.bin_size for b in source_bins]
    source_counts = [source_hist[b] for b in source_bins]
    
    dest_bins = sorted(dest_hist.keys())
    dest_time_bins = [b * tb_monitor.bin_size for b in dest_bins]
    dest_counts = [dest_hist[b] for b in dest_bins]
    
    plt.figure(figsize=(14, 12))
    
    # Plot source and destination histograms.
    plt.subplot(3, 1, 1)
    plt.bar(source_time_bins, source_counts, width=tb_monitor.bin_size*0.8, color='skyblue', align='edge', label='Source (Sent)')
    plt.bar(dest_time_bins, dest_counts, width=tb_monitor.bin_size*0.8, color='salmon', align='edge', label='Destination (Received)', alpha=0.7)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Packet Count")
    plt.title("Source and Destination Histograms (With Congestion)")
    plt.legend()
    
    # Plot cross-correlation.
    plt.subplot(3, 1, 2)
    plt.stem(lag_times, corr)
    plt.xlabel("Delay (seconds)")
    plt.ylabel("Cross-correlation")
    plt.title("Cross-correlation (Estimated Delay Distribution)")
    
    # Plot the normalized empirical delay distribution (from cross-correlation weights).
    total_corr = np.sum(corr)
    norm_corr = corr / total_corr if total_corr != 0 else corr
    plt.subplot(3, 1, 3)
    plt.bar(lag_times, norm_corr, width=tb_monitor.bin_size*0.8, color='lightgreen', align='edge')
    plt.xlabel("Delay (seconds)")
    plt.ylabel("Normalized Frequency")
    plt.title("Empirical Delay Distribution (From Cross-correlation)")
    
    plt.tight_layout()
    plt.show()
