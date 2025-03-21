import time
import random
import matplotlib.pyplot as plt
import numpy as np
import math
from passive_monitoring.time_binning.time_bin_sketch import TimeBinSketch
from passive_monitoring.time_binning.time_bin_monitoring import TimeBinMonitor
from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
from active_monitoring_evolution.ground_truth import GroundTruthNetwork



# if __name__ == '__main__':
#     # Create the ground truth network.
#     network = GroundTruthNetwork(paths="1")
#     # Instantiate the passive simulator.
#     passive = PassiveSimulator(network)
    
#     # Enable congestion simulation on the destination switch.
#     # This will simulate congestion intervals with higher drop probability and extra delay.
#     passive.enable_congestion_simulation(network.DESTINATION)
    
#     # Create the time bin monitor with a granulated bin size of 0.1 seconds.
#     tb_monitor = TimeBinMonitor(passive, bin_size=0.1)
#     tb_monitor.enable_monitoring()
    
#     # Run the traffic simulation: simulate for 10 seconds with an average interarrival time of 100 ms.
#     passive.simulate_traffic(duration_seconds=10, avg_interarrival_ms=100)
    
#     # Retrieve the histograms from the TimeBinMonitor.
#     source_hist = tb_monitor.get_source_histogram()
#     dest_hist = tb_monitor.get_destination_histogram()
#     print("Source (sent) histogram (bin -> count):", source_hist)
#     print("Destination (received) histogram (bin -> count):", dest_hist)
    
#     # --- Estimate the Delay Distribution via Cross-correlation ---
#     # Convert the histograms (dictionaries) to arrays over a common bin range.
#     all_bins = set(source_hist.keys()) | set(dest_hist.keys())
#     max_bin = max(all_bins)
#     source_arr = np.zeros(max_bin + 1)
#     dest_arr = np.zeros(max_bin + 1)
#     for b, count in source_hist.items():
#         source_arr[b] = count
#     for b, count in dest_hist.items():
#         dest_arr[b] = count

#     # Compute the cross-correlation between the destination and source arrays.
#     corr = np.correlate(dest_arr, source_arr, mode='full')
#     lags = np.arange(-len(source_arr)+1, len(source_arr))
#     # Convert lag (in bins) to time (seconds).
#     lag_times = lags * tb_monitor.bin_size

#     # The lag with the maximum correlation corresponds to the estimated average delay.
#     max_corr_idx = np.argmax(corr)
#     estimated_shift_bins = lags[max_corr_idx]
#     estimated_delay = estimated_shift_bins * tb_monitor.bin_size
#     print("Estimated average delay (seconds):", estimated_delay)
    
#     # --- Graphing the Data ---
#     # Convert histogram bin indices to actual time values.
#     source_bins = sorted(source_hist.keys())
#     source_time_bins = [b * tb_monitor.bin_size for b in source_bins]
#     source_counts = [source_hist[b] for b in source_bins]
    
#     dest_bins = sorted(dest_hist.keys())
#     dest_time_bins = [b * tb_monitor.bin_size for b in dest_bins]
#     dest_counts = [dest_hist[b] for b in dest_bins]
    
#     plt.figure(figsize=(14, 8))
    
#     # Plot Source and Destination Histograms.
#     plt.subplot(2, 1, 1)
#     plt.bar(source_time_bins, source_counts, width=tb_monitor.bin_size*0.8, color='skyblue', align='edge', label='Source (Sent)')
#     plt.bar(dest_time_bins, dest_counts, width=tb_monitor.bin_size*0.8, color='salmon', align='edge', label='Destination (Received)', alpha=0.7)
#     plt.xlabel("Time (seconds)")
#     plt.ylabel("Packet Count")
#     plt.title("Source and Destination Histograms (With Congestion)")
#     plt.legend()
    
#     # Plot the cross-correlation curve to visualize the delay shift.
#     plt.subplot(2, 1, 2)
#     plt.stem(lag_times, corr)
#     plt.xlabel("Delay (seconds)")
#     plt.ylabel("Cross-correlation")
#     plt.title("Cross-correlation (Estimated Delay Distribution)")
    
#     plt.tight_layout()
#     plt.show()


def compute_empirical_stats(hist, bin_size):
    """
    Given a histogram (a dict mapping bin indices to counts) and the bin size,
    compute the empirical mean and standard deviation.
    The histogram is assumed to represent a distribution over time (in seconds).
    """
    total = sum(hist.values())
    if total == 0:
        return 0.0, 0.0
    # Compute the mean (each bin index multiplied by bin_size gives the bin's time)
    mean = sum((bin_index * bin_size * count) for bin_index, count in hist.items()) / total
    # Compute the variance
    variance = sum(((bin_index * bin_size - mean) ** 2) * count for bin_index, count in hist.items()) / total
    std = math.sqrt(variance)
    return mean, std

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
    
    # Retrieve histograms.
    source_hist = tb_monitor.get_source_histogram()
    dest_hist = tb_monitor.get_destination_histogram()
    print("Source (sent) histogram (bin -> count):", source_hist)
    print("Destination (received) histogram (bin -> count):", dest_hist)
    
    # --- Compute Empirical Delay Statistics from the Destination Histogram ---
    # We assume that the destination histogram approximates the delay distribution.
    empirical_mean, empirical_std = compute_empirical_stats(dest_hist, tb_monitor.bin_size)
    print("Empirical mean delay (seconds):", empirical_mean)
    print("Empirical std of delay (seconds):", empirical_std)
    
    # --- Compare with the Underlying Normal Distribution ---
    # The passive simulator's compare_distribution_parameters function uses the ground truth network's
    # delay parameters (obtained via network.get_distribution_parameters) as the "actual" distribution.
    # We pass our empirical estimates as the "predicted" distribution.
    kl_div = passive.compare_distribution_parameters(empirical_mean, empirical_std)
    print("KL divergence:", kl_div)
    
    # --- Graphing the Histograms and Cross-correlation ---
    # For histograms, convert bin indices to time values.
    source_bins = sorted(source_hist.keys())
    source_time_bins = [b * tb_monitor.bin_size for b in source_bins]
    source_counts = [source_hist[b] for b in source_bins]
    
    dest_bins = sorted(dest_hist.keys())
    dest_time_bins = [b * tb_monitor.bin_size for b in dest_bins]
    dest_counts = [dest_hist[b] for b in dest_bins]
    
    plt.figure(figsize=(14, 8))
    
    plt.subplot(3, 1, 1)
    plt.bar(source_time_bins, source_counts, width=tb_monitor.bin_size*0.8, color='skyblue', align='edge', label='Source (Sent)')
    plt.bar(dest_time_bins, dest_counts, width=tb_monitor.bin_size*0.8, color='salmon', align='edge', label='Destination (Received)', alpha=0.7)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Packet Count")
    plt.title("Source and Destination Histograms (With Congestion)")
    plt.legend()
    
    # Compute cross-correlation between destination and source histograms.
    all_bins = set(source_hist.keys()) | set(dest_hist.keys())
    max_bin = max(all_bins)
    source_arr = np.zeros(max_bin + 1)
    dest_arr = np.zeros(max_bin + 1)
    for b, count in source_hist.items():
        source_arr[b] = count
    for b, count in dest_hist.items():
        dest_arr[b] = count
    corr = np.correlate(dest_arr, source_arr, mode='full')
    lags = np.arange(-len(source_arr) + 1, len(source_arr))
    lag_times = lags * tb_monitor.bin_size
    
    plt.subplot(3, 1, 2)
    plt.stem(lag_times, corr)
    plt.xlabel("Delay (seconds)")
    plt.ylabel("Cross-correlation")
    plt.title("Cross-correlation (Estimated Delay Distribution)")
    
    # Plot the empirical delay distribution (as derived from the destination histogram)
    # as a bar chart.
    # For this, we simply plot the normalized counts of the destination histogram.
    total_received = sum(dest_counts)
    norm_dest_counts = [c / total_received for c in dest_counts] if total_received else dest_counts
    plt.subplot(3, 1, 3)
    plt.bar(dest_time_bins, norm_dest_counts, width=tb_monitor.bin_size*0.8, color='lightgreen', align='edge')
    plt.xlabel("Delay (seconds)")
    plt.ylabel("Normalized Frequency")
    plt.title("Empirical Delay Distribution (Destination Histogram)")
    
    plt.tight_layout()
    plt.show()