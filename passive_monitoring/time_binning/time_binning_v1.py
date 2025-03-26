import time
import random
import matplotlib.pyplot as plt
import numpy as np
from passive_monitoring.time_binning.time_bin_sketch import TimeBinSketch
from passive_monitoring.time_binning.time_bin_monitoring import TimeBinMonitor
from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
from active_monitoring_evolution.ground_truth import GroundTruthNetwork

if __name__ == '__main__':
    network = GroundTruthNetwork(paths="1")
    passive = PassiveSimulator(network)
    
    # Set a drop probability on the destination switch to trigger some drops.
    passive.set_drop_probability(network.DESTINATION, 0.2)
    
    tb_monitor = TimeBinMonitor(passive, bin_size=0.1)
    tb_monitor.enable_monitoring()
    
    passive.simulate_traffic(duration_seconds=10, avg_interarrival_ms=100)
    
    source_hist = tb_monitor.get_source_histogram()
    dest_hist = tb_monitor.get_destination_histogram()
    print("Source (sent) histogram (bin -> count):", source_hist)
    print("Destination (received) histogram (bin -> count):", dest_hist)
    
    # --- Estimate the Delay Distribution via Cross-correlation ---
    # Convert the histograms (dictionaries) into arrays over a common range of bins.
    all_bins = set(source_hist.keys()) | set(dest_hist.keys())
    max_bin = max(all_bins)
    source_arr = np.zeros(max_bin+1)
    dest_arr = np.zeros(max_bin+1)
    for b, count in source_hist.items():
        source_arr[b] = count
    for b, count in dest_hist.items():
        dest_arr[b] = count

    # Compute cross-correlation between destination and source histograms.
    corr = np.correlate(dest_arr, source_arr, mode='full')
    lags = np.arange(-len(source_arr)+1, len(source_arr))
    lag_times = lags * tb_monitor.bin_size

    # The lag corresponding to the maximum correlation is our estimated average delay.
    max_corr_idx = np.argmax(corr)
    estimated_shift_bins = lags[max_corr_idx]
    estimated_delay = estimated_shift_bins * tb_monitor.bin_size
    print("Estimated average delay (seconds):", estimated_delay)
    
    # --- Graphing ---
    # Convert bin indices to time (seconds) for the histograms.
    source_bins = sorted(source_hist.keys())
    source_time_bins = [b * tb_monitor.bin_size for b in source_bins]
    source_counts = [source_hist[b] for b in source_bins]
    
    dest_bins = sorted(dest_hist.keys())
    dest_time_bins = [b * tb_monitor.bin_size for b in dest_bins]
    dest_counts = [dest_hist[b] for b in dest_bins]
    
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    plt.bar(source_time_bins, source_counts, width=tb_monitor.bin_size*0.8, color='skyblue', align='edge', label='Source (Sent)')
    plt.bar(dest_time_bins, dest_counts, width=tb_monitor.bin_size*0.8, color='salmon', align='edge', label='Destination (Received)', alpha=0.7)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Packet Count")
    plt.title("Source and Destination Histograms")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.stem(lag_times, corr)
    plt.xlabel("Delay (seconds)")
    plt.ylabel("Cross-correlation")
    plt.title("Cross-correlation (Estimated Delay Distribution)")
    
    plt.tight_layout()
    plt.show()