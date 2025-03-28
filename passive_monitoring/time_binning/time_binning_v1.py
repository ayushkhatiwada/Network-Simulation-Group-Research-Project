import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
from active_monitoring_evolution.ground_truth import GroundTruthNetwork
from passive_monitoring.time_binning.time_bin_monitoring import TimeBinMonitor
from passive_monitoring.time_binning.delay_estimator import DelayDistributionEstimator

if __name__ == '__main__':
    network = GroundTruthNetwork(paths="1")
    passive = PassiveSimulator(network)
    
    # Set a drop probability (e.g., 20%) at the destination.
    drop_probability = 0.2
    passive.set_drop_probability(network.DESTINATION, drop_probability)
    
    # Configure time-bin monitoring.
    bin_size = 0.001  
    start_time = time.time()
    tb_monitor = TimeBinMonitor(passive, bin_size, start_time)
    tb_monitor.enable_monitoring()
    
    # Simulate traffic.
    simulation_duration = 100  
    avg_interarrival_ms = 20   
    passive.simulate_traffic(duration_seconds=simulation_duration, avg_interarrival_ms=avg_interarrival_ms)
    
    # Retrieve histograms.
    source_hist = tb_monitor.get_source_histogram()
    dest_hist = tb_monitor.get_destination_histogram()
    source_sliding_hist = tb_monitor.get_source_sliding_histogram()
    dest_sliding_hist = tb_monitor.get_destination_sliding_histogram()
    
    # Estimate delays using fallback with smoothing.
    window_size = 10  
    alpha = 1        
    cost_function = 'exponential'
    smooth_kernel = 3
    
    estimator = DelayDistributionEstimator()
    estimator.update_from_histograms(source_hist, dest_hist, bin_size, window_size, alpha, cost_function, use_fallback=True, smooth_kernel=smooth_kernel)
    estimator.update_from_histograms(source_sliding_hist, dest_sliding_hist, bin_size, window_size, alpha, cost_function, use_fallback=True, smooth_kernel=smooth_kernel)
    
    delays = estimator.get_all_delays()
    delays = [d for d in delays if d is not None and not np.isnan(d)]
    delays_ms = [d * 1000 for d in delays]
    
    plt.figure(figsize=(8, 5))
    plt.hist(delays_ms, bins=20, density=True, alpha=0.6, edgecolor='black')
    plt.xlabel("Delay (ms)")
    plt.ylabel("Probability Density")
    plt.title("Delay Distribution (With Drops & Fallback)")
    
    if delays_ms:
        mu_ms, std_ms = norm.fit(delays_ms)
        print("Normal fit parameters (ms):", mu_ms, std_ms)
        if std_ms < 1e-6:
            print("Standard deviation too small; skipping KL divergence computation.")
        else:
            kl_div = passive.compare_distribution_parameters(mu_ms, std_ms)
            print("KL divergence:", kl_div)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        pdf = norm.pdf(x, mu_ms, std_ms)
        plt.plot(x, pdf, 'r', linewidth=2,
                 label=f'Normal Fit\nμ={mu_ms:.2f} ms, σ={std_ms:.2f} ms')
        plt.legend()
    
    plt.show()
    
    # Alternative: Global offset via cross-correlation.
    all_bins = set(source_hist.keys()) | set(dest_hist.keys())
    if all_bins:
        max_bin = max(all_bins)
        source_arr = np.zeros(max_bin+1)
        dest_arr = np.zeros(max_bin+1)
        for b, count in source_hist.items():
            source_arr[b] = count
        for b, count in dest_hist.items():
            dest_arr[b] = count
        
        corr = np.correlate(dest_arr, source_arr, mode='full')
        lags = np.arange(-len(source_arr)+1, len(source_arr))
        lag_times = lags * bin_size
        max_corr_idx = np.argmax(corr)
        estimated_shift_bins = lags[max_corr_idx]
        estimated_delay = estimated_shift_bins * bin_size
        print("Estimated average delay via cross-correlation (s):", estimated_delay)
        
        plt.figure(figsize=(8, 5))
        plt.stem(lag_times, corr, use_line_collection=True)
        plt.xlabel("Delay (s)")
        plt.ylabel("Cross-correlation")
        plt.title("Cross-Correlation of Histograms")
        plt.show()
