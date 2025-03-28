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
    
    # Configure time-bin monitoring.
    bin_size = 0.0001 
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
    
    # Estimate delays using original matching
    window_size = 10 
    alpha = 1       
    cost_function = 'exponential'
    
    estimator = DelayDistributionEstimator()
    estimator.update_from_histograms(source_hist, dest_hist, bin_size, window_size, alpha, cost_function, use_fallback=False)
    estimator.update_from_histograms(source_sliding_hist, dest_sliding_hist, bin_size, window_size, alpha, cost_function, use_fallback=False)
    
    delays = estimator.get_all_delays()
    delays_ms = [d * 1000 for d in delays]
    
    plt.figure(figsize=(8, 5))
    plt.hist(delays_ms, bins=20, density=True, alpha=0.6, edgecolor='black')
    plt.xlabel("Delay (ms)")
    plt.ylabel("Probability Density")
    plt.title("Delay Distribution (No Drops, Weighted Matching)")
    
    if delays_ms:
        mu_ms, std_ms = norm.fit(delays_ms)
        print("Normal fit parameters (ms):", mu_ms, std_ms)
        kl_div = passive.compare_distribution_parameters(mu_ms, std_ms)
        print("KL divergence:", kl_div)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        pdf = norm.pdf(x, mu_ms, std_ms)
        plt.plot(x, pdf, 'r', linewidth=2,
                 label=f'Normal Fit\nμ={mu_ms:.4f} ms, σ={std_ms:.4f} ms')
        plt.legend()
    plt.show()
