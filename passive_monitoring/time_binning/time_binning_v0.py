import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
from active_monitoring_evolution.ground_truth import GroundTruthNetwork
from passive_monitoring.time_binning.time_bin_monitoring import TimeBinMonitor
from passive_monitoring.time_binning.delay_estimator import DelayDistributionEstimator

if __name__ == '__main__':
    # Set up the ground truth network and passive simulator.
    network = GroundTruthNetwork(paths="1")
    passive = PassiveSimulator(network)
    
    # Configure the time-bin monitoring system.
    bin_size = 0.0001  # seconds per bin
    start_time = time.time()
    tb_monitor = TimeBinMonitor(passive, bin_size, start_time)
    tb_monitor.enable_monitoring()
    
    # Simulate traffic.
    simulation_duration = 100  # seconds
    avg_interarrival_ms = 20
    passive.simulate_traffic(duration_seconds=simulation_duration, avg_interarrival_ms=avg_interarrival_ms)
    
    # Retrieve histograms from both the discrete and sliding sketches.
    source_hist = tb_monitor.get_source_histogram()
    dest_hist = tb_monitor.get_destination_histogram()
    source_sliding_hist = tb_monitor.get_source_sliding_histogram()
    dest_sliding_hist = tb_monitor.get_destination_sliding_histogram()
    
    # Define matching parameters.
    window_size = 10  # in bin units; adjust as needed
    alpha = 1         # weighting parameter; adjust to penalize larger differences
    cost_function = 'exponential'  # or 'quadratic'
    
    # Create a DelayDistributionEstimator and update it using both histogram pairs.
    estimator = DelayDistributionEstimator()
    estimator.update_from_histograms(source_hist, dest_hist, bin_size, window_size, alpha, cost_function)
    estimator.update_from_histograms(source_sliding_hist, dest_sliding_hist, bin_size, window_size, alpha, cost_function)
    
    # Retrieve delay values and convert to milliseconds.
    delays = estimator.get_all_delays()
    delays_ms = [d * 1000 for d in delays]
    
    # Plot the combined delay distribution histogram.
    plt.figure(figsize=(8, 5))
    n, bins, patches = plt.hist(delays_ms, bins=20, density=True, alpha=0.6, edgecolor='black')
    plt.xlabel('Delay (ms)')
    plt.ylabel('Probability Density')
    plt.title('Combined Delay Distribution with Weighted Dynamic Matching')
    
    # Fit a normal distribution to the delays (if any) and compare with ground truth.
    if delays_ms:
        mu_ms, std_ms = norm.fit(delays_ms)
        print("Normal fit parameters (ms):", mu_ms, std_ms)
        kl_div = passive.compare_distribution_parameters(mu_ms, std_ms)
        print("KL divergence:", kl_div)
        
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu_ms, std_ms)
        plt.plot(x, p, 'r', linewidth=2,
                 label=f'Normal Fit\nμ={mu_ms:.4f} ms\nσ={std_ms:.4f} ms')
        plt.legend()
    
    plt.show()
