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
    # simulation_duration = 100
    # avg_interarrival_ms = 20
    simulation_duration = 10
    avg_interarrival_ms = 20
    passive.simulate_traffic(duration_seconds=simulation_duration, avg_interarrival_ms=avg_interarrival_ms)
    
    # Retrieve histograms.
    source_hist = tb_monitor.get_source_histogram()
    dest_hist = tb_monitor.get_destination_histogram()
    source_sliding_hist = tb_monitor.get_source_sliding_histogram()
    dest_sliding_hist = tb_monitor.get_destination_sliding_histogram()
    
    # Estimate delays using original matching
    window_size = 9
    alpha = 2
    cost_function = 'exponential'
    
    estimator = DelayDistributionEstimator()
    estimator.update_from_histograms(source_hist, dest_hist, bin_size, window_size, alpha, cost_function, use_fallback=False)
    estimator.update_from_histograms(source_sliding_hist, dest_sliding_hist, bin_size, window_size, alpha, cost_function, use_fallback=False)
    
    delays = estimator.get_all_delays()
    delays_ms = [d * 1000 for d in delays]

    if not delays_ms:
        print("No delay data was computed.")
        exit()
    
    # estimate the normal distribution
    est_mu, est_std = norm.fit(delays_ms)
    print("Estimated normal fit parameters (ms):", est_mu, est_std)
    kl_div = passive.compare_distribution_parameters(est_mu, est_std)
    print("KL divergence:", kl_div)
    
    # get the underlying distribution
    true_params = network.get_distribution_parameters()[0][2]
    true_mu = true_params["mean"]
    true_std = true_params["std"]
    print("True underlying distribution parameters (ms):", true_mu, true_std)
    
    # get graphs
    lower_bound = min(est_mu - 4*est_std, true_mu - 4*true_std)
    upper_bound = max(est_mu + 4*est_std, true_mu + 4*true_std)
    x = np.linspace(lower_bound, upper_bound, 200)
    
    # Compute PDFs.
    est_pdf = norm.pdf(x, est_mu, est_std)
    true_pdf = norm.pdf(x, true_mu, true_std)
    
    # plot on the same plane
    plt.figure(figsize=(8, 5))
    plt.plot(x, est_pdf, 'r-', linewidth=2,
             label=f'Estimated Normal\nμ={est_mu:.4f} ms, σ={est_std:.4f} ms')
    plt.plot(x, true_pdf, 'b--', linewidth=2,
             label=f'True Normal\nμ={true_mu:.4f} ms, σ={true_std:.4f} ms')
    plt.xlabel("Delay (ms)")
    plt.ylabel("Probability Density")
    plt.title("Estimated vs. True Delay Distributions")
    plt.legend()
    plt.show()