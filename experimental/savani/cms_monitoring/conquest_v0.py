import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import math

from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
from active_monitoring_evolution.ground_truth import GroundTruthNetwork
from experimental.savani.cms_monitoring.conquest_monitor import ConQuestMonitor
from experimental.savani.cms_monitoring.delay_estimator import DelayDistributionEstimator


def compute_kl_divergence(est_mu, est_std, true_mu, true_std):
    # Compute KL divergence between two normal distributions.
    return math.log(est_std / true_std) + ((true_std**2 + (true_mu - est_mu)**2) / (2 * est_std**2)) - 0.5

if __name__ == '__main__':
    # -------------------------------
    # Setup network and simulator (normal, drop-free).
    # -------------------------------
    network = GroundTruthNetwork(paths="1")
    passive = PassiveSimulator(network)
    # No packet drops or congestion enabled.
    
    # -------------------------------
    # Configure ConQuest-based monitoring.
    # -------------------------------
    # Choose a snapshot window T (e.g., 1 ms).
    T = 0.000001  # seconds per snapshot
    start_time = time.time()
    monitor = ConQuestMonitor(passive, T, start_time)
    monitor.enable_monitoring()
    
    # -------------------------------
    # Simulate traffic.
    # -------------------------------
    simulation_duration = 100   # seconds
    avg_interarrival_ms = 20   # ms
    passive.simulate_traffic(duration_seconds=simulation_duration, avg_interarrival_ms=avg_interarrival_ms)
    
    # -------------------------------
    # Retrieve histograms from source and destination sketches.
    # -------------------------------
    source_hist = monitor.get_source_histogram()
    dest_hist = monitor.get_destination_histogram()
    
    
    # -------------------------------
    # Estimate delays from the snapshot histograms.
    # -------------------------------
    estimator = DelayDistributionEstimator()
    estimator.update_from_histograms(source_hist, dest_hist, T)
    delays = estimator.get_all_delays()  # delays in seconds
    if not delays:
        print("No delay data computed!")
        exit(0)
    delays_ms = [d * 1000 for d in delays]  # convert to milliseconds
    
    # -------------------------------
    # Fit a normal distribution to the estimated delays.
    # -------------------------------
    est_mu, est_std = norm.fit(delays_ms)
    print("Estimated Normal Fit (ms): μ=%.4f, σ=%.4f" % (est_mu, est_std))
    
    # -------------------------------
    # Retrieve the true underlying distribution from the network.
    # -------------------------------
    # Assume network.get_distribution_parameters()[0][2] returns a dict with "mean" and "std" (in ms).
    true_params = network.get_distribution_parameters()[0][2]
    true_mu = true_params["mean"]
    true_std = true_params["std"]
    print("True Underlying Distribution (ms): μ=%.4f, σ=%.4f" % (true_mu, true_std))
    
    # -------------------------------
    # Compute KL divergence.
    # -------------------------------
    kl_div = compute_kl_divergence(est_mu, est_std, true_mu, true_std)
    print("KL divergence: %.4f" % kl_div)
    
    # -------------------------------
    # Plot the estimated and true normal PDFs.
    # -------------------------------
    lower_bound = min(est_mu - 4*est_std, true_mu - 4*true_std)
    upper_bound = max(est_mu + 4*est_std, true_mu + 4*true_std)
    x = np.linspace(lower_bound, upper_bound, 200)
    
    est_pdf = norm.pdf(x, est_mu, est_std)
    true_pdf = norm.pdf(x, true_mu, true_std)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x, est_pdf, 'r-', linewidth=2,
             label=f"Estimated Normal\nμ={est_mu:.2f} ms, σ={est_std:.2f} ms")
    plt.plot(x, true_pdf, 'b--', linewidth=2,
             label=f"True Normal\nμ={true_mu:.2f} ms, σ={true_std:.2f} ms")
    plt.xlabel("Delay (ms)")
    plt.ylabel("Probability Density")
    plt.title("Estimated vs. True Delay Distributions (ConQuest)")
    plt.legend()
    plt.show()
