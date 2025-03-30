# import time
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.stats import norm

# from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
# from active_monitoring_evolution.ground_truth import GroundTruthNetwork
# from passive_monitoring.time_binning.time_bin_monitoring import TimeBinMonitor
# from passive_monitoring.time_binning.delay_estimator import DelayDistributionEstimator

# if __name__ == '__main__':
#     network = GroundTruthNetwork(paths="1")
#     passive = PassiveSimulator(network)
    
#     # Enable variable congestion simulation on the destination switch.
#     # This will generate congestion intervals with variable intensities.
#     passive.enable_variable_congestion_simulation(network.DESTINATION, max_intensity=5.0, seed=42)
    

#     # configuring time bins
#     # With variable congestion and extended delays, use a coarser bin size.
#     bin_size = 0.001  
#     start_time = time.time()
#     tb_monitor = TimeBinMonitor(passive, bin_size, start_time)
#     tb_monitor.enable_monitoring()
    
#     # simulate traffic
#     simulation_duration = 100  
#     avg_interarrival_ms = 20   
#     passive.simulate_traffic(duration_seconds=simulation_duration, avg_interarrival_ms=avg_interarrival_ms)
    
#     # get histograms
#     source_hist = tb_monitor.get_source_histogram()
#     dest_hist = tb_monitor.get_destination_histogram()
#     source_sliding_hist = tb_monitor.get_source_sliding_histogram()
#     dest_sliding_hist = tb_monitor.get_destination_sliding_histogram()
    
    
#     # estimate delays with fallback
#     window_size = 10      # in bin units
#     alpha = 1             # weighting parameter
#     cost_function = 'exponential'
#     smooth_kernel = 3     # kernel size for histogram smoothing
    
#     estimator = DelayDistributionEstimator()
#     estimator.update_from_histograms(source_hist, dest_hist, bin_size, window_size, alpha, cost_function,
#                                        use_fallback=True, smooth_kernel=smooth_kernel)
#     estimator.update_from_histograms(source_sliding_hist, dest_sliding_hist, bin_size, window_size, alpha, cost_function,
#                                        use_fallback=True, smooth_kernel=smooth_kernel)
    
#     delays = estimator.get_all_delays()  # delays in seconds
#     delays = [d for d in delays if d is not None and not np.isnan(d)]
#     delays_ms = [d * 1000 for d in delays]  # convert to milliseconds
    
#     if not delays_ms:
#         print("No delay data was computed.")
#         exit()
    
#     # fit a normal distribution
#     est_mu, est_std = norm.fit(delays_ms)
#     print("Estimated normal fit parameters (ms):", est_mu, est_std)
#     if est_std < 1e-6:
#         print("Standard deviation too small; skipping KL divergence computation.")
#     else:
#         kl_div = passive.compare_distribution_parameters(est_mu, est_std)
#         print("KL divergence:", kl_div)
    
#     # get underlying distribution
#     true_params = network.get_distribution_parameters()[0][2]  
#     true_mu = true_params["mean"]
#     true_std = true_params["std"]
#     print("True underlying distribution (ms):", true_mu, true_std)
    
#     # get pdfs
#     lower_bound = min(est_mu - 4 * est_std, true_mu - 4 * true_std)
#     upper_bound = max(est_mu + 4 * est_std, true_mu + 4 * true_std)
#     x = np.linspace(lower_bound, upper_bound, 200)
    
#     est_pdf = norm.pdf(x, est_mu, est_std)
#     true_pdf = norm.pdf(x, true_mu, true_std)
    
#     # plot graphs on the same plane
#     plt.figure(figsize=(8, 5))
#     plt.plot(x, est_pdf, 'r-', linewidth=2,
#              label=f'Estimated Normal\nμ={est_mu:.2f} ms, σ={est_std:.2f} ms')
#     plt.plot(x, true_pdf, 'b--', linewidth=2,
#              label=f'True Normal\nμ={true_mu:.2f} ms, σ={true_std:.2f} ms')
#     plt.xlabel("Delay (ms)")
#     plt.ylabel("Probability Density")
#     plt.title("Estimated vs. True Delay Distributions (Variable Congestion)")
#     plt.legend()
#     plt.show()
    
#     # altenative: global offset
#     all_bins = set(source_hist.keys()) | set(dest_hist.keys())
#     if all_bins:
#         max_bin = max(all_bins)
#         source_arr = np.zeros(max_bin + 1)
#         dest_arr = np.zeros(max_bin + 1)
#         for b, count in source_hist.items():
#             source_arr[b] = count
#         for b, count in dest_hist.items():
#             dest_arr[b] = count
#         corr = np.correlate(dest_arr, source_arr, mode='full')
#         lags = np.arange(-len(source_arr) + 1, len(source_arr))
#         lag_times = lags * bin_size
#         max_corr_idx = np.argmax(corr)
#         estimated_shift_bins = lags[max_corr_idx]
#         estimated_delay = estimated_shift_bins * bin_size
#         print("Estimated average delay via cross-correlation (s):", estimated_delay)
        
#         plt.figure(figsize=(8, 5))
#         plt.stem(lag_times, corr, use_line_collection=True)
#         plt.xlabel("Delay (s)")
#         plt.ylabel("Cross-correlation")
#         plt.title("Cross-Correlation of Histograms")
#         plt.show()


import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
from active_monitoring_evolution.ground_truth import GroundTruthNetwork
from passive_monitoring.time_binning.time_bin_monitoring import TimeBinMonitor
from passive_monitoring.time_binning.delay_estimator import DelayDistributionEstimator

if __name__ == '__main__':
    # Initialize ground truth network and passive simulator.
    network = GroundTruthNetwork(paths="1")
    passive = PassiveSimulator(network)
    
    # --- Incorporate packet drops via congestion simulation ---
    # Enable variable congestion simulation on the destination switch.
    # This simulates packet drops and delays with variable intensity.
    passive.enable_variable_congestion_simulation(network.DESTINATION, max_intensity=5.0, seed=42)
    
    # --- Set up time-bin monitoring ---
    bin_size = 0.0001 
    start_time = time.time()
    tb_monitor = TimeBinMonitor(passive, bin_size, start_time)
    tb_monitor.enable_monitoring()
    
    # --- Simulate traffic ---
    simulation_duration = 100         # seconds
    avg_interarrival_ms = 20          # average interarrival time (ms)
    passive.simulate_traffic(duration_seconds=simulation_duration, avg_interarrival_ms=avg_interarrival_ms)
    
    # --- Retrieve histograms ---
    source_hist = tb_monitor.get_source_histogram()
    dest_hist = tb_monitor.get_destination_histogram()
    source_sliding_hist = tb_monitor.get_source_sliding_histogram()
    dest_sliding_hist = tb_monitor.get_destination_sliding_histogram()
    
    # --- Estimate delays ---
    # Here we use the fallback strategy in the estimator so that when a matching candidate is missing
    # (possibly due to packet drops) a fallback delay (via cross-correlation) is used.
    window_size = 9
    alpha = 2
    cost_function = 'exponential'
    
    estimator = DelayDistributionEstimator()
    estimator.update_from_histograms(source_hist, dest_hist, bin_size, window_size, alpha, cost_function, use_fallback=True)
    estimator.update_from_histograms(source_sliding_hist, dest_sliding_hist, bin_size, window_size, alpha, cost_function, use_fallback=True)
    
    delays = estimator.get_all_delays()
    
    if not delays:
        print("No delay data was computed.")
        exit()
    
    # --- Post-processing filtration ---
    # In the presence of packet drops, some delay estimates may be extreme.
    # Here we filter out delays that are more than 3 standard deviations from the mean.
    delays_array = np.array(delays)
    mean_delay = np.mean(delays_array)
    std_delay = np.std(delays_array)
    filtered_delays = delays_array[(delays_array >= mean_delay - 3 * std_delay) & (delays_array <= mean_delay + 3 * std_delay)]
    delays = filtered_delays.tolist()
    
    delays_ms = [d * 1000 for d in delays]

    # --- Fit a normal distribution to the estimated delays ---
    est_mu, est_std = norm.fit(delays_ms)
    print("Estimated normal fit parameters (ms):", est_mu, est_std)
    kl_div = passive.compare_distribution_parameters(est_mu, est_std)
    print("KL divergence:", kl_div)
    
    # --- Get the true underlying distribution ---
    true_params = network.get_distribution_parameters()[0][2]
    true_mu = true_params["mean"]
    true_std = true_params["std"]
    print("True underlying distribution parameters (ms):", true_mu, true_std)
    
    # --- Prepare values for plotting ---
    lower_bound = min(est_mu - 4 * est_std, true_mu - 4 * true_std)
    upper_bound = max(est_mu + 4 * est_std, true_mu + 4 * true_std)
    x = np.linspace(lower_bound, upper_bound, 200)
    
    est_pdf = norm.pdf(x, est_mu, est_std)
    true_pdf = norm.pdf(x, true_mu, true_std)
    
    # --- Plot the estimated and true PDFs ---
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
