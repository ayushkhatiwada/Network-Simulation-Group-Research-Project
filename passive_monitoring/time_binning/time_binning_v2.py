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
    
#     passive.enable_congestion_simulation(network.DESTINATION)
    
#     # time bin monitoring configuration
#     bin_size = 0.001 
#     start_time = time.time()
#     tb_monitor = TimeBinMonitor(passive, bin_size, start_time)
#     tb_monitor.enable_monitoring()
    
#     # simulating traffic
#     simulation_duration = 100
#     avg_interarrival_ms = 20  
#     passive.simulate_traffic(duration_seconds=simulation_duration, avg_interarrival_ms=avg_interarrival_ms)
    
#     # get histograms
#     source_hist = tb_monitor.get_source_histogram()
#     dest_hist = tb_monitor.get_destination_histogram()
#     source_sliding_hist = tb_monitor.get_source_sliding_histogram()
#     dest_sliding_hist = tb_monitor.get_destination_sliding_histogram()
    
#     # congestion fallback with smoothing
#     window_size = 10     
#     alpha = 1             
#     cost_function = 'exponential'
#     smooth_kernel = 3    
    
#     estimator = DelayDistributionEstimator()
#     estimator.update_from_histograms(source_hist, dest_hist, bin_size, window_size, alpha, cost_function, use_fallback=True, smooth_kernel=smooth_kernel)
#     estimator.update_from_histograms(source_sliding_hist, dest_sliding_hist, bin_size, window_size, alpha, cost_function, use_fallback=True, smooth_kernel=smooth_kernel)
    
#     delays = estimator.get_all_delays() 
#     delays = [d for d in delays if d is not None and not np.isnan(d)]
#     delays_ms = [d * 1000 for d in delays] 
    
#     # plotting the delay distribution
#     plt.figure(figsize=(8, 5))
#     plt.hist(delays_ms, bins=20, density=True, alpha=0.6, edgecolor='black')
#     plt.xlabel("Delay (ms)")
#     plt.ylabel("Probability Density")
#     plt.title("Delay Distribution with Congestion (Extended Delays & Drops)")
    
#     if delays_ms:
#         mu_ms, std_ms = norm.fit(delays_ms)
#         print("Normal fit parameters (ms):", mu_ms, std_ms)
#         if std_ms < 1e-6:
#             print("Standard deviation too small; skipping KL divergence computation.")
#         else:
#             kl_div = passive.compare_distribution_parameters(mu_ms, std_ms)
#             print("KL divergence:", kl_div)
#         xmin, xmax = plt.xlim()
#         x = np.linspace(xmin, xmax, 100)
#         pdf = norm.pdf(x, mu_ms, std_ms)
#         plt.plot(x, pdf, 'r', linewidth=2,
#                  label=f'Normal Fit\nμ={mu_ms:.2f} ms, σ={std_ms:.2f} ms')
#         plt.legend()
#     plt.show()
    
#     # alternative approach
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


'''
BREAK
'''

# import time
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.stats import norm

# from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
# from active_monitoring_evolution.ground_truth import GroundTruthNetwork
# from passive_monitoring.time_binning.time_bin_monitoring import TimeBinMonitor
# from passive_monitoring.time_binning.delay_estimator import DelayDistributionEstimator

# if __name__ == '__main__':
#     # setup network simulator
#     network = GroundTruthNetwork(paths="1")
#     passive = PassiveSimulator(network)
    
#     # Enable fixed congestion simulation on the destination switch.
#     # This will use the fixed congestion parameters (e.g., 40% drop during congestion,
#     # and an extra delay factor of 1.5) 
#     passive.enable_congestion_simulation(network.DESTINATION)
    
#     # configuring time bin monitoring
#     bin_size = 0.001  
#     start_time = time.time()
#     tb_monitor = TimeBinMonitor(passive, bin_size, start_time)
#     tb_monitor.enable_monitoring()
    

#     # simulating traffic
#     simulation_duration = 100   
#     avg_interarrival_ms = 20  
#     passive.simulate_traffic(duration_seconds=simulation_duration, avg_interarrival_ms=avg_interarrival_ms)
    
#     # get histograms
#     source_hist = tb_monitor.get_source_histogram()
#     dest_hist = tb_monitor.get_destination_histogram()
#     source_sliding_hist = tb_monitor.get_source_sliding_histogram()
#     dest_sliding_hist = tb_monitor.get_destination_sliding_histogram()
    
#     # estimate delays
#     # In congestion scenarios, we enable fallback (with histogram smoothing)
#     # to handle extended delays and missing events.
#     window_size = 10     
#     alpha = 1             # weighting parameter
#     cost_function = 'exponential'
#     smooth_kernel = 3     # kernel size for smoothing
    
#     estimator = DelayDistributionEstimator()
#     estimator.update_from_histograms(source_hist, dest_hist, bin_size, window_size, alpha, cost_function,
#                                        use_fallback=True, smooth_kernel=smooth_kernel)
#     estimator.update_from_histograms(source_sliding_hist, dest_sliding_hist, bin_size, window_size, alpha, cost_function,
#                                        use_fallback=True, smooth_kernel=smooth_kernel)
    
#     delays = estimator.get_all_delays()  # delays in seconds
#     # Filter out any invalid values.
#     delays = [d for d in delays if d is not None and not np.isnan(d)]
#     delays_ms = [d * 1000 for d in delays]  # convert to milliseconds
    
#     if not delays_ms:
#         print("No delay data was computed.")
#         exit()
    
    
#     # Fit an ND for estimated delays
#     est_mu, est_std = norm.fit(delays_ms)
#     print("Estimated normal fit parameters (ms):", est_mu, est_std)
#     if est_std < 1e-6:
#         print("Standard deviation too small; skipping KL divergence computation.")
#     else:
#         kl_div = passive.compare_distribution_parameters(est_mu, est_std)
#         print("KL divergence:", kl_div)
    
    
#     # true underlying distribution
#     true_params = network.get_distribution_parameters()[0][2]
#     true_mu = true_params["mean"]
#     true_std = true_params["std"]
#     print("True underlying distribution (ms):", true_mu, true_std)
    
#     # get pdfs
#     lower_bound = min(est_mu - 4*est_std, true_mu - 4*true_std)
#     upper_bound = max(est_mu + 4*est_std, true_mu + 4*true_std)
#     x = np.linspace(lower_bound, upper_bound, 200)
    
#     est_pdf = norm.pdf(x, est_mu, est_std)
#     true_pdf = norm.pdf(x, true_mu, true_std)
    
#     # plot graphs on the same plane
#     plt.figure(figsize=(8, 5))
#     plt.plot(x, est_pdf, 'r-', linewidth=2,
#              label=f'Estimated Normal\nμ={est_mu:.2f} ms, σ={est_std:.2f} ms')
#     plt.plot(x, true_pdf, 'b--', linewidth=2,
#              label=f'True Normal\nμ={true_mu:.2f} ms, σ={true_std:.2f} ms')
#     plt.xlabel("Delay (ms)")
#     plt.ylabel("Probability Density")
#     plt.title("Estimated vs. True Delay Distributions (Congestion)")
#     plt.legend()
#     plt.show()
    
#     # alternative global offset
#     all_bins = set(source_hist.keys()) | set(dest_hist.keys())
#     if all_bins:
#         max_bin = max(all_bins)
#         source_arr = np.zeros(max_bin+1)
#         dest_arr = np.zeros(max_bin+1)
#         for b, count in source_hist.items():
#             source_arr[b] = count
#         for b, count in dest_hist.items():
#             dest_arr[b] = count
#         corr = np.correlate(dest_arr, source_arr, mode='full')
#         lags = np.arange(-len(source_arr)+1, len(source_arr))
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
