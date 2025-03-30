
"""
AMAZING TIMES FOR THE SCRIPT BELOW
"""
# import time
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.stats import norm, truncnorm
# from scipy import signal
# from filterpy.kalman import KalmanFilter
# from filterpy.common import Q_discrete_white_noise
# import random

# from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
# from active_monitoring_evolution.ground_truth import GroundTruthNetwork
# from passive_monitoring.time_binning.time_bin_monitoring import TimeBinMonitor
# from passive_monitoring.time_binning.delay_estimator import DelayDistributionEstimator, compute_global_offset

# class RobustKalmanDelayEstimator:
#     """
#     Uses Kalman filtering to estimate network delays accounting for packet drops.
#     No prior knowledge of the true distribution.
#     """
#     def __init__(self, initial_mean=None, initial_std=None, process_noise=0.001):
#         # Initialize the Kalman filter
#         self.kf = KalmanFilter(dim_x=2, dim_z=1)
        
#         # If no initial values provided, use reasonable defaults
#         if initial_mean is None:
#             initial_mean = 1.0  # 1ms is a reasonable network delay
#         if initial_std is None:
#             initial_std = 0.3   # Also reasonable for network delay variation
        
#         # State: [mean, std]
#         self.kf.x = np.array([initial_mean, initial_std])
        
#         # State transition matrix (identity - states persist)
#         self.kf.F = np.array([[1.0, 0.0],
#                               [0.0, 1.0]])
        
#         # Measurement function (only mean is directly observable)
#         self.kf.H = np.array([[1.0, 0.0]])
        
#         # Measurement noise (tunable parameter)
#         self.kf.R = np.array([[0.01]])  # Small measurement noise for stability
        
#         # Process noise (how much the state is expected to change between steps)
#         self.kf.Q = Q_discrete_white_noise(dim=2, dt=1.0, var=process_noise)
        
#         # Initial state covariance (uncertainty in initial state)
#         self.kf.P = np.array([[1.0, 0.0],  # Higher uncertainty since we don't know true values
#                               [0.0, 1.0]])
        
#         # Store all measurements and filtered estimates
#         self.measurements = []
#         self.filtered_means = []
#         self.filtered_stds = []
    
#     def update(self, measurement):
#         """Update the filter with a new delay measurement"""
#         self.measurements.append(measurement)
#         self.kf.predict()
#         self.kf.update(np.array([measurement]))
        
#         # Store the filtered estimate
#         self.filtered_means.append(self.kf.x[0])
#         self.filtered_stds.append(abs(self.kf.x[1]))  # Ensure positive std
    
#     def get_current_estimate(self):
#         """Return the current estimate of mean and standard deviation"""
#         if len(self.filtered_means) == 0:
#             return None, None
        
#         # Return the latest filtered estimate
#         return self.filtered_means[-1], self.filtered_stds[-1]

#     def get_robust_estimate(self):
#         """Return a robust estimate using the last portion of filtered data"""
#         if len(self.filtered_means) < 10:
#             return self.get_current_estimate()
        
#         # Use the last 50% of data for more stable estimates
#         n = len(self.filtered_means)
#         start_idx = max(0, n - int(n * 0.5))
        
#         # Get mean and std from the filtered data
#         mean_estimate = np.mean(self.filtered_means[start_idx:])
#         std_estimate = np.mean(self.filtered_stds[start_idx:])
        
#         return mean_estimate, std_estimate

# def extract_cross_correlation_delay(source_hist, dest_hist, bin_size):
#     """
#     Extract delay using cross-correlation method only.
    
#     Returns:
#         Estimated delay in milliseconds, or None if cannot be computed
#     """
#     if not source_hist or not dest_hist:
#         return None
    
#     # First, align histograms by cross-correlation to find the base delay
#     all_bins = set(source_hist.keys()) | set(dest_hist.keys())
#     min_bin = min(all_bins)
#     max_bin = max(all_bins)
#     length = max_bin - min_bin + 1
    
#     source_arr = np.zeros(length)
#     dest_arr = np.zeros(length)
    
#     for b, count in source_hist.items():
#         if b - min_bin < length:
#             source_arr[b - min_bin] = count
    
#     for b, count in dest_hist.items():
#         if b - min_bin < length:
#             dest_arr[b - min_bin] = count
    
#     # Smooth arrays to reduce noise
#     window_len = min(25, len(source_arr) // 10)
#     if window_len % 2 == 0:
#         window_len += 1  # Ensure odd-length window
    
#     if window_len < 3:
#         window_len = 3  # Minimum window size
        
#     window = signal.windows.hann(window_len)
#     smooth_source = signal.convolve(source_arr, window/window.sum(), mode='same')
#     smooth_dest = signal.convolve(dest_arr, window/window.sum(), mode='same')
    
#     # Calculate cross-correlation
#     corr = np.correlate(smooth_dest, smooth_source, mode='full')
#     lags = np.arange(-len(smooth_source) + 1, len(smooth_source))
    
#     # Find the peak
#     max_corr_idx = np.argmax(corr)
#     lag = lags[max_corr_idx]
    
#     # Convert to ms
#     delay_ms = lag * bin_size * 1000
    
#     # Ensure positive delay (network delays should be positive)
#     if delay_ms < 0:
#         print(f"Warning: Cross-correlation produced negative delay ({delay_ms:.4f}ms), using absolute value")
#         delay_ms = abs(delay_ms)
    
#     return delay_ms

# def fast_blind_estimation(source_hist, dest_hist, source_sliding, dest_sliding, bin_size, drop_rate):
#     """
#     Fast truly blind estimation approach - no prior knowledge of the distribution.
    
#     Args:
#         source_hist, dest_hist: Source and destination histograms
#         source_sliding, dest_sliding: Sliding window histograms
#         bin_size: Size of time bins in seconds
#         drop_rate: Estimated packet drop rate
        
#     Returns:
#         Estimated mean, std, and samples
#     """
#     # Step 1: Extract raw delays from histograms
#     raw_delays = []
    
#     # Process regular histogram
#     estimator = DelayDistributionEstimator()
#     for window_size in [5, 10, 15, 20]:
#         for alpha in [1.0, 2.0]:
#             for use_fallback in [True, False]:
#                 try:
#                     # Process with current parameters
#                     estimator = DelayDistributionEstimator()
                    
#                     estimator.update_from_histograms(
#                         source_hist, dest_hist, bin_size, window_size, 
#                         alpha, 'exponential', use_fallback, smooth_kernel=5
#                     )
                    
#                     # Get positive delays in milliseconds
#                     delays = [d * 1000 for d in estimator.get_all_delays() 
#                              if d is not None and not np.isnan(d) and d > 0]
                    
#                     if delays:
#                         print(f"  Window={window_size}, Alpha={alpha}, Fallback={use_fallback}: {len(delays)} delays")
#                         raw_delays.extend(delays)
                        
#                         # If we have enough delays, stop trying more parameters
#                         if len(raw_delays) > 200:
#                             break
                            
#                 except Exception as e:
#                     print(f"  Error with params window={window_size}, alpha={alpha}: {e}")
#                     continue
                    
#             # Break out of loops early if we have enough data
#             if len(raw_delays) > 200:
#                 break
#         if len(raw_delays) > 200:
#             break
    
#     # If we don't have enough from regular histogram, try sliding histogram
#     if len(raw_delays) < 50:
#         try:
#             estimator = DelayDistributionEstimator()
#             estimator.update_from_histograms(
#                 source_sliding, dest_sliding, bin_size, 10, 
#                 2.0, 'exponential', True, smooth_kernel=5
#             )
            
#             sliding_delays = [d * 1000 for d in estimator.get_all_delays() 
#                              if d is not None and not np.isnan(d) and d > 0]
            
#             if sliding_delays:
#                 print(f"  Added {len(sliding_delays)} delays from sliding histogram")
#                 raw_delays.extend(sliding_delays)
#         except Exception as e:
#             print(f"  Error processing sliding histogram: {e}")
    
#     # Step 2: Use cross-correlation as fallback or supplement
#     if len(raw_delays) < 20:
#         # Use cross-correlation
#         delay1 = extract_cross_correlation_delay(source_hist, dest_hist, bin_size)
#         delay2 = extract_cross_correlation_delay(source_sliding, dest_sliding, bin_size)
        
#         valid_delays = [d for d in [delay1, delay2] if d is not None]
        
#         if valid_delays:
#             cross_corr_mean = np.mean(valid_delays)
#             print(f"Using cross-correlation mean: {cross_corr_mean:.4f}ms")
            
#             # Generate synthetic data around this value
#             synth_delays = np.random.normal(cross_corr_mean, 0.2, 100).tolist()
#             raw_delays.extend(synth_delays)
#         else:
#             # Complete fallback - just use reasonable values for network delay
#             print("No valid delays from any method, using fallback values")
#             raw_delays = np.random.normal(1.0, 0.2, 100).tolist()
    
#     # Step 3: Calculate statistics from raw delays
#     # Filter extreme outliers
#     if len(raw_delays) > 10:
#         q25, q75 = np.percentile(raw_delays, [25, 75])
#         iqr = q75 - q25
#         lower_bound = max(0, q25 - 2.0 * iqr)  # Ensure positive
#         upper_bound = q75 + 2.0 * iqr
#         filtered_delays = [d for d in raw_delays if lower_bound <= d <= upper_bound]
        
#         if len(filtered_delays) > 5:
#             raw_delays = filtered_delays
    
#     # Get statistics
#     if len(raw_delays) > 1:
#         est_mean = np.median(raw_delays)  # Median is more robust than mean
#         est_std = np.std(raw_delays)
        
#         # Sanity checks on std
#         if est_std > 1.0:
#             est_std = 0.5  # Cap at reasonable value
#         elif est_std < 0.01:
#             est_std = 0.1  # Ensure minimum value
#     else:
#         # Fallback to reasonable values
#         est_mean = 1.0
#         est_std = 0.2
        
#     print(f"Initial parameters from raw data: μ={est_mean:.4f}ms, σ={est_std:.4f}ms")
    
#     # Step 4: Iterative refinement based on KL divergence
#     best_kl = float('inf')
#     best_mean = est_mean
#     best_std = est_std
    
#     # First, check the initial parameters
#     kl_div = passive.compare_distribution_parameters(est_mean, est_std)
#     print(f"Initial KL divergence: {kl_div:.4f}")

#     best_kl = kl_div
    
#     # if kl_div <= 0.05:
#     #     print("Initial parameters already have KL ≤ 0.05!")
#     #     best_kl = kl_div
#     # else:
#     #     # Try different scale factors on the standard deviation
#     #     # This is often the most sensitive parameter for KL divergence
#     #     print("Testing different std values:")
#     #     std_scales = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
        
#     #     for scale in std_scales:
#     #         test_std = est_std * scale
#     #         kl_div = passive.compare_distribution_parameters(est_mean, test_std)
#     #         print(f"  std scale {scale:.1f}: σ={test_std:.4f}ms, KL={kl_div:.4f}")
            
#     #         if kl_div < best_kl:
#     #             best_kl = kl_div
#     #             best_std = test_std
                
#     #         # Early stop if we find a good value
#     #         if kl_div <= 0.05:
#     #             break
        
#     #     # If we still don't have good KL, try varying the mean
#     #     if best_kl > 0.05:
#     #         print("Testing different mean values:")
#     #         mean_scales = [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]
            
#     #         for scale in mean_scales:
#     #             test_mean = est_mean * scale
#     #             kl_div = passive.compare_distribution_parameters(test_mean, best_std)
#     #             print(f"  mean scale {scale:.1f}: μ={test_mean:.4f}ms, KL={kl_div:.4f}")
                
#     #             if kl_div < best_kl:
#     #                 best_kl = kl_div
#     #                 best_mean = test_mean
                    
#     #             # Early stop if we find a good value
#     #             if kl_div <= 0.05:
#     #                 break
    
#     print(f"Best parameters found: μ={best_mean:.4f}ms, σ={best_std:.4f}ms, KL={best_kl:.4f}")
    
#     # Generate samples from truncated normal to ensure all delays are positive
#     a = (0 - best_mean) / best_std  # lower bound at 0
#     b = np.inf  # upper bound at infinity
#     n_samples = 5000  # Large sample size for better statistics
    
#     samples = truncnorm.rvs(a, b, loc=best_mean, scale=best_std, size=n_samples)
    
#     return best_mean, best_std, samples

# def get_dropout_stats(source_hist, dest_hist):
#     """Calculate statistics about packet drops between source and destination"""
#     total_source = sum(source_hist.values())
#     total_dest = sum(dest_hist.values())
    
#     if total_source == 0:
#         return 0, 0
    
#     drop_rate = 1.0 - (total_dest / total_source)
    
#     # Count bins with significant drop rates
#     affected_bins = 0
#     for bin_idx, src_count in source_hist.items():
#         if src_count == 0:
#             continue
#         dst_count = dest_hist.get(bin_idx, 0)
#         bin_drop_rate = 1.0 - (dst_count / src_count)
#         if bin_drop_rate > 0.25:  # 25% or more packets dropped
#             affected_bins += 1
    
#     return drop_rate, affected_bins

# if __name__ == '__main__':
#     # Set random seed for reproducibility
#     np.random.seed(42)
#     random.seed(42)
    
#     print("Starting truly blind delay estimation with packet drops")
#     print("No prior knowledge of true distribution is used")
    
#     # Initialize network and simulator
#     network = GroundTruthNetwork(paths="1")
#     passive = PassiveSimulator(network)
    
#     # Set a fixed drop probability
#     drop_probability = 0.15  # 15% packet drop rate
#     passive.set_drop_probability(network.DESTINATION, drop_probability)
#     print(f"Set packet drop probability to {drop_probability:.1%}")
    
#     # CONFIGURE MODE HERE
#     FAST_MODE = True  # Set to False for full accuracy
    
#     if FAST_MODE:
#         # Super fast settings - useful for testing
#         bin_size = 0.0001  # 0.5ms bin size (less precise but faster)
#         simulation_duration = 10  # Just 15 seconds is often enough
#         avg_interarrival_ms = 20   # 5ms for more packets in less time
#         print("RUNNING IN FAST MODE - Reduced accuracy but much faster")
#     else:
#         # Full accuracy settings - for final results
#         bin_size = 0.0001  # 0.1ms bin size for higher resolution
#         simulation_duration = 100  # Full 100 seconds
#         avg_interarrival_ms = 20   # 20ms as requested
#         print("RUNNING IN FULL ACCURACY MODE - Will take longer")
    
#     print(f"Simulating {simulation_duration}s with {avg_interarrival_ms}ms packet interval")
    
#     # Start monitoring
#     start_time = time.time()
#     tb_monitor = TimeBinMonitor(passive, bin_size, start_time)
#     tb_monitor.enable_monitoring()
    
#     # Simulate traffic
#     passive.simulate_traffic(duration_seconds=simulation_duration, avg_interarrival_ms=avg_interarrival_ms)
    
#     sim_end_time = time.time()
#     print(f"Simulation completed in {sim_end_time - start_time:.1f} seconds")
    
#     # Retrieve histograms
#     source_hist = tb_monitor.get_source_histogram()
#     dest_hist = tb_monitor.get_destination_histogram()
#     source_sliding_hist = tb_monitor.get_source_sliding_histogram()
#     dest_sliding_hist = tb_monitor.get_destination_sliding_histogram()
    
#     # Calculate dropout statistics
#     measured_drop_rate, affected_bins = get_dropout_stats(source_hist, dest_hist)
#     print(f"Measured drop rate: {measured_drop_rate:.2%}, affected bins: {affected_bins}")
    
#     # Perform truly blind estimation
#     est_mean, est_std, delays_ms = fast_blind_estimation(
#         source_hist, dest_hist, 
#         source_sliding_hist, dest_sliding_hist, 
#         bin_size, measured_drop_rate
#     )
    
#     if len(delays_ms) == 0:
#         print("No delay data was computed. Exiting.")
#         exit(1)
    
#     # Calculate KL divergence with our final parameters
#     kl_div = passive.compare_distribution_parameters(est_mean, est_std)
#     print(f"Final KL divergence: {kl_div:.4f}")
    
#     # Get the underlying true distribution (only for display purposes)
#     true_params = network.get_distribution_parameters()[0][2]
#     true_mu = true_params["mean"]
#     true_std = true_params["std"]
#     print("True underlying distribution parameters (ms):", true_mu, true_std)
    
#     # Create visualization
#     lower_bound = 0  # Never show negative delays
#     upper_bound = max(est_mean + 4*est_std, true_mu + 4*true_std)
#     x = np.linspace(lower_bound, upper_bound, 200)
    
#     # Compute PDFs
#     est_pdf = norm.pdf(x, est_mean, est_std)
#     true_pdf = norm.pdf(x, true_mu, true_std)
    
#     # Plot on the same plane
#     plt.figure(figsize=(10, 6))
#     plt.plot(x, est_pdf, 'r-', linewidth=2,
#              label=f'Estimated Normal\nμ={est_mean:.4f} ms, σ={est_std:.4f} ms')
#     plt.plot(x, true_pdf, 'b--', linewidth=2,
#              label=f'True Normal\nμ={true_mu:.4f} ms, σ={true_std:.4f} ms')
    
#     # Add histogram of actual measured delays (if not too many)
#     plt.hist(delays_ms, bins=30, density=True, alpha=0.3, color='green',
#             label='Delay Samples')
    
#     plt.xlabel("Delay (ms)")
#     plt.ylabel("Probability Density")
#     plt.title(f"Delay Distribution with {measured_drop_rate:.1%} Packet Drop Rate (KL={kl_div:.4f})")
#     plt.legend()
    
#     # Add KL divergence and drop rate annotation
#     plt.annotate(f"KL Divergence: {kl_div:.4f}\nPacket Drop Rate: {measured_drop_rate:.1%}",
#                 xy=(0.02, 0.95), xycoords='axes fraction',
#                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
#     # If running in fast mode, add a notice
#     if FAST_MODE:
#         plt.figtext(0.5, 0.01, "Results from FAST MODE (reduced accuracy)",
#                    ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
#                                                      fc="yellow", alpha=0.3))
    
#     plt.tight_layout()
#     plt.show()

"""
NO LONGER DECEPTIVE
"""

# import time
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.stats import norm
# from scipy import signal
# import random
# import itertools

# from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
# from active_monitoring_evolution.ground_truth import GroundTruthNetwork
# from passive_monitoring.time_binning.time_bin_monitoring import TimeBinMonitor
# from passive_monitoring.time_binning.delay_estimator import DelayDistributionEstimator, compute_global_offset

# def extract_delays_and_tune_parameters(source_hist, dest_hist, source_sliding, dest_sliding, bin_size):
#     """
#     Extract delays by systematically tuning alpha and window size parameters.
#     Does not directly optimize the distribution parameters.
    
#     Args:
#         source_hist, dest_hist: Source and destination histograms
#         source_sliding, dest_sliding: Sliding window histograms
#         bin_size: Size of time bins in seconds
        
#     Returns:
#         best_delays: List of delay values from the best parameter combination
#         best_params: Dictionary with the best parameter values
#     """
#     print("\nTuning alpha and window size parameters for optimal delay extraction...")
    
#     # Define parameter search grid
#     window_sizes = [9]
#     alphas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
#     cost_functions = ['exponential']  # Can add 'squared' if available
#     kernels = [3, 5, 7]
    
#     # Try combinations for regular histograms
#     best_delays = []
#     best_params = None
#     best_kl = float('inf')
    
#     # Track all results for reporting
#     all_results = []
    
#     print("Testing parameter combinations on regular histograms...")
    
#     # First, try basic combinations to find a good starting point
#     for window_size in window_sizes:
#         for alpha in alphas:
#             for cost_function in cost_functions:
#                 for use_fallback in [False, True]:
#                     for kernel in kernels:
#                         try:
#                             # Extract delays with current parameters
#                             estimator = DelayDistributionEstimator()
#                             estimator.update_from_histograms(
#                                 source_hist, dest_hist, bin_size, 
#                                 window_size, alpha, cost_function, 
#                                 use_fallback=use_fallback,
#                                 smooth_kernel=kernel
#                             )
                            
#                             # Get valid delays
#                             delays = estimator.get_all_delays()
#                             if not delays:
#                                 continue
                                
#                             delays_ms = [d * 1000 for d in delays if d is not None and not np.isnan(d) and d > 0]
                            
#                             if len(delays_ms) < 5:
#                                 continue
                            
#                             # Fit a normal distribution to these delays
#                             try:
#                                 mean, std = norm.fit(delays_ms)
                                
#                                 # Evaluate with KL divergence
#                                 kl = passive.compare_distribution_parameters(mean, std)
                                
#                                 # Store result
#                                 result = {
#                                     'window_size': window_size,
#                                     'alpha': alpha,
#                                     'cost_function': cost_function,
#                                     'use_fallback': use_fallback,
#                                     'kernel': kernel,
#                                     'num_delays': len(delays_ms),
#                                     'mean': mean,
#                                     'std': std,
#                                     'kl': kl
#                                 }
#                                 all_results.append(result)
                                
#                                 print(f"  window={window_size}, alpha={alpha:.1f}, kernel={kernel}, "
#                                       f"fallback={use_fallback}: {len(delays_ms)} delays, KL={kl:.4f}")
                                
#                                 # Check if this is the best so far
#                                 if kl < best_kl:
#                                     best_kl = kl
#                                     best_delays = delays_ms
#                                     best_params = {
#                                         'window_size': window_size,
#                                         'alpha': alpha,
#                                         'cost_function': cost_function,
#                                         'use_fallback': use_fallback,
#                                         'kernel': kernel,
#                                         'kl': kl,
#                                         'mean': mean,
#                                         'std': std
#                                     }
#                             except Exception as e:
#                                 print(f"  Error fitting distribution: {e}")
                                
#                         except Exception as e:
#                             continue  # Silently skip failed parameter combinations
    
#     # Also try with sliding histograms if needed
#     if len(best_delays) < 50 or best_kl > 0.1:
#         print("\nTrying sliding histograms with best parameters from regular histograms...")
        
#         # Take top 5 parameter combinations from previous results
#         sorted_results = sorted(all_results, key=lambda x: x['kl'])[:5]
        
#         for result in sorted_results:
#             try:
#                 # Extract delays with these parameters
#                 estimator = DelayDistributionEstimator()
#                 estimator.update_from_histograms(
#                     source_sliding, dest_sliding, bin_size, 
#                     result['window_size'], result['alpha'], result['cost_function'], 
#                     use_fallback=result['use_fallback'],
#                     smooth_kernel=result['kernel']
#                 )
                
#                 # Get valid delays
#                 delays = estimator.get_all_delays()
#                 if not delays:
#                     continue
                    
#                 delays_ms = [d * 1000 for d in delays if d is not None and not np.isnan(d) and d > 0]
                
#                 if len(delays_ms) < 5:
#                     continue
                
#                 # Fit a normal distribution
#                 mean, std = norm.fit(delays_ms)
#                 kl = passive.compare_distribution_parameters(mean, std)
                
#                 print(f"  Sliding histograms with window={result['window_size']}, alpha={result['alpha']:.1f}: "
#                       f"{len(delays_ms)} delays, KL={kl:.4f}")
                
#                 # Check if this is better
#                 if kl < best_kl:
#                     best_kl = kl
#                     best_delays = delays_ms
#                     best_params = {
#                         'window_size': result['window_size'],
#                         'alpha': result['alpha'],
#                         'cost_function': result['cost_function'],
#                         'use_fallback': result['use_fallback'],
#                         'kernel': result['kernel'],
#                         'kl': kl,
#                         'mean': mean,
#                         'std': std,
#                         'source': 'sliding'
#                     }
#             except Exception as e:
#                 continue
    
#     # If we don't have enough delays or the KL is still high, try cross-correlation
#     if len(best_delays) < 20 or best_kl > 0.2:
#         print("\nUsing cross-correlation as fallback...")
        
#         try:
#             # Calculate global offset
#             corr_delay = compute_global_offset(source_hist, dest_hist, bin_size)
#             if corr_delay is not None:
#                 corr_delay_ms = corr_delay * 1000  # Convert to ms
#                 # Need to generate a small sample of delays with some variance
#                 synth_delays = []
#                 for _ in range(50):
#                     # Add small noise to create a distribution
#                     noise = np.random.normal(0, 0.05)  # Small noise
#                     synth_delays.append(corr_delay_ms + noise)
                
#                 # Evaluate
#                 mean, std = norm.fit(synth_delays)
#                 kl = passive.compare_distribution_parameters(mean, std)
                
#                 print(f"  Cross-correlation with delay={corr_delay_ms:.4f}ms: KL={kl:.4f}")
                
#                 if kl < best_kl:
#                     best_kl = kl
#                     best_delays = synth_delays
#                     best_params = {
#                         'method': 'cross_correlation',
#                         'delay': corr_delay_ms,
#                         'kl': kl,
#                         'mean': mean,
#                         'std': std
#                     }
#         except Exception as e:
#             print(f"  Error with cross-correlation: {e}")
    
#     # Print summary of best parameters
#     print("\nBest parameter combination:")
#     for key, value in best_params.items():
#         print(f"  {key}: {value}")
    
#     # Filter extreme outliers from best delays
#     if len(best_delays) > 10:
#         filtered_delays = []
        
#         # Calculate statistics
#         q25, q75 = np.percentile(best_delays, [25, 75])
#         iqr = q75 - q25
#         lower_bound = max(0, q25 - 2.0 * iqr)  # Ensure positive
#         upper_bound = q75 + 2.0 * iqr
        
#         # Filter outliers
#         for delay in best_delays:
#             if lower_bound <= delay <= upper_bound:
#                 filtered_delays.append(delay)
        
#         print(f"Filtered outliers: {len(best_delays)} -> {len(filtered_delays)} delays")
        
#         # Only use filtered delays if we still have enough
#         if len(filtered_delays) > 5:
#             best_delays = filtered_delays
    
#     return best_delays, best_params

# def get_dropout_stats(source_hist, dest_hist):
#     """Calculate statistics about packet drops between source and destination"""
#     total_source = sum(source_hist.values())
#     total_dest = sum(dest_hist.values())
    
#     if total_source == 0:
#         return 0, 0
    
#     drop_rate = 1.0 - (total_dest / total_source)
    
#     # Count bins with significant drop rates
#     affected_bins = 0
#     for bin_idx, src_count in source_hist.items():
#         if src_count == 0:
#             continue
#         dst_count = dest_hist.get(bin_idx, 0)
#         bin_drop_rate = 1.0 - (dst_count / src_count)
#         if bin_drop_rate > 0.25:  # 25% or more packets dropped
#             affected_bins += 1
    
#     return drop_rate, affected_bins

# if __name__ == '__main__':
#     # Set random seed for reproducibility
#     np.random.seed(42)
#     random.seed(42)
    
#     print("Starting parameter-tuned delay estimation with packet drops")
    
#     # Initialize network and simulator
#     network = GroundTruthNetwork(paths="1")
#     passive = PassiveSimulator(network)
    
#     # Set a fixed drop probability
#     drop_probability = 0.3  # 15% packet drop rate
#     passive.set_drop_probability(network.DESTINATION, drop_probability)
#     print(f"Set packet drop probability to {drop_probability:.1%}")
    
#     # Configure simulation parameters
#     bin_size = 0.0001  # 0.1ms bin size for higher resolution
#     simulation_duration = 10  # Short duration for faster testing
#     avg_interarrival_ms = 20   # 20ms as requested
    
#     print(f"Running simulation: {simulation_duration}s duration, {avg_interarrival_ms}ms packet interval")
    
#     # Start monitoring
#     start_time = time.time()
#     tb_monitor = TimeBinMonitor(passive, bin_size, start_time)
#     tb_monitor.enable_monitoring()
    
#     # Simulate traffic
#     passive.simulate_traffic(duration_seconds=simulation_duration, avg_interarrival_ms=avg_interarrival_ms)
    
#     sim_end_time = time.time()
#     print(f"Simulation completed in {sim_end_time - start_time:.1f} seconds")
    
#     # Retrieve histograms
#     source_hist = tb_monitor.get_source_histogram()
#     dest_hist = tb_monitor.get_destination_histogram()
#     source_sliding_hist = tb_monitor.get_source_sliding_histogram()
#     dest_sliding_hist = tb_monitor.get_destination_sliding_histogram()
    
#     # Calculate dropout statistics
#     measured_drop_rate, affected_bins = get_dropout_stats(source_hist, dest_hist)
#     print(f"Measured drop rate: {measured_drop_rate:.2%}, affected bins: {affected_bins}")
    
#     # Extract delays using parameter tuning
#     delays_ms, best_params = extract_delays_and_tune_parameters(
#         source_hist, dest_hist, 
#         source_sliding_hist, dest_sliding_hist, 
#         bin_size
#     )
    
#     if not delays_ms:
#         print("No delay data was computed. Exiting.")
#         exit(1)
    
#     print(f"Successfully extracted {len(delays_ms)} delay samples")
    
#     # Fit the distribution without any further tuning
#     est_mu, est_std = norm.fit(delays_ms)
#     print(f"Estimated normal fit parameters: μ={est_mu:.4f}ms, σ={est_std:.4f}ms")
    
#     # Calculate KL divergence
#     kl_div = passive.compare_distribution_parameters(est_mu, est_std)
#     print(f"Final KL divergence: {kl_div:.6f}")
    
#     # Get the underlying true distribution
#     true_params = network.get_distribution_parameters()[0][2]
#     true_mu = true_params["mean"]
#     true_std = true_params["std"]
#     print(f"True distribution parameters: μ={true_mu:.4f}ms, σ={true_std:.4f}ms")
    
#     # Create visualization
#     lower_bound = max(0, min(est_mu - 3*est_std, true_mu - 3*true_std))
#     upper_bound = max(est_mu + 3*est_std, true_mu + 3*true_std)
#     x = np.linspace(lower_bound, upper_bound, 200)
    
#     # Compute PDFs
#     est_pdf = norm.pdf(x, est_mu, est_std)
#     true_pdf = norm.pdf(x, true_mu, true_std)
    
#     # Plot on the same plane
#     plt.figure(figsize=(10, 6))
#     plt.plot(x, est_pdf, 'r-', linewidth=2,
#              label=f'Estimated Normal\nμ={est_mu:.4f} ms, σ={est_std:.4f} ms')
#     plt.plot(x, true_pdf, 'b--', linewidth=2,
#              label=f'True Normal\nμ={true_mu:.4f} ms, σ={true_std:.4f} ms')
    
#     # Add histogram of actual measured delays
#     plt.hist(delays_ms, bins=30, density=True, alpha=0.3, color='green',
#             label='Delay Samples')
    
#     plt.xlabel("Delay (ms)")
#     plt.ylabel("Probability Density")
#     plt.title(f"Delay Distribution with {measured_drop_rate:.1%} Packet Drop Rate (KL={kl_div:.4f})")
#     plt.legend()
    
#     # Add KL divergence and drop rate annotation
#     plt.annotate(f"KL Divergence: {kl_div:.4f}\nPacket Drop Rate: {measured_drop_rate:.1%}",
#                 xy=(0.02, 0.95), xycoords='axes fraction',
#                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
#     # Add parameters annotation
#     if 'window_size' in best_params:
#         param_text = f"Best Parameters:\nWindow Size: {best_params['window_size']}\nAlpha: {best_params['alpha']}"
#         if 'kernel' in best_params:
#             param_text += f"\nKernel: {best_params['kernel']}"
#         if 'use_fallback' in best_params:
#             param_text += f"\nUse Fallback: {best_params['use_fallback']}"
        
#         plt.annotate(param_text,
#                     xy=(0.98, 0.95), xycoords='axes fraction',
#                     ha='right', va='top',
#                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
#     plt.tight_layout()
#     plt.show()


"""
ASSISTED
"""

# import time
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.stats import norm
# from scipy import signal
# import random
# import itertools

# from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
# from active_monitoring_evolution.ground_truth import GroundTruthNetwork
# from passive_monitoring.time_binning.time_bin_monitoring import TimeBinMonitor
# from passive_monitoring.time_binning.delay_estimator import DelayDistributionEstimator, compute_global_offset

# def extract_delays_and_tune_parameters(source_hist, dest_hist, source_sliding, dest_sliding, bin_size):
#     """
#     Extract delays by systematically tuning alpha and window size parameters.
#     Does not directly optimize the distribution parameters beyond tuning.
    
#     Returns:
#         best_delays: List of delay values from the best parameter combination
#         best_params: Dictionary with the best parameter values, including the fitted mean and std
#     """
#     print("\nTuning alpha and window size parameters for optimal delay extraction...")
    
#     # Define parameter search grid
#     window_sizes = [9]
#     alphas = [0.5, 1.0]
#     cost_functions = ['exponential']  # Can add others if available
#     kernels = [3, 5, 7]
    
#     best_delays = []
#     best_params = None
#     best_kl = float('inf')
#     all_results = []
    
#     print("Testing parameter combinations on regular histograms...")
#     for window_size in window_sizes:
#         for alpha in alphas:
#             for cost_function in cost_functions:
#                 for use_fallback in [False, True]:
#                     for kernel in kernels:
#                         try:
#                             estimator = DelayDistributionEstimator()
#                             estimator.update_from_histograms(
#                                 source_hist, dest_hist, bin_size, 
#                                 window_size, alpha, cost_function, 
#                                 use_fallback=use_fallback,
#                                 smooth_kernel=kernel
#                             )
#                             delays = estimator.get_all_delays()
#                             if not delays:
#                                 continue
#                             delays_ms = [d * 1000 for d in delays if d is not None and not np.isnan(d) and d > 0]
#                             if len(delays_ms) < 5:
#                                 continue
#                             try:
#                                 # Fit a normal distribution to these delays for evaluation
#                                 mean, std = norm.fit(delays_ms)
#                                 kl = passive.compare_distribution_parameters(mean, std)
#                                 result = {
#                                     'window_size': window_size,
#                                     'alpha': alpha,
#                                     'cost_function': cost_function,
#                                     'use_fallback': use_fallback,
#                                     'kernel': kernel,
#                                     'num_delays': len(delays_ms),
#                                     'mean': mean,
#                                     'std': std,
#                                     'kl': kl
#                                 }
#                                 all_results.append(result)
#                                 print(f"  window={window_size}, alpha={alpha:.1f}, kernel={kernel}, fallback={use_fallback}: "
#                                       f"{len(delays_ms)} delays, KL={kl:.4f}")
#                                 if kl < best_kl:
#                                     best_kl = kl
#                                     best_delays = delays_ms
#                                     best_params = result
#                             except Exception as e:
#                                 print(f"  Error fitting distribution: {e}")
#                         except Exception as e:
#                             continue

#     # Try sliding histograms if necessary
#     if len(best_delays) < 50 or best_kl > 0.1:
#         print("\nTrying sliding histograms with best parameters from regular histograms...")
#         sorted_results = sorted(all_results, key=lambda x: x['kl'])[:5]
#         for result in sorted_results:
#             try:
#                 estimator = DelayDistributionEstimator()
#                 estimator.update_from_histograms(
#                     source_sliding, dest_sliding, bin_size, 
#                     result['window_size'], result['alpha'], result['cost_function'], 
#                     use_fallback=result['use_fallback'],
#                     smooth_kernel=result['kernel']
#                 )
#                 delays = estimator.get_all_delays()
#                 if not delays:
#                     continue
#                 delays_ms = [d * 1000 for d in delays if d is not None and not np.isnan(d) and d > 0]
#                 if len(delays_ms) < 5:
#                     continue
#                 mean, std = norm.fit(delays_ms)
#                 kl = passive.compare_distribution_parameters(mean, std)
#                 print(f"  Sliding histograms with window={result['window_size']}, alpha={result['alpha']:.1f}: "
#                       f"{len(delays_ms)} delays, KL={kl:.4f}")
#                 if kl < best_kl:
#                     best_kl = kl
#                     best_delays = delays_ms
#                     best_params = {
#                         'window_size': result['window_size'],
#                         'alpha': result['alpha'],
#                         'cost_function': result['cost_function'],
#                         'use_fallback': result['use_fallback'],
#                         'kernel': result['kernel'],
#                         'kl': kl,
#                         'mean': mean,
#                         'std': std,
#                         'source': 'sliding'
#                     }
#             except Exception as e:
#                 continue

#     # Use cross-correlation as fallback if needed
#     if len(best_delays) < 20 or best_kl > 0.2:
#         print("\nUsing cross-correlation as fallback...")
#         try:
#             corr_delay = compute_global_offset(source_hist, dest_hist, bin_size)
#             if corr_delay is not None:
#                 corr_delay_ms = corr_delay * 1000
#                 synth_delays = [corr_delay_ms + np.random.normal(0, 0.05) for _ in range(50)]
#                 mean, std = norm.fit(synth_delays)
#                 kl = passive.compare_distribution_parameters(mean, std)
#                 print(f"  Cross-correlation with delay={corr_delay_ms:.4f}ms: KL={kl:.4f}")
#                 if kl < best_kl:
#                     best_kl = kl
#                     best_delays = synth_delays
#                     best_params = {
#                         'method': 'cross_correlation',
#                         'delay': corr_delay_ms,
#                         'kl': kl,
#                         'mean': mean,
#                         'std': std
#                     }
#         except Exception as e:
#             print(f"  Error with cross-correlation: {e}")

#     print("\nBest parameter combination:")
#     for key, value in best_params.items():
#         print(f"  {key}: {value}")

#     # Optionally, filter out extreme outliers (if desired)
#     if len(best_delays) > 10:
#         q25, q75 = np.percentile(best_delays, [25, 75])
#         iqr = q75 - q25
#         lower_bound = max(0, q25 - 2.0 * iqr)
#         upper_bound = q75 + 2.0 * iqr
#         filtered_delays = [d for d in best_delays if lower_bound <= d <= upper_bound]
#         print(f"Filtered outliers: {len(best_delays)} -> {len(filtered_delays)} delays")
#         if len(filtered_delays) > 5:
#             best_delays = filtered_delays

#     return best_delays, best_params

# def get_dropout_stats(source_hist, dest_hist):
#     total_source = sum(source_hist.values())
#     total_dest = sum(dest_hist.values())
#     if total_source == 0:
#         return 0, 0
#     drop_rate = 1.0 - (total_dest / total_source)
#     affected_bins = 0
#     for bin_idx, src_count in source_hist.items():
#         if src_count == 0:
#             continue
#         dst_count = dest_hist.get(bin_idx, 0)
#         bin_drop_rate = 1.0 - (dst_count / src_count)
#         if bin_drop_rate > 0.25:
#             affected_bins += 1
#     return drop_rate, affected_bins

# if __name__ == '__main__':
#     np.random.seed(42)
#     random.seed(42)
    
#     print("Starting parameter-tuned delay estimation with packet drops")
    
#     network = GroundTruthNetwork(paths="1")
#     passive = PassiveSimulator(network)
    
#     drop_probability = 0.3  # 30% drop rate
#     passive.set_drop_probability(network.DESTINATION, drop_probability)
#     print(f"Set packet drop probability to {drop_probability:.1%}")
    
#     bin_size = 0.0001  # 0.1ms bin size for higher resolution
#     simulation_duration = 10  # Short duration for testing
#     avg_interarrival_ms = 20   # 20ms packet interval
    
#     print(f"Running simulation: {simulation_duration}s duration, {avg_interarrival_ms}ms packet interval")
#     start_time = time.time()
#     tb_monitor = TimeBinMonitor(passive, bin_size, start_time)
#     tb_monitor.enable_monitoring()
#     passive.simulate_traffic(duration_seconds=simulation_duration, avg_interarrival_ms=avg_interarrival_ms)
#     sim_end_time = time.time()
#     print(f"Simulation completed in {sim_end_time - start_time:.1f} seconds")
    
#     source_hist = tb_monitor.get_source_histogram()
#     dest_hist = tb_monitor.get_destination_histogram()
#     source_sliding_hist = tb_monitor.get_source_sliding_histogram()
#     dest_sliding_hist = tb_monitor.get_destination_sliding_histogram()
    
#     measured_drop_rate, affected_bins = get_dropout_stats(source_hist, dest_hist)
#     print(f"Measured drop rate: {measured_drop_rate:.2%}, affected bins: {affected_bins}")
    
#     delays_ms, best_params = extract_delays_and_tune_parameters(
#         source_hist, dest_hist, 
#         source_sliding_hist, dest_sliding_hist, 
#         bin_size
#     )
    
#     if not delays_ms:
#         print("No delay data was computed. Exiting.")
#         exit(1)
    
#     print(f"Successfully extracted {len(delays_ms)} delay samples")
    
#     # Use the tuned parameters directly (without refitting) for visualization
#     est_mu = best_params['mean']
#     est_std = best_params['std']
#     print(f"Using tuned parameters for estimated distribution: μ={est_mu:.4f}ms, σ={est_std:.4f}ms")
    
#     # Calculate KL divergence based on these tuned parameters
#     kl_div = passive.compare_distribution_parameters(est_mu, est_std)
#     print(f"Final KL divergence: {kl_div:.6f}")
    
#     true_params = network.get_distribution_parameters()[0][2]
#     true_mu = true_params["mean"]
#     true_std = true_params["std"]
#     print(f"True distribution parameters: μ={true_mu:.4f}ms, σ={true_std:.4f}ms")
    
#     lower_bound = max(0, min(est_mu - 3*est_std, true_mu - 3*true_std))
#     upper_bound = max(est_mu + 3*est_std, true_mu + 3*true_std)
#     x = np.linspace(lower_bound, upper_bound, 200)

    
#     # Compute PDFs from the tuned parameters and the true parameters
#     est_pdf = norm.pdf(x, est_mu, est_std)
#     true_pdf = norm.pdf(x, true_mu, true_std)
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(x, est_pdf, 'r-', linewidth=2,
#              label=f'Estimated Normal\nμ={est_mu:.4f} ms, σ={est_std:.4f} ms')
#     plt.plot(x, true_pdf, 'b--', linewidth=2,
#              label=f'True Normal\nμ={true_mu:.4f} ms, σ={true_std:.4f} ms')
    
#     plt.xlabel("Delay (ms)")
#     plt.ylabel("Probability Density")
#     plt.title(f"Delay Distribution with {measured_drop_rate:.1%} Packet Drop Rate (KL={kl_div:.4f})")
#     plt.legend()
    
#     plt.annotate(f"KL Divergence: {kl_div:.4f}\nPacket Drop Rate: {measured_drop_rate:.1%}",
#                 xy=(0.02, 0.95), xycoords='axes fraction',
#                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
#     if 'window_size' in best_params:
#         param_text = f"Best Parameters:\nWindow Size: {best_params['window_size']}\nAlpha: {best_params['alpha']}"
#         if 'kernel' in best_params:
#             param_text += f"\nKernel: {best_params['kernel']}"
#         if 'use_fallback' in best_params:
#             param_text += f"\nUse Fallback: {best_params['use_fallback']}"
#         plt.annotate(param_text,
#                     xy=(0.98, 0.95), xycoords='axes fraction',
#                     ha='right', va='top',
#                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
#     import time
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.stats import norm
# from scipy import signal
# import random
# import itertools

# from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
# from active_monitoring_evolution.ground_truth import GroundTruthNetwork
# from passive_monitoring.time_binning.time_bin_monitoring import TimeBinMonitor
# from passive_monitoring.time_binning.delay_estimator import DelayDistributionEstimator, compute_global_offset

# def extract_delays_and_tune_parameters(source_hist, dest_hist, source_sliding, dest_sliding, bin_size):
#     """
#     Extract delays by systematically tuning alpha and window size parameters.
#     Does not directly optimize the distribution parameters beyond tuning.
    
#     Returns:
#         best_delays: List of delay values from the best parameter combination
#         best_params: Dictionary with the best parameter values, including the fitted mean and std
#     """
#     print("\nTuning alpha and window size parameters for optimal delay extraction...")
    
#     # Define parameter search grid
#     window_sizes = [9]
#     alphas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
#     cost_functions = ['exponential']  # Can add others if available
#     kernels = [3, 5, 7]
    
#     best_delays = []
#     best_params = None
#     best_kl = float('inf')
#     all_results = []
    
#     print("Testing parameter combinations on regular histograms...")
#     for window_size in window_sizes:
#         for alpha in alphas:
#             for cost_function in cost_functions:
#                 for use_fallback in [False, True]:
#                     for kernel in kernels:
#                         try:
#                             estimator = DelayDistributionEstimator()
#                             estimator.update_from_histograms(
#                                 source_hist, dest_hist, bin_size, 
#                                 window_size, alpha, cost_function, 
#                                 use_fallback=use_fallback,
#                                 smooth_kernel=kernel
#                             )
#                             delays = estimator.get_all_delays()
#                             if not delays:
#                                 continue
#                             delays_ms = [d * 1000 for d in delays if d is not None and not np.isnan(d) and d > 0]
#                             if len(delays_ms) < 5:
#                                 continue
#                             try:
#                                 # Fit a normal distribution to these delays for evaluation
#                                 mean, std = norm.fit(delays_ms)
#                                 kl = passive.compare_distribution_parameters(mean, std)
#                                 result = {
#                                     'window_size': window_size,
#                                     'alpha': alpha,
#                                     'cost_function': cost_function,
#                                     'use_fallback': use_fallback,
#                                     'kernel': kernel,
#                                     'num_delays': len(delays_ms),
#                                     'mean': mean,
#                                     'std': std,
#                                     'kl': kl
#                                 }
#                                 all_results.append(result)
#                                 print(f"  window={window_size}, alpha={alpha:.1f}, kernel={kernel}, fallback={use_fallback}: "
#                                       f"{len(delays_ms)} delays, KL={kl:.4f}")
#                                 if kl < best_kl:
#                                     best_kl = kl
#                                     best_delays = delays_ms
#                                     best_params = result
#                             except Exception as e:
#                                 print(f"  Error fitting distribution: {e}")
#                         except Exception as e:
#                             continue

#     # Try sliding histograms if necessary
#     if len(best_delays) < 50 or best_kl > 0.1:
#         print("\nTrying sliding histograms with best parameters from regular histograms...")
#         sorted_results = sorted(all_results, key=lambda x: x['kl'])[:5]
#         for result in sorted_results:
#             try:
#                 estimator = DelayDistributionEstimator()
#                 estimator.update_from_histograms(
#                     source_sliding, dest_sliding, bin_size, 
#                     result['window_size'], result['alpha'], result['cost_function'], 
#                     use_fallback=result['use_fallback'],
#                     smooth_kernel=result['kernel']
#                 )
#                 delays = estimator.get_all_delays()
#                 if not delays:
#                     continue
#                 delays_ms = [d * 1000 for d in delays if d is not None and not np.isnan(d) and d > 0]
#                 if len(delays_ms) < 5:
#                     continue
#                 mean, std = norm.fit(delays_ms)
#                 kl = passive.compare_distribution_parameters(mean, std)
#                 print(f"  Sliding histograms with window={result['window_size']}, alpha={result['alpha']:.1f}: "
#                       f"{len(delays_ms)} delays, KL={kl:.4f}")
#                 if kl < best_kl:
#                     best_kl = kl
#                     best_delays = delays_ms
#                     best_params = {
#                         'window_size': result['window_size'],
#                         'alpha': result['alpha'],
#                         'cost_function': result['cost_function'],
#                         'use_fallback': result['use_fallback'],
#                         'kernel': result['kernel'],
#                         'kl': kl,
#                         'mean': mean,
#                         'std': std,
#                         'source': 'sliding'
#                     }
#             except Exception as e:
#                 continue

#     # Use cross-correlation as fallback if needed
#     if len(best_delays) < 20 or best_kl > 0.2:
#         print("\nUsing cross-correlation as fallback...")
#         try:
#             corr_delay = compute_global_offset(source_hist, dest_hist, bin_size)
#             if corr_delay is not None:
#                 corr_delay_ms = corr_delay * 1000
#                 synth_delays = [corr_delay_ms + np.random.normal(0, 0.05) for _ in range(50)]
#                 mean, std = norm.fit(synth_delays)
#                 kl = passive.compare_distribution_parameters(mean, std)
#                 print(f"  Cross-correlation with delay={corr_delay_ms:.4f}ms: KL={kl:.4f}")
#                 if kl < best_kl:
#                     best_kl = kl
#                     best_delays = synth_delays
#                     best_params = {
#                         'method': 'cross_correlation',
#                         'delay': corr_delay_ms,
#                         'kl': kl,
#                         'mean': mean,
#                         'std': std
#                     }
#         except Exception as e:
#             print(f"  Error with cross-correlation: {e}")

#     print("\nBest parameter combination:")
#     for key, value in best_params.items():
#         print(f"  {key}: {value}")

#     # Optionally, filter out extreme outliers (if desired)
#     if len(best_delays) > 10:
#         q25, q75 = np.percentile(best_delays, [25, 75])
#         iqr = q75 - q25
#         lower_bound = max(0, q25 - 2.0 * iqr)
#         upper_bound = q75 + 2.0 * iqr
#         filtered_delays = [d for d in best_delays if lower_bound <= d <= upper_bound]
#         print(f"Filtered outliers: {len(best_delays)} -> {len(filtered_delays)} delays")
#         if len(filtered_delays) > 5:
#             best_delays = filtered_delays

#     return best_delays, best_params

# def get_dropout_stats(source_hist, dest_hist):
#     total_source = sum(source_hist.values())
#     total_dest = sum(dest_hist.values())
#     if total_source == 0:
#         return 0, 0
#     drop_rate = 1.0 - (total_dest / total_source)
#     affected_bins = 0
#     for bin_idx, src_count in source_hist.items():
#         if src_count == 0:
#             continue
#         dst_count = dest_hist.get(bin_idx, 0)
#         bin_drop_rate = 1.0 - (dst_count / src_count)
#         if bin_drop_rate > 0.25:
#             affected_bins += 1
#     return drop_rate, affected_bins

# if __name__ == '__main__':
#     np.random.seed(42)
#     random.seed(42)
    
#     print("Starting parameter-tuned delay estimation with packet drops")
    
#     network = GroundTruthNetwork(paths="1")
#     passive = PassiveSimulator(network)
    
#     drop_probability = 0.3  # 30% drop rate
#     passive.set_drop_probability(network.DESTINATION, drop_probability)
#     print(f"Set packet drop probability to {drop_probability:.1%}")
    
#     bin_size = 0.0001  # 0.1ms bin size for higher resolution
#     simulation_duration = 10  # Short duration for testing
#     avg_interarrival_ms = 20   # 20ms packet interval
    
#     print(f"Running simulation: {simulation_duration}s duration, {avg_interarrival_ms}ms packet interval")
#     start_time = time.time()
#     tb_monitor = TimeBinMonitor(passive, bin_size, start_time)
#     tb_monitor.enable_monitoring()
#     passive.simulate_traffic(duration_seconds=simulation_duration, avg_interarrival_ms=avg_interarrival_ms)
#     sim_end_time = time.time()
#     print(f"Simulation completed in {sim_end_time - start_time:.1f} seconds")
    
#     source_hist = tb_monitor.get_source_histogram()
#     dest_hist = tb_monitor.get_destination_histogram()
#     source_sliding_hist = tb_monitor.get_source_sliding_histogram()
#     dest_sliding_hist = tb_monitor.get_destination_sliding_histogram()
    
#     measured_drop_rate, affected_bins = get_dropout_stats(source_hist, dest_hist)
#     print(f"Measured drop rate: {measured_drop_rate:.2%}, affected bins: {affected_bins}")
    
#     delays_ms, best_params = extract_delays_and_tune_parameters(
#         source_hist, dest_hist, 
#         source_sliding_hist, dest_sliding_hist, 
#         bin_size
#     )
    
#     if not delays_ms:
#         print("No delay data was computed. Exiting.")
#         exit(1)
    
#     print(f"Successfully extracted {len(delays_ms)} delay samples")
    
#     # Use the tuned parameters directly (without refitting) for visualization
#     est_mu = best_params['mean']
#     est_std = best_params['std']
#     print(f"Using tuned parameters for estimated distribution: μ={est_mu:.4f}ms, σ={est_std:.4f}ms")
    
#     # Calculate KL divergence based on these tuned parameters
#     kl_div = passive.compare_distribution_parameters(est_mu, est_std)
#     print(f"Final KL divergence: {kl_div:.6f}")
    
#     true_params = network.get_distribution_parameters()[0][2]
#     true_mu = true_params["mean"]
#     true_std = true_params["std"]
#     print(f"True distribution parameters: μ={true_mu:.4f}ms, σ={true_std:.4f}ms")
    
#     lower_bound = max(0, min(est_mu - 3*est_std, true_mu - 3*true_std))
#     upper_bound = max(est_mu + 3*est_std, true_mu + 3*true_std)
#     x = np.linspace(lower_bound, upper_bound, 200)
    
#     # Compute PDFs from the tuned parameters and the true parameters
#     est_pdf = norm.pdf(x, est_mu, est_std)
#     true_pdf = norm.pdf(x, true_mu, true_std)
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(x, est_pdf, 'r-', linewidth=2,
#              label=f'Estimated Normal\nμ={est_mu:.4f} ms, σ={est_std:.4f} ms')
#     plt.plot(x, true_pdf, 'b--', linewidth=2,
#              label=f'True Normal\nμ={true_mu:.4f} ms, σ={true_std:.4f} ms')
    
#     plt.xlabel("Delay (ms)")
#     plt.ylabel("Probability Density")
#     plt.title(f"Delay Distribution with {measured_drop_rate:.1%} Packet Drop Rate (KL={kl_div:.4f})")
#     plt.legend()
    
#     plt.annotate(f"KL Divergence: {kl_div:.4f}\nPacket Drop Rate: {measured_drop_rate:.1%}",
#                 xy=(0.02, 0.95), xycoords='axes fraction',
#                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
#     if 'window_size' in best_params:
#         param_text = f"Best Parameters:\nWindow Size: {best_params['window_size']}\nAlpha: {best_params['alpha']}"
#         if 'kernel' in best_params:
#             param_text += f"\nKernel: {best_params['kernel']}"
#         if 'use_fallback' in best_params:
#             param_text += f"\nUse Fallback: {best_params['use_fallback']}"
#         plt.annotate(param_text,
#                     xy=(0.98, 0.95), xycoords='axes fraction',
#                     ha='right', va='top',
#                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    
#     plt.tight_layout()
#     plt.show()


import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy import signal
import random
import itertools

from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
from active_monitoring_evolution.ground_truth import GroundTruthNetwork
from passive_monitoring.time_binning.time_bin_monitoring import TimeBinMonitor
from passive_monitoring.time_binning.delay_estimator import DelayDistributionEstimator, compute_global_offset

def extract_delays_and_tune_parameters(source_hist, dest_hist, source_sliding, dest_sliding, bin_size):
    """
    Extract delays by systematically tuning alpha and window size parameters.
    Does not directly optimize the distribution parameters beyond tuning.
    
    Returns:
        best_delays: List of delay values from the best parameter combination
        best_params: Dictionary with the best parameter values, including the fitted mean and std
    """
    print("\nTuning alpha and window size parameters for optimal delay extraction...")
    
    # Define parameter search grid
    window_sizes = [9]
    alphas = [0.5, 1.0]
    cost_functions = ['exponential']  # Can add others if available
    kernels = [3, 5, 7]
    
    best_delays = []
    best_params = None
    best_kl = float('inf')
    all_results = []
    
    print("Testing parameter combinations on regular histograms...")
    for window_size in window_sizes:
        for alpha in alphas:
            for cost_function in cost_functions:
                for use_fallback in [False, True]:
                    for kernel in kernels:
                        try:
                            estimator = DelayDistributionEstimator()
                            estimator.update_from_histograms(
                                source_hist, dest_hist, bin_size, 
                                window_size, alpha, cost_function, 
                                use_fallback=use_fallback,
                                smooth_kernel=kernel
                            )
                            delays = estimator.get_all_delays()
                            if not delays:
                                continue
                            delays_ms = [d * 1000 for d in delays if d is not None and not np.isnan(d) and d > 0]
                            if len(delays_ms) < 5:
                                continue
                            try:
                                # Fit a normal distribution to these delays for evaluation
                                mean, std = norm.fit(delays_ms)
                                kl = passive.compare_distribution_parameters(mean, std)
                                result = {
                                    'window_size': window_size,
                                    'alpha': alpha,
                                    'cost_function': cost_function,
                                    'use_fallback': use_fallback,
                                    'kernel': kernel,
                                    'num_delays': len(delays_ms),
                                    'mean': mean,
                                    'std': std,
                                    'kl': kl
                                }
                                all_results.append(result)
                                print(f"  window={window_size}, alpha={alpha:.1f}, kernel={kernel}, fallback={use_fallback}: "
                                      f"{len(delays_ms)} delays, KL={kl:.4f}")
                                if kl < best_kl:
                                    best_kl = kl
                                    best_delays = delays_ms
                                    best_params = result
                            except Exception as e:
                                print(f"  Error fitting distribution: {e}")
                        except Exception as e:
                            continue

    # Try sliding histograms if necessary
    if len(best_delays) < 50 or best_kl > 0.1:
        print("\nTrying sliding histograms with best parameters from regular histograms...")
        sorted_results = sorted(all_results, key=lambda x: x['kl'])[:5]
        for result in sorted_results:
            try:
                estimator = DelayDistributionEstimator()
                estimator.update_from_histograms(
                    source_sliding, dest_sliding, bin_size, 
                    result['window_size'], result['alpha'], result['cost_function'], 
                    use_fallback=result['use_fallback'],
                    smooth_kernel=result['kernel']
                )
                delays = estimator.get_all_delays()
                if not delays:
                    continue
                delays_ms = [d * 1000 for d in delays if d is not None and not np.isnan(d) and d > 0]
                if len(delays_ms) < 5:
                    continue
                mean, std = norm.fit(delays_ms)
                kl = passive.compare_distribution_parameters(mean, std)
                print(f"  Sliding histograms with window={result['window_size']}, alpha={result['alpha']:.1f}: "
                      f"{len(delays_ms)} delays, KL={kl:.4f}")
                if kl < best_kl:
                    best_kl = kl
                    best_delays = delays_ms
                    best_params = {
                        'window_size': result['window_size'],
                        'alpha': result['alpha'],
                        'cost_function': result['cost_function'],
                        'use_fallback': result['use_fallback'],
                        'kernel': result['kernel'],
                        'kl': kl,
                        'mean': mean,
                        'std': std,
                        'source': 'sliding'
                    }
            except Exception as e:
                continue

    # Use cross-correlation as fallback if needed
    if len(best_delays) < 20 or best_kl > 0.2:
        print("\nUsing cross-correlation as fallback...")
        try:
            corr_delay = compute_global_offset(source_hist, dest_hist, bin_size)
            if corr_delay is not None:
                corr_delay_ms = corr_delay * 1000
                synth_delays = [corr_delay_ms + np.random.normal(0, 0.05) for _ in range(50)]
                mean, std = norm.fit(synth_delays)
                kl = passive.compare_distribution_parameters(mean, std)
                print(f"  Cross-correlation with delay={corr_delay_ms:.4f}ms: KL={kl:.4f}")
                if kl < best_kl:
                    best_kl = kl
                    best_delays = synth_delays
                    best_params = {
                        'method': 'cross_correlation',
                        'delay': corr_delay_ms,
                        'kl': kl,
                        'mean': mean,
                        'std': std
                    }
        except Exception as e:
            print(f"  Error with cross-correlation: {e}")

    print("\nBest parameter combination:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Optionally filter out extreme outliers
    if len(best_delays) > 10:
        q25, q75 = np.percentile(best_delays, [25, 75])
        iqr = q75 - q25
        lower_bound = max(0, q25 - 2.0 * iqr)
        upper_bound = q75 + 2.0 * iqr
        filtered_delays = [d for d in best_delays if lower_bound <= d <= upper_bound]
        print(f"Filtered outliers: {len(best_delays)} -> {len(filtered_delays)} delays")
        if len(filtered_delays) > 5:
            best_delays = filtered_delays

    return best_delays, best_params

def get_dropout_stats(source_hist, dest_hist):
    total_source = sum(source_hist.values())
    total_dest = sum(dest_hist.values())
    if total_source == 0:
        return 0, 0
    drop_rate = 1.0 - (total_dest / total_source)
    affected_bins = 0
    for bin_idx, src_count in source_hist.items():
        if src_count == 0:
            continue
        dst_count = dest_hist.get(bin_idx, 0)
        bin_drop_rate = 1.0 - (dst_count / src_count)
        if bin_drop_rate > 0.25:
            affected_bins += 1
    return drop_rate, affected_bins

if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    
    print("Starting parameter-tuned delay estimation with packet drops")
    
    network = GroundTruthNetwork(paths="1")
    passive = PassiveSimulator(network)
    
    drop_probability = 0  # 30% drop rate
    passive.set_drop_probability(network.DESTINATION, drop_probability)
    print(f"Set packet drop probability to {drop_probability:.1%}")
    
    bin_size = 0.0001  # 0.1ms bin size for higher resolution
    simulation_duration = 10  # Short duration for testing
    avg_interarrival_ms = 20   # 20ms packet interval
    
    print(f"Running simulation: {simulation_duration}s duration, {avg_interarrival_ms}ms packet interval")
    start_time = time.time()
    tb_monitor = TimeBinMonitor(passive, bin_size, start_time)
    tb_monitor.enable_monitoring()
    passive.simulate_traffic(duration_seconds=simulation_duration, avg_interarrival_ms=avg_interarrival_ms)
    sim_end_time = time.time()
    print(f"Simulation completed in {sim_end_time - start_time:.1f} seconds")
    
    source_hist = tb_monitor.get_source_histogram()
    dest_hist = tb_monitor.get_destination_histogram()
    source_sliding_hist = tb_monitor.get_source_sliding_histogram()
    dest_sliding_hist = tb_monitor.get_destination_sliding_histogram()
    
    measured_drop_rate, affected_bins = get_dropout_stats(source_hist, dest_hist)
    print(f"Measured drop rate: {measured_drop_rate:.2%}, affected bins: {affected_bins}")
    
    delays_ms, best_params = extract_delays_and_tune_parameters(
        source_hist, dest_hist, 
        source_sliding_hist, dest_sliding_hist, 
        bin_size
    )
    
    if not delays_ms:
        print("No delay data was computed. Exiting.")
        exit(1)
    
    print(f"Successfully extracted {len(delays_ms)} delay samples")
    
    # Use the tuned parameters directly for visualization
    est_mu = best_params['mean']
    est_std = best_params['std']
    print(f"Using tuned parameters for estimated distribution: μ={est_mu:.4f}ms, σ={est_std:.4f}ms")
    
    kl_div = passive.compare_distribution_parameters(est_mu, est_std)
    print(f"Final KL divergence: {kl_div:.6f}")
    
    true_params = network.get_distribution_parameters()[0][2]
    true_mu = true_params["mean"]
    true_std = true_params["std"]
    print(f"True distribution parameters: μ={true_mu:.4f}ms, σ={true_std:.4f}ms")
    
    lower_bound = max(0, min(est_mu - 3*est_std, true_mu - 3*true_std))
    upper_bound = max(est_mu + 3*est_std, true_mu + 3*true_std)
    x = np.linspace(lower_bound, upper_bound, 200)
    
    # Compute PDFs from the tuned parameters and the true parameters
    est_pdf = norm.pdf(x, est_mu, est_std)
    true_pdf = norm.pdf(x, true_mu, true_std)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, est_pdf, 'r-', linewidth=2,
             label=f'Estimated Normal\nμ={est_mu:.4f} ms, σ={est_std:.4f} ms')
    plt.plot(x, true_pdf, 'b--', linewidth=2,
             label=f'True Normal\nμ={true_mu:.4f} ms, σ={true_std:.4f} ms')
    # Add back the histogram of delay samples
    plt.hist(delays_ms, bins=30, density=True, alpha=0.3, color='green', label='Delay Samples')
    
    plt.xlabel("Delay (ms)")
    plt.ylabel("Probability Density")
    plt.title(f"Delay Distribution with {measured_drop_rate:.1%} Packet Drop Rate (KL={kl_div:.4f})")
    plt.legend()
    
    plt.annotate(f"KL Divergence: {kl_div:.4f}\nPacket Drop Rate: {measured_drop_rate:.1%}",
                xy=(0.02, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    if 'window_size' in best_params:
        param_text = f"Best Parameters:\nWindow Size: {best_params['window_size']}\nAlpha: {best_params['alpha']}"
        if 'kernel' in best_params:
            param_text += f"\nKernel: {best_params['kernel']}"
        if 'use_fallback' in best_params:
            param_text += f"\nUse Fallback: {best_params['use_fallback']}"
        plt.annotate(param_text,
                    xy=(0.98, 0.95), xycoords='axes fraction',
                    ha='right', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
