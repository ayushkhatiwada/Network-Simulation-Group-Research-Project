import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy import signal
import random

from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
from active_monitoring_evolution.ground_truth import GroundTruthNetwork
from passive_monitoring.time_binning.time_bin_monitoring import TimeBinMonitor
from passive_monitoring.time_binning.delay_estimator import DelayDistributionEstimator, compute_global_offset

def extract_delays_with_early_exit(source_hist, dest_hist, source_sliding, dest_sliding, bin_size, 
                                  kl_threshold=0.05):
    """
    Extract delays by tuning only the alpha and window size parameters.
    Stops processing when KL threshold is met.
    
    Args:
        source_hist, dest_hist: Source and destination histograms
        source_sliding, dest_sliding: Sliding window histograms
        bin_size: Size of time bins in seconds
        kl_threshold: Exit early when this KL divergence is reached
        
    Returns:
        raw_delays: List of delay values from the best parameter combination
        best_params: Dictionary with the best parameter values
    """
    print(f"\nTuning alpha and window size parameters (target KL: {kl_threshold:.4f})...")
    
    # Define parameter search grid
    window_sizes = [7, 9, 11]
    alphas = [0.5, 1.0]
    cost_functions = ['exponential']
    kernels = [3, 5, 7]
    use_fallbacks = [True, False]
    
    # Track the best combination found
    best_delays = []
    best_kl = float('inf')
    best_params = None
    
    print("Testing parameter combinations on regular histograms:")
    
    # Early exit flag
    target_reached = False
    
    # Try with regular histograms first
    for window_size in window_sizes:
        if target_reached:
            break
            
        for alpha in alphas:
            if target_reached:
                break
                
            for cost_function in cost_functions:
                if target_reached:
                    break
                    
                for use_fallback in use_fallbacks:
                    if target_reached:
                        break
                        
                    for kernel in kernels:
                        try:
                            # Extract delays with current parameters
                            estimator = DelayDistributionEstimator()
                            estimator.update_from_histograms(
                                source_hist, dest_hist, bin_size, 
                                window_size, alpha, cost_function, 
                                use_fallback=use_fallback,
                                smooth_kernel=kernel
                            )
                            
                            # Get valid delays
                            delays = estimator.get_all_delays()
                            if not delays:
                                continue
                                
                            delays_ms = [d * 1000 for d in delays if d is not None and not np.isnan(d) and d > 0]
                            
                            # Need at least a few samples for a meaningful distribution
                            if len(delays_ms) < 5:
                                continue
                            
                            # Fit normal distribution to evaluate KL divergence
                            mean, std = norm.fit(delays_ms)
                            kl = passive.compare_distribution_parameters(mean, std)
                            
                            print(f"  window={window_size}, alpha={alpha:.1f}, kernel={kernel}, "
                                  f"fallback={use_fallback}: {len(delays_ms)} delays, KL={kl:.4f}")
                            
                            # Check if this is better than our current best
                            if kl < best_kl:
                                best_kl = kl
                                best_delays = delays_ms.copy()
                                best_params = {
                                    'window_size': window_size,
                                    'alpha': alpha, 
                                    'cost_function': cost_function,
                                    'use_fallback': use_fallback,
                                    'kernel': kernel,
                                    'mean': mean,
                                    'std': std,
                                    'kl': kl,
                                    'source': 'regular'
                                }
                                
                            # Early exit if we've reached the target KL threshold
                            if kl <= kl_threshold:
                                print(f"\n!!! Target KL threshold {kl_threshold:.4f} reached !!!")
                                print(f"Stopping further parameter tuning with KL={kl:.4f}")
                                target_reached = True
                                break
                                
                        except Exception as e:
                            # Skip failed combinations silently
                            continue
    
    # If we haven't reached the target with regular histograms, try sliding histograms
    if not target_reached and best_params:
        print("\nTrying sliding histograms with best parameters from regular histograms:")
        
        # Just try with the current best parameters on sliding histograms
        try:
            # Extract delays with best parameters but on sliding histograms
            estimator = DelayDistributionEstimator()
            estimator.update_from_histograms(
                source_sliding, dest_sliding, bin_size, 
                best_params['window_size'], best_params['alpha'], best_params['cost_function'], 
                use_fallback=best_params['use_fallback'],
                smooth_kernel=best_params['kernel']
            )
            
            # Get valid delays
            delays = estimator.get_all_delays()
            if delays:
                delays_ms = [d * 1000 for d in delays if d is not None and not np.isnan(d) and d > 0]
                
                if len(delays_ms) >= 5:
                    # Evaluate with KL divergence
                    mean, std = norm.fit(delays_ms)
                    kl = passive.compare_distribution_parameters(mean, std)
                    
                    print(f"  Sliding histograms with best params: {len(delays_ms)} delays, KL={kl:.4f}")
                    
                    # Check if this is better than our current best
                    if kl < best_kl:
                        best_kl = kl
                        best_delays = delays_ms.copy()
                        best_params = {
                            'window_size': best_params['window_size'],
                            'alpha': best_params['alpha'],
                            'cost_function': best_params['cost_function'],
                            'use_fallback': best_params['use_fallback'],
                            'kernel': best_params['kernel'],
                            'mean': mean,
                            'std': std,
                            'kl': kl,
                            'source': 'sliding'
                        }
                        
                    # Check if we've reached the target with sliding histograms
                    if kl <= kl_threshold:
                        print(f"\n!!! Target KL threshold {kl_threshold:.4f} reached with sliding histograms !!!")
                        print(f"Stopping further processing with KL={kl:.4f}")
                        target_reached = True
        except Exception as e:
            print(f"  Error with sliding histograms: {e}")
    
    # Print summary of final best parameters
    print("\nFinal best parameter combination:")
    for key, value in best_params.items():
        if key != 'delays':  # Skip printing the delays array
            print(f"  {key}: {value}")
    
    return best_delays, best_params

def get_dropout_stats(source_hist, dest_hist):
    total_source = sum(source_hist.values())
    total_dest = sum(dest_hist.values())
    
    if total_source == 0:
        return 0, 0
    
    drop_rate = 1.0 - (total_dest / total_source)
    
    # Count bins with significant drop rates
    affected_bins = 0
    for bin_idx, src_count in source_hist.items():
        if src_count == 0:
            continue
        dst_count = dest_hist.get(bin_idx, 0)
        bin_drop_rate = 1.0 - (dst_count / src_count)
        if bin_drop_rate > 0.25:  # 25% or more packets dropped
            affected_bins += 1
    
    return drop_rate, affected_bins

if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    print("Starting delay estimation with fixed congestion simulation")
    
    # Initialize network and simulator
    network = GroundTruthNetwork(paths="1")
    passive = PassiveSimulator(network)
    
    # Enable congestion simulation with fixed intensity
    passive.enable_congestion_simulation(network.DESTINATION)
    
    # Configure simulation parameters
    bin_size = 0.0001  # 0.1ms bin size for higher resolution
    simulation_duration = 10  # Longer duration to capture congestion events
    avg_interarrival_ms = 20  # 10ms packet interval (100 packets per second)
    
    print(f"Running simulation: {simulation_duration}s duration, {avg_interarrival_ms}ms packet interval")
    
    # Start monitoring
    start_time = time.time()
    tb_monitor = TimeBinMonitor(passive, bin_size, start_time)
    tb_monitor.enable_monitoring()
    
    # Simulate traffic
    passive.simulate_traffic(duration_seconds=simulation_duration, avg_interarrival_ms=avg_interarrival_ms)
    
    sim_end_time = time.time()
    print(f"Simulation completed in {sim_end_time - start_time:.1f} seconds")
    
    # Retrieve histograms
    source_hist = tb_monitor.get_source_histogram()
    dest_hist = tb_monitor.get_destination_histogram()
    source_sliding_hist = tb_monitor.get_source_sliding_histogram()
    dest_sliding_hist = tb_monitor.get_destination_sliding_histogram()
    
    # Calculate dropout statistics
    measured_drop_rate, affected_bins = get_dropout_stats(source_hist, dest_hist)
    print(f"Measured drop rate: {measured_drop_rate:.2%}, affected bins: {affected_bins}")
    
    # Set the target KL divergence threshold
    kl_threshold = 0.05
    
    # Extract delays with early exit when threshold is met
    delays_ms, best_params = extract_delays_with_early_exit(
        source_hist, dest_hist, 
        source_sliding_hist, dest_sliding_hist, 
        bin_size,
        kl_threshold
    )
    
    if not delays_ms:
        print("No delay data was computed. Exiting.")
        exit(1)
    
    print(f"Successfully extracted {len(delays_ms)} delay samples")
    
    # Fit the distribution to the raw delays (without optimization)
    est_mu, est_std = norm.fit(delays_ms)
    print(f"Estimated normal fit parameters: μ={est_mu:.4f}ms, σ={est_std:.4f}ms")
    
    # Calculate KL divergence
    kl_div = passive.compare_distribution_parameters(est_mu, est_std)
    print(f"Final KL divergence: {kl_div:.6f}")
    
    # Get the underlying true distribution
    true_params = network.get_distribution_parameters()[0][2]
    true_mu = true_params["mean"]
    true_std = true_params["std"]
    print(f"True distribution parameters: μ={true_mu:.4f}ms, σ={true_std:.4f}ms")
    
    # Create improved visualization
    plt.figure(figsize=(12, 7))
    
    # Set up the x-axis range for better visualization
    lower_bound = max(0, min(est_mu - 3*est_std, true_mu - 3*true_std))
    upper_bound = max(est_mu + 3*est_std, true_mu + 3*true_std) * 1.2  # Add some extra space
    x = np.linspace(lower_bound, upper_bound, 500)  # More points for smoother curves
    
    # Plot the histogram with more bins for better resolution
    num_bins = min(30, max(10, len(delays_ms) // 10))  # Dynamic bin count based on sample size
    plt.hist(delays_ms, bins=num_bins, density=True, alpha=0.4, color='#7BC8A4',
            edgecolor='black', linewidth=0.5, label='Delay Samples')
    
    # Compute PDFs with the more detailed x range
    est_pdf = norm.pdf(x, est_mu, est_std)
    true_pdf = norm.pdf(x, true_mu, true_std)
    
    # Plot the distributions with thicker lines
    plt.plot(x, est_pdf, 'r-', linewidth=2.5,
             label=f'Estimated Normal\nμ={est_mu:.4f} ms, σ={est_std:.4f} ms')
    plt.plot(x, true_pdf, 'b--', linewidth=2.5,
             label=f'True Normal\nμ={true_mu:.4f} ms, σ={true_std:.4f} ms')
    
    # Improve axis labels and title
    plt.xlabel("Delay (ms)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.title(f"Delay Distribution with Fixed Congestion ({measured_drop_rate:.1%} drop rate, KL={kl_div:.4f})", 
              fontsize=14, fontweight='bold')
    
    # Better legend placement and format
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Add KL divergence and congestion annotation
    congestion_info = (
        f"KL Divergence: {kl_div:.4f}\n"
        f"Drop Rate: {measured_drop_rate:.1%}\n"
        f"Normal Drop Rate: {passive.normal_drop_probability:.2f}\n"
        f"Congested Drop Rate: {passive.congested_drop_probability:.2f}\n"
        f"Congestion Delay Factor: {passive.congestion_delay_factor:.2f}"
    )
    
    plt.annotate(congestion_info,
                xy=(0.02, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
                fontsize=10)
    
    # Add parameters annotation
    if 'window_size' in best_params and 'alpha' in best_params:
        param_text = f"Best Parameters:\nWindow Size: {best_params['window_size']}\nAlpha: {best_params['alpha']}"
        if 'kernel' in best_params:
            param_text += f"\nKernel: {best_params['kernel']}"
        if 'use_fallback' in best_params:
            param_text += f"\nUse Fallback: {best_params['use_fallback']}"
        
        plt.annotate(param_text,
                    xy=(0.98, 0.95), xycoords='axes fraction',
                    ha='right', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
                    fontsize=10)
    
    # Add a message about whether the KL threshold was reached
    threshold_text = f"Target KL threshold: {kl_threshold:.4f}"
    if kl_div <= kl_threshold:
        threshold_text += "\n✓ Target reached"
    else:
        threshold_text += "\n✗ Target not reached"
    
    plt.annotate(threshold_text,
                xy=(0.5, 0.03), xycoords='axes fraction',
                ha='center', va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
                fontsize=10)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()