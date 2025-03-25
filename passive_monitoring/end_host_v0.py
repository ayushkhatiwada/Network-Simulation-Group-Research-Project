import time
import numpy as np
from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
from active_monitoring_evolution.ground_truth import GroundTruthNetwork
from passive_monitoring.end_host_latency_measurement import EndHostEstimation

def evolution_0_find_optimal_window():
    """
    Evolution 0: Find the optimal window size for accurate latency monitoring
    assuming exponential delay distribution and no packet loss or congestion.
    """
    print("=== Evolution 0: Finding Optimal Window Size (Exponential Distribution, No Congestion) ===")
    
    # Parameters
    target_kl = 0.05
    min_window_size = 10
    max_window_size = 500
    window_increment = 10
    trials_per_window = 10
    simulation_duration = 5  # seconds
    
    # Ground truth exponential parameters
    true_mean = 0.01  # seconds
    true_lambda = 1 / true_mean

    # Store results
    results = []
    
    for window_size in range(min_window_size, max_window_size + 1, window_increment):
        print(f"\nTesting window size: {window_size}")
        kl_scores = []
        
        for trial in range(trials_per_window):
            # Create new network and simulator for each trial
            network = GroundTruthNetwork()
            simulator = PassiveSimulator(network)
            
            # Set normal drop probability to 0 (no dropping)
            simulator.normal_drop_probability = 0.0
            
            # Create latency monitor with the current window size
            monitor = EndHostEstimation(
                window_size=window_size,
                apply_filtering=False,
                # discard_method="median_filter",
                true_mean=true_mean  # Used internally for comparison
            )
            
            # Disable periodic print reporting
            monitor.report_interval = float('inf')
            
            # Attach monitor to destination switch
            simulator.attach_sketch(network.DESTINATION, monitor)
            
            # Run simulation
            simulator.simulate_traffic(duration_seconds=simulation_duration, avg_interarrival_ms=10)
            
            # Get the final estimated parameters
            params = monitor.estimate_parameters()
            
            if params['estimated_lambda'] is None:
                continue  # Not enough data
            
            # Calculate KL divergence between true and estimated exponential distributions
            kl_score = monitor.kl_divergence(true_lambda, params['estimated_lambda'])
            kl_scores.append(kl_score)
            
            # Optional: print progress
            if (trial + 1) % 10 == 0:
                print(f"  Completed {trial + 1}/{trials_per_window} trials")
        
        # Aggregate results
        if kl_scores:
            avg_kl = np.mean(kl_scores)
            success_rate = np.mean([kl <= target_kl for kl in kl_scores])
            results.append((window_size, avg_kl, success_rate))
            
            print(f"  Window size: {window_size}")
            print(f"  Average KL score: {avg_kl:.6f}")
            print(f"  Success rate: {success_rate:.2%}")
            
            # Early stopping if target is met
            if avg_kl <= target_kl and success_rate >= 0.8:
                print(f"\n*** Optimal window size found: {window_size} ***")
                break
    
    # Final result summary
    print("\n=== Evolution 0 Results ===")
    print("Window Size | Avg KL Score | Success Rate")
    print("----------------------------------------")
    for window_size, avg_kl, success_rate in results:
        print(f"{window_size:11d} | {avg_kl:12.6f} | {success_rate:11.2%}")
    
    # Determine smallest qualifying window
    optimal_windows = [w for w, kl, r in results if kl <= target_kl and r >= 0.8]
    if optimal_windows:
        optimal_window = min(optimal_windows)
        print(f"\nSmallest window size for achieving KL score ≤ {target_kl}: {optimal_window}")
    else:
        print(f"\nNo window size achieved the target KL score of {target_kl} consistently.")
    
    return results

if __name__ == "__main__":
    evolution_0_find_optimal_window()
