import time
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import bisect

from passive_monitoring.time_binning.time_bin_sketch import TimeBinSketch
from passive_monitoring.time_binning.time_bin_monitoring import TimeBinMonitor
from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
from active_monitoring_evolution.ground_truth import GroundTruthNetwork

def compute_delay_distribution_weighted(source_hist, dest_hist, bin_size, window_size, alpha):
    """
    Expand the histograms into lists of bin indices for each event.
    For each source event, consider destination events within a dynamic window 
    [s, s + window_size] (only future or simultaneous events).
    For each candidate destination event d, compute a weighted cost:
      cost = |d - s| * exp(alpha * |d - s|)
    Choose the candidate with the smallest cost as the match, then compute the delay.
    
    Parameters:
      source_hist: Histogram from the source (dict mapping bin index to count)
      dest_hist: Histogram from the destination (dict mapping bin index to count)
      bin_size: The size of each bin (in seconds)
      window_size: Dynamic window size (in number of bins) within which to search
      alpha: Weighting parameter for the cost function
      
    Returns:
      delays: List of delay values (in seconds) computed as (d - s)*bin_size.
    """
    source_events = []
    for bin_index in sorted(source_hist.keys()):
        source_events.extend([bin_index] * source_hist[bin_index])
    
    dest_events = []
    for bin_index in sorted(dest_hist.keys()):
        dest_events.extend([bin_index] * dest_hist[bin_index])
    
    source_events.sort()
    dest_events.sort()
    
    delays = []
    for s in source_events:
        if not dest_events:
            break  # No more destination events available.
        # Define dynamic window boundaries: only consider destination events at or after s.
        left_bound = s
        right_bound = s + window_size
        # Use bisect to quickly find the range of destination events in the window.
        l_index = bisect.bisect_left(dest_events, left_bound)
        r_index = bisect.bisect_right(dest_events, right_bound)
        
        # Build candidate list, filtering out any destination events that are before s.
        candidates = []
        for idx in range(l_index, r_index):
            d = dest_events[idx]
            if d < s:
                continue
            diff = d - s
            cost = diff * np.exp(alpha * diff)
            candidates.append((cost, idx, d))
        
        if not candidates:
            # No valid candidate found; skip this source event.
            continue
        
        # Choose the candidate with the smallest weighted cost.
        cost, chosen_index, d = min(candidates, key=lambda x: x[0])
        delays.append((d - s) * bin_size)
        # Remove the chosen destination event to enforce one-to-one matching.
        dest_events.pop(chosen_index)
    
    return delays

if __name__ == '__main__':
    # Set up the network and simulator.
    network = GroundTruthNetwork(paths="1")
    passive = PassiveSimulator(network)
    
    # Choose an appropriate bin size (in seconds); you can experiment with this value.
    bin_size = 0.0001  
    start_time = time.time()
    tb_monitor = TimeBinMonitor(passive, bin_size=bin_size, start_time=start_time)
    tb_monitor.enable_monitoring()
    
    # Simulate traffic for 10 seconds with an average interarrival time of 10 ms.
    passive.simulate_traffic(duration_seconds=100, avg_interarrival_ms=20)
    
    # Retrieve both the discrete and sliding histograms.
    source_hist = tb_monitor.get_source_histogram()
    dest_hist = tb_monitor.get_destination_histogram()
    source_sliding_hist = tb_monitor.get_source_sliding_histogram()
    dest_sliding_hist = tb_monitor.get_destination_sliding_histogram()
    
    # Define parameters for dynamic window matching.
    window_size = 10  # in bin units; adjust as needed
    alpha = 1    # weighting parameter; adjust to penalize larger differences more strongly
    
    # Compute delay distributions using weighted matching with dynamic windows.
    delays_discrete = compute_delay_distribution_weighted(source_hist, dest_hist, bin_size, window_size, alpha)
    delays_sliding = compute_delay_distribution_weighted(source_sliding_hist, dest_sliding_hist, bin_size, window_size, alpha)
    
    # Combine delays from both discrete and sliding approaches.
    delays = delays_discrete + delays_sliding
    
    # Convert delays to milliseconds.
    delays_ms = [d * 1000 for d in delays]
    
    # Plot the combined delay distribution.
    plt.figure(figsize=(8, 5))
    n, bins, patches = plt.hist(delays_ms, bins=20, density=True, alpha=0.6, edgecolor='black')
    plt.xlabel('Delay (ms)')
    plt.ylabel('Probability Density')
    plt.title('Combined Delay Distribution with Weighted Dynamic Matching')
    
    # Fit a normal distribution to the delays and compare with the ground truth.
    if delays_ms:
        mu_ms, std_ms = norm.fit(delays_ms)
        print("Comparing distribution parameters in ms:")
        kl_div = passive.compare_distribution_parameters(mu_ms, std_ms)
        print(f"KL divergence: {kl_div}")
        
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu_ms, std_ms)
        plt.plot(x, p, 'r', linewidth=2, 
                 label=f'Normal Fit\nμ={mu_ms:.4f} ms\nσ={std_ms:.4f} ms')
        plt.legend()
    
    plt.show()

