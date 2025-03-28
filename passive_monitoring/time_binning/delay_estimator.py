import bisect
import numpy as np
import math

def compute_delay_distribution_weighted(source_hist, dest_hist, bin_size, window_size, alpha, cost_func='exponential'):
    """
    Expand the histograms into lists of bin indices for each event.
    For each source event, consider destination events within the window [s, s+window_size].
    For each candidate destination event d, compute a weighted cost:
      - If cost_func == 'exponential':
            cost = (d - s) * exp(alpha * (d - s))
      - If cost_func == 'quadratic':
            cost = (d - s)^2
    The candidate with the smallest cost is selected and the delay is computed as (d - s)*bin_size.
    
    Returns:
      delays: List of delay values (in seconds).
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
        left_bound = s
        right_bound = s + window_size
        l_index = bisect.bisect_left(dest_events, left_bound)
        r_index = bisect.bisect_right(dest_events, right_bound)
        
        candidates = []
        for idx in range(l_index, r_index):
            d = dest_events[idx]
            if d < s:
                continue
            diff = d - s
            if cost_func == 'exponential':
                cost = diff * math.exp(alpha * diff)
            elif cost_func == 'quadratic':
                cost = diff**2
            else:
                cost = diff * math.exp(alpha * diff)
            candidates.append((cost, idx, d))
        
        if not candidates:
            continue
        
        cost, chosen_index, d = min(candidates, key=lambda x: x[0])
        delays.append((d - s) * bin_size)
        # Remove the chosen destination event to enforce one-to-one matching.
        dest_events.pop(chosen_index)
    
    return delays

class DelayDistributionEstimator:
    """
    This class uses weighted dynamic matching on time-bin histograms to compute delay values.
    It supports tuning via parameters (bin_size, window_size, alpha, and cost function type)
    and is designed to be extended to handle packet drops, congestion, etc.
    """
    def __init__(self):
        self.delays = []
    
    def update_from_histograms(self, source_hist, dest_hist, bin_size, window_size, alpha, cost_func='exponential'):
        delays = compute_delay_distribution_weighted(source_hist, dest_hist, bin_size, window_size, alpha, cost_func)
        self.delays.extend(delays)
    
    def get_all_delays(self):
        return self.delays
    
    def get_summary(self):
        if not self.delays:
            return {}
        arr = np.array(self.delays)
        return {
            'mean': np.mean(arr),
            'median': np.median(arr),
            'p90': np.percentile(arr, 90)
        }
    
    def get_quantile(self, q):
        if not self.delays:
            return None
        return np.percentile(self.delays, q * 100)
