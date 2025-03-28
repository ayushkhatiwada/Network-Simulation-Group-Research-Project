import bisect
import numpy as np
import math
from scipy.signal import convolve

def smooth_histogram(hist, kernel_size=3):
    """
    Smooths a histogram (dict of {bin: count}) using a simple moving-average.
    Returns a new histogram (as a dict) covering the same range.
    """
    if not hist:
        return {}
    # Determine the full range of bins.
    min_bin = min(hist.keys())
    max_bin = max(hist.keys())
    arr = np.zeros(max_bin - min_bin + 1)
    for b, count in hist.items():
        arr[b - min_bin] = count
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = convolve(arr, kernel, mode='same')
    return {b + min_bin: smoothed[i] for i, b in enumerate(range(min_bin, max_bin+1))}

def compute_global_offset(source_hist, dest_hist, bin_size):
    """
    Compute a global delay offset using cross-correlation.
    Histograms are converted to arrays over a common range.
    Returns the estimated offset in seconds.
    """
    if not source_hist or not dest_hist:
        return 0.0
    all_bins = set(source_hist.keys()) | set(dest_hist.keys())
    min_bin = min(all_bins)
    max_bin = max(all_bins)
    length = max_bin - min_bin + 1
    source_arr = np.zeros(length)
    dest_arr = np.zeros(length)
    for b, count in source_hist.items():
        source_arr[b - min_bin] = count
    for b, count in dest_hist.items():
        dest_arr[b - min_bin] = count
    corr = np.correlate(dest_arr, source_arr, mode='full')
    lags = np.arange(-len(source_arr)+1, len(source_arr))
    max_corr_idx = np.argmax(corr)
    best_lag = lags[max_corr_idx]
    return best_lag * bin_size

def compute_delay_distribution_weighted(source_hist, dest_hist, bin_size, window_size, alpha, cost_func='exponential'):
    """
    Original weighted matching.
    For each source event, consider destination events within [s, s+window_size].
    Compute cost for each candidate:
      - if 'exponential': cost = (d-s) * exp(alpha*(d-s))
      - if 'quadratic': cost = (d-s)^2
    If no candidate is found, skip that source event.
    
    Returns a list of delay values (in seconds).
    """
    source_events = []
    for bin_index in sorted(source_hist.keys()):
        count = source_hist[bin_index]
        source_events.extend([bin_index] * int(round(count)))
    
    dest_events = []
    for bin_index in sorted(dest_hist.keys()):
        count = dest_hist[bin_index]
        dest_events.extend([bin_index] * int(round(count)))
    
    source_events.sort()
    dest_events.sort()
    
    delays = []
    for s in source_events:
        if not dest_events:
            break
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
        dest_events.pop(chosen_index)
    
    return delays

def compute_delay_distribution_weighted_fallback(source_hist, dest_hist, bin_size, window_size, alpha, fallback_delay, cost_func='exponential'):
    """
    Improved matching with fallback.
    For each source event, consider destination events within [s, s+window_size].
    If no candidate is found, assign fallback_delay.
    
    Returns a list of delay values (in seconds).
    """
    source_events = []
    for bin_index in sorted(source_hist.keys()):
        count = source_hist[bin_index]
        source_events.extend([bin_index] * int(round(count)))
    
    dest_events = []
    for bin_index in sorted(dest_hist.keys()):
        count = dest_hist[bin_index]
        dest_events.extend([bin_index] * int(round(count)))
    
    source_events.sort()
    dest_events.sort()
    
    delays = []
    for s in source_events:
        if not dest_events:
            delays.append(fallback_delay)
            continue
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
            delays.append(fallback_delay)
            continue
        
        cost, chosen_index, d = min(candidates, key=lambda x: x[0])
        delays.append((d - s) * bin_size)
        dest_events.pop(chosen_index)
    
    return delays

class DelayDistributionEstimator:
    """
    Computes delay values from time-bin histograms using weighted matching.
    If use_fallback is True, the histograms are first smoothed and a fallback delay
    is computed via cross-correlation, then used when no candidate is found.
    """
    def __init__(self):
        self.delays = []  # delays in seconds

    def update_from_histograms(self, source_hist, dest_hist, bin_size, window_size, alpha, cost_func='exponential', use_fallback=False, smooth_kernel=3):
        if use_fallback:
            smooth_source = smooth_histogram(source_hist, kernel_size=smooth_kernel)
            smooth_dest = smooth_histogram(dest_hist, kernel_size=smooth_kernel)
            fallback_delay = compute_global_offset(smooth_source, smooth_dest, bin_size)
            delays = compute_delay_distribution_weighted_fallback(smooth_source, smooth_dest, bin_size, window_size, alpha, fallback_delay, cost_func)
        else:
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
