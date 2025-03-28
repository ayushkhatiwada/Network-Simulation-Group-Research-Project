import bisect

def compute_delay_distribution(source_hist, dest_hist, T):
    """
    Expand the snapshot histograms into lists of snapshot indices.
    Pair source and destination events (up to the minimum count) and compute
    delays as the difference in snapshot indices multiplied by T.
    """
    source_events = []
    for idx in sorted(source_hist.keys()):
        source_events.extend([idx] * source_hist[idx])
    
    dest_events = []
    for idx in sorted(dest_hist.keys()):
        dest_events.extend([idx] * dest_hist[idx])
    
    source_events.sort()
    dest_events.sort()
    
    n = min(len(source_events), len(dest_events))
    delays = [(dest_events[i] - source_events[i]) * T for i in range(n)]
    return delays

class DelayDistributionEstimator:
    """
    Computes an estimated delay distribution by pairing snapshot indices from source and destination.
    
    In this simplified version, we assume that each packet’s delay is approximated by the difference in
    snapshot indices (multiplied by T) between when it was transmitted (recorded by the source sketch)
    and when it departed (recorded by the destination sketch).
    """
    def __init__(self):
        self.delays = []
    
    def update_from_histograms(self, source_hist, dest_hist, T):
        delays = compute_delay_distribution(source_hist, dest_hist, T)
        self.delays.extend(delays)
    
    def get_all_delays(self):
        return self.delays
