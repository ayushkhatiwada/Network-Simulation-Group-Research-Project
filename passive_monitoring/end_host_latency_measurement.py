import time
import numpy as np
from collections import deque

class EndHostEstimation:
    def __init__(self, window_size=50, apply_filtering=False, discard_method=None, 
                 true_mean=0.8, true_std=0.15, 
                 true_congested_mean=0.015, true_congested_std=0.000225):
        """
        End-host sketch for real-time latency monitoring and delay distribution estimation.
        Assumes the actual ground-truth latency is stored in packet.true_delay (in ms).
        """
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.apply_filtering = apply_filtering
        self.discard_method = discard_method
        self.discard_fraction = 0.1
        
        self.true_mean = true_mean
        self.true_std = true_std
        self.true_congested_mean = true_congested_mean
        self.true_congested_std = true_congested_std

        self.total_packets = 0
        self.congestion_scores = []
        self.last_report_time = time.time()
        self.report_interval = 1.0

        self.detection_threshold = 0.05
        self.current_state = "NORMAL"
        self.state_change_times = []

    def process_packet(self, packet, switch_id):
        """Called by the switch when a packet arrives. Stores true delay if available."""
        if hasattr(packet, 'true_delay'):
            # latency = packet.true_delay / 1000.0  # convert ms to seconds
            latency = packet.true_delay
            self.update(latency)
        else:
            print("[WARNING] Packet is missing 'true_delay' attribute.")
        self.total_packets += 1

    def update(self, latency):
        """Update the sliding window with the given latency sample."""
        if 0 < latency < 1.0:
            self.latencies.append(latency)

    def estimate_parameters(self):
        """Estimate the mean and std deviation of the normal delay distribution."""
        if len(self.latencies) < 3:
            return {'estimated_mean': None, 'estimated_std': None}
        # print(self.latencies[-1])
        latencies = np.array(self.latencies)

        # Apply optional filtering
        if self.apply_filtering:
            if self.discard_method == "trimmed":
                lower = int(len(latencies) * self.discard_fraction)
                upper = len(latencies) - lower
                if lower < upper:
                    latencies = np.sort(latencies)[lower:upper]
            elif self.discard_method == "median_filter":
                median = np.median(latencies)
                dev = np.abs(latencies - median)
                thresh = np.median(dev) * 2
                latencies = latencies[dev < thresh]
            elif self.discard_method == "threshold":
                mean = np.mean(latencies)
                std = np.std(latencies)
                latencies = latencies[(latencies >= mean - 2*std) & (latencies <= mean + 2*std)]

        # print(f"returning pred_mean: {np.mean(latencies)}, pred_std: {np.std(latencies, ddof=1)}")
        return {
            'estimated_mean': np.mean(latencies),
            'estimated_std': np.std(latencies, ddof=1)
        }

    def kl_divergence(self, mu0, var0, mu1, var1):
        """KL divergence between two normal distributions."""
        if var0 <= 0 or var1 <= 0:
            return float('inf')
        return 0.5 * ((var0 / var1) + ((mu1 - mu0) ** 2) / var1 - 1 + np.log(var1 / var0))

    def get_summary(self):
        """Returns a summary of state changes (if congestion detection is used)."""
        if not self.state_change_times:
            return "No congestion events detected."

        summary = f"Detected {len(self.state_change_times)} state changes:\n"
        for i, (timestamp, old_state, new_state) in enumerate(self.state_change_times):
            summary += f"  {i+1}. Time: {timestamp:.2f}s - {old_state} -> {new_state}\n"
        return summary
