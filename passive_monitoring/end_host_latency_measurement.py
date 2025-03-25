
import time
import numpy as np
from collections import deque

class EndHostEstimation:
    def __init__(self, window_size=50, apply_filtering=False, discard_method=None, 
                 true_mean=0.01, true_variance=0.0001, 
                 true_congested_mean=0.015, true_congested_variance=0.000225):
        """
        Sketch that monitors latency and calculates statistics in real-time
        """
        self.window_size = window_size # number of recent latency measurements to keep
        self.latencies = deque(maxlen=window_size)
        self.apply_filtering = apply_filtering
        self.discard_method = discard_method # filter method
        self.discard_fraction = 0.1 
        
        # ground truth parameters for KL calculation
        self.true_mean = true_mean
        self.true_variance = true_variance
        self.true_congested_mean = true_congested_mean
        self.true_congested_variance = true_congested_variance
        
        # store timestamps for calculating packet-pair latencies
        self.last_packet_time = None
        self.source_switch_id = None
        self.destination_switch_id = None
        
        # tracking stats
        self.total_packets = 0
        self.congestion_scores = []
        self.last_report_time = time.time()
        self.report_interval = 1.0  # Report every second
        
        # congestion detection
        self.detection_threshold = 0.05  # KL threshold
        self.current_state = "NORMAL"
        self.state_change_times = []

    def process_packet(self, packet, switch_id):
        """Process a packet at a switch and record timing information"""
        current_time = time.time()
        
        # initialize switch IDs if not already set
        if self.source_switch_id is None and hasattr(packet, 'source'):
            self.source_switch_id = packet.source
        if self.destination_switch_id is None and hasattr(packet, 'destination'):
            self.destination_switch_id = packet.destination
            
        # record the time between consecutive packets
        if self.last_packet_time is not None:
            latency = current_time - self.last_packet_time
            self.update(latency)
            
            # check if it's time to report statistics
            if current_time - self.last_report_time >= self.report_interval:
                self.report_statistics()
                self.last_report_time = current_time
                
        self.last_packet_time = current_time
        self.total_packets += 1
    
    def update(self, latency):
        """Add a new latency measurement to the sliding window"""
        if latency > 0 and latency < 1.0: 
            self.latencies.append(latency)
    
    def estimate_parameters(self):
        """Estimate λ (rate) for exponential distribution from observed latencies"""
        if len(self.latencies) < 3:
            return {'estimated_lambda': None, 'estimated_mean': None}

        latencies = np.array(self.latencies)

        # Apply filtering if needed
        if self.apply_filtering:
            if self.discard_method == "trimmed":
                lower_bound = int(len(latencies) * self.discard_fraction)
                upper_bound = len(latencies) - lower_bound
                if lower_bound < upper_bound:
                    latencies = np.sort(latencies)[lower_bound:upper_bound]
            elif self.discard_method == "median_filter":
                median_value = np.median(latencies)
                deviation = np.abs(latencies - median_value)
                threshold = np.median(deviation) * 2
                latencies = latencies[deviation < threshold]
            elif self.discard_method == "threshold":
                mean = np.mean(latencies)
                std_dev = np.std(latencies)
                latencies = latencies[(latencies >= mean - 2 * std_dev) & (latencies <= mean + 2 * std_dev)]

        estimated_mean = np.mean(latencies)
        estimated_lambda = 1 / estimated_mean if estimated_mean > 0 else None

        return {'estimated_lambda': estimated_lambda, 'estimated_mean': estimated_mean}

    def kl_divergence(self, lambda_true, lambda_est):
        """
        KL divergence between Exponential(lambda_true) and Exponential(lambda_est)
        KL = log(lambda_est / lambda_true) + (lambda_true / lambda_est) - 1
        """
        if lambda_est is None or lambda_est <= 0 or lambda_true <= 0:
            return float('inf')
        return np.log(lambda_est / lambda_true) + (lambda_true / lambda_est) - 1


    def detect_congestion(self, estimated_lambda):
        """Detect congestion by comparing KL divergence with normal vs congested λ values"""
        kl_normal = self.kl_divergence(1 / self.true_mean, estimated_lambda)
        kl_congested = self.kl_divergence(1 / self.true_congested_mean, estimated_lambda)

        if kl_normal <= kl_congested:
            return "NORMAL", kl_normal
        else:
            return "CONGESTED", kl_congested

    def report_statistics(self):
        """Report current λ estimate and congestion state"""
        params = self.estimate_parameters()
        
        if params['estimated_lambda'] is None:
            print(f"Time: {time.time():.2f}s | Not enough data yet ({len(self.latencies)} samples)")
            return

        detected_state, kl_score = self.detect_congestion(params['estimated_lambda'])

        if detected_state != self.current_state:
            self.state_change_times.append((time.time(), self.current_state, detected_state))
            print(f"*** STATE CHANGE: {self.current_state} -> {detected_state} ***")
            self.current_state = detected_state

        self.congestion_scores.append((time.time(), kl_score, detected_state))

        print(f"Time: {time.time():.2f}s | State: {detected_state}")
        print(f"  Estimated Mean: {params['estimated_mean']:.6f}")
        print(f"  Estimated Lambda: {params['estimated_lambda']:.6f}")
        print(f"  KL Score: {kl_score:.6f}")
        print(f"  Sample Size: {len(self.latencies)}")
        print(f"  Total Packets: {self.total_packets}")

        
    def get_summary(self):
        """Get a summary of the monitoring results"""
        if not self.state_change_times:
            return "No congestion events detected."
            
        summary = f"Detected {len(self.state_change_times)} state changes:\n"
        for i, (timestamp, old_state, new_state) in enumerate(self.state_change_times):
            summary += f"  {i+1}. Time: {timestamp:.2f}s - {old_state} -> {new_state}\n"
            
        return summary
