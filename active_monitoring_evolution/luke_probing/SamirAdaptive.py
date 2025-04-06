import time
import numpy as np
import statistics
from collections import deque
from base_prober import BaseProber
from typing import Dict, Any

class AdaptiveActiveProber(BaseProber):
    def __init__(self, simulator, max_probes_per_second=10, time_limit=100,
                 min_rate=0.5, drop_threshold=0.20, spike_threshold=1.5):
        super().__init__(simulator, max_probes_per_second)
        self.time_limit = time_limit
        self.max_rate = max_probes_per_second
        self.min_rate = min_rate
        self.current_rate = max_probes_per_second
        
        self.drop_threshold = drop_threshold
        self.spike_threshold = spike_threshold
        self.baseline_delays = []
        
        self.in_congestion = False
        self.last_delays = deque(maxlen=5)
        self.last_drops = deque(maxlen=5)
        self.baseline_delay = None
    
    def update_baseline(self):
        if len(self.last_delays) < 3 or len(self.last_drops) < 3:
            return
            
        drop_rate = sum(self.last_drops) / len(self.last_drops)
        delay_variance = statistics.variance(self.last_delays)
        
        if drop_rate < 0.05 and delay_variance < 0.1:
            self.baseline_delay = statistics.mean(self.last_delays)
            self.baseline_delays.append(self.baseline_delay)
    
    def check_congestion(self, result):
        if result is None:
            self.last_drops.append(1)
        else:
            self.last_drops.append(0)
            if not self.in_congestion:
                self.last_delays.append(result)
                
        if self.baseline_delay:
            if result and result > self.spike_threshold * statistics.median(self.baseline_delays):
                return "congestion"
                
        if len(self.last_drops) >= 3:
            drop_rate = sum(self.last_drops) / len(self.last_drops)
            if drop_rate > self.drop_threshold:
                return "congestion"
                
        self.update_baseline()
        return "stable"
    
    def adjust_rate(self, status):
        if status == "congestion":
            self.in_congestion = True
            self.current_rate = self.min_rate
        else:
            self.current_rate = self.max_rate
            self.in_congestion = False
    
    def _do_probe(self):
        current_time = 0.0
        probes_in_current_second = 0
        last_probe_time = 0
        
        while current_time < self.time_limit:
            # Measure CPU time for metrics
            start_cpu_time = time.process_time()
            
            if probes_in_current_second < self.max_rate:
                # Send probe and check congestion
                result = self.send_probe(current_time, int(current_time))
                status = self.check_congestion(result)
                self.adjust_rate(status)
                
                probes_in_current_second += 1
                last_probe_time = current_time
                
                # Update distribution estimate for the current second
                new_mean, new_std = self._update_distribution_estimate(
                    int(current_time), probes_in_current_second, start_cpu_time
                )
                
                # Record metrics for this time slot
                end_cpu_time = time.process_time()
                cpu_time = end_cpu_time - start_cpu_time
                
                # Add metrics if this is a new second
                time_slot = int(current_time)
                self.metrics_per_timeslot.append((
                    time_slot,
                    new_mean,
                    new_std,
                    probes_in_current_second,
                    cpu_time
                ))
            
            # Adjust time based on current rate and congestion state
            if current_time + (1 / self.current_rate) > int(last_probe_time) + 1:
                if self.in_congestion:
                    current_time += 1.0 / self.current_rate
                else:
                    current_time = int(last_probe_time) + 1
                probes_in_current_second = 0
            else:
                current_time += 1 / self.current_rate
    
    def _update_distribution_estimate(self, time_slot, probes_sent, start_cpu_time):
        # Only use valid (non-None) delay values
        delays = [d for _, d in self.probe_history if d is not None]
        if delays:
            estimated_mean = np.mean(delays)
            estimated_std = np.std(delays)
            return estimated_mean, estimated_std
        return 0, 0

    def get_metrics(self) -> Dict[str, Any]:
        time_slots = [m[0] for m in self.metrics_per_timeslot]
        cpu_times = {ts: m[4] for ts, m in zip(time_slots, self.metrics_per_timeslot)}
        probes_per_ts = {ts: m[3] for ts, m in zip(time_slots, self.metrics_per_timeslot)}
        
        return {
            "probes_sent": sum(m[3] for m in self.metrics_per_timeslot),
            "cpu_times": cpu_times,
            "probes_per_timeslot": probes_per_ts,
            "metrics_per_timeslot": self.metrics_per_timeslot,
            "total_probes": sum(m[3] for m in self.metrics_per_timeslot)
        }