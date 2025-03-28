import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class BaseProber:    
    def __init__(self, simulator, max_probes_per_second=5):
        self.simulator = simulator
        self.max_probes_per_second = max_probes_per_second

        # probes to calculate distribution
        self.probe_history = [] 
        
        # (time_slot, estimated_mean, estimated_std, probes_sent, cpu_time)
        self.metrics_per_timeslot = [] 
        
    # public method
    def probe(self):
        self._do_probe()
        
    # just send maxi probes per second.
    def _do_probe(self):
        max_time = self.simulator.max_departure_time

        # i decide not to probe in second 100-- adding the offest triggers
        # ValueError: ('Time must be in [0, 100.0] sec.', 100.2)
        for second in range(int(max_time)):
            probes_sent = 0
            # measure 'compute' used in this timeslot
            start_cpu_time = time.process_time()
            
            for i in range(self.max_probes_per_second):
                #  avoid caching from exact integer times
                t = second + (i / self.max_probes_per_second)
                self.send_probe(t, second)
                probes_sent += 1
            
            # update distribution estimate
            new_mean, new_std = self._update_distribution_estimate(second, probes_sent, start_cpu_time)

            end_cpu_time = time.process_time()
            cpu_time = end_cpu_time - start_cpu_time

            # these time slots have metrics
            self.metrics_per_timeslot.append((
                second, 
                new_mean, 
                new_std, 
                probes_sent,
                cpu_time
            ))
    
    def send_probe(self, t, time_slot):
        delay = self.simulator.send_probe_at(t)
        self.probe_history.append((t, delay))
        return delay
    
    def _update_distribution_estimate(self, time_slot, probes_sent, start_cpu_time):
        delays = [d for _, d in self.probe_history if d is not None]
        if delays:
            estimated_mean = np.mean(delays)
            estimated_std = np.std(delays)
        return estimated_mean, estimated_std
            
          
    # get metrics for plotting
    def get_metrics(self) -> Dict[str, Any]:
        time_slots = [m[0] for m in self.metrics_per_timeslot]
        cpu_times = {ts: m[4] for ts, m in zip(time_slots, self.metrics_per_timeslot)}
        probes_per_ts = {ts: m[3] for ts, m in zip(time_slots, self.metrics_per_timeslot)}
        
        return {
            "probes_sent": sum(m[3] for m in self.metrics_per_timeslot),
            "cpu_times": cpu_times,
            "probes_per_timeslot": probes_per_ts,
            "metrics_per_timeslot": self.metrics_per_timeslot
        }
    
  