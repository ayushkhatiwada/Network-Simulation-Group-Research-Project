import time
import numpy as np
import logging
from typing import Dict, Any, List
from collections import deque

from base_prober import BaseProber

def median_absolute_deviation(data):
    med = np.median(data)
    return np.median(np.abs(np.array(data) - med))

class AdaptiveProber(BaseProber):
    def __init__(self, simulator, max_probes_per_second=5, sliding_window=10, confidence_level=0.95, 
                 min_samples=5, congestion_z=1.5, outlier_z=2.5, debug=True, log_filename="prober_debug.log"):

        super().__init__(simulator, max_probes_per_second)
        
        self.sliding_window = sliding_window
        self.confidence_level = confidence_level
        self.min_samples = min_samples
        
        self.baseline_delays = []
        self.uncongested_delays = []
        
        self.is_congested = False
        self.congestion_count = 0
        self.normal_count = 0
        
        # When congested, probe at a lower rate.
        self.congestion_probe_rate = max(1, int(self.max_probes_per_second * 0.2))
        self.current_probe_rate = self.max_probes_per_second
        
        # Smoothing the probe rate transitions.
        self.probe_rate_history = deque([self.max_probes_per_second] * 3, maxlen=3)
        
        # prevent numerical issues
        self.min_std = 0.01
        
        # adaptive thresholds
        self.congestion_z = congestion_z
        self.outlier_z = outlier_z

        #logging 
        self.debug = debug
        if self.debug:
            logging.basicConfig(filename=log_filename,
                                filemode='w',  
                                level=logging.DEBUG,
                                format='%(asctime)s %(levelname)s: %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
            self.logger = logging.getLogger(__name__)

            # logging headers
            header = (f"AdaptiveProber Initialized with parameters: "
                      f"max_probes_per_second={max_probes_per_second}, "
                      f"sliding_window={sliding_window}, min_samples={min_samples}, "
                      f"congestion_z={congestion_z}, outlier_z={outlier_z}")
            self.logger.info(header)

    def _do_probe(self):
        max_time = self.simulator.max_departure_time
        
        for second in range(int(max_time)):
            start_cpu_time = time.process_time()
            
            # Detect congestion using recent probe history.
            raw_congestion = self._detect_congestion(second)
            
            # update congestion state.
            if raw_congestion and not self.is_congested:
                self.congestion_count += 1
                if self.congestion_count >= 3:
                    self.is_congested = True
                    self.congestion_count = 0
                    self.normal_count = 0
                    if self.debug:
                        self.logger.info(f"[Second {second}] Switching to CONGESTED state.")
            elif not raw_congestion and self.is_congested:
                self.normal_count += 1
                if self.normal_count >= 5:
                    self.is_congested = False
                    self.normal_count = 0
                    self.congestion_count = 0
                    if self.debug:
                        self.logger.info(f"[Second {second}] Switching back to NORMAL state.")
            elif not raw_congestion:
                self.congestion_count = 0

            # adjusting probe rate.
            target_rate = self.congestion_probe_rate if self.is_congested else self.max_probes_per_second
            self.probe_rate_history.append(target_rate)
            self.current_probe_rate = int(np.mean(self.probe_rate_history))
            
            if self.debug:
                self.logger.info(f"[Second {second}] Congestion: {self.is_congested}, Probe Rate: {self.current_probe_rate}")
            
            probes_sent = 0
            for i in range(self.current_probe_rate):
                t = second + (i / self.current_probe_rate)
                delay = self.send_probe(t, second)
                probes_sent += 1
                
                # use valid delays only if not in congestion.
                if delay is not None and not self.is_congested:
                    # build baseline in early seconds.
                    if len(self.baseline_delays) < self.min_samples * 2 and second < 10:
                        self.baseline_delays.append(delay)
                        if self.debug:
                            self.logger.info(f"[Second {second}] Added {delay:.3f} to baseline_delays.")
                    self.uncongested_delays.append(delay)
                    if self.debug:
                        self.logger.info(f"[Second {second}] Added {delay:.3f} to uncongested_delays.")
            
            # Calculate the uncongested delay estimates.
            new_mean, new_std = self._calculate_distribution_estimate()
            
            end_cpu_time = time.process_time()
            cpu_time = end_cpu_time - start_cpu_time
            
            self.metrics_per_timeslot.append((
                second,
                new_mean,
                new_std,
                probes_sent,
                cpu_time
            ))
    
    def _detect_congestion(self, current_second):
        recent_delays = [d for t, d in self.probe_history[-self.sliding_window:] if d is not None]
        
        if len(recent_delays) < self.min_samples:
            return False
        
        # build the baseline during the early phase.
        if len(self.baseline_delays) < self.min_samples * 2 and current_second < 10:
            self.baseline_delays.extend(recent_delays)
            if self.debug:
                self.logger.info(f"[Second {current_second}] Building baseline: added {len(recent_delays)} delays.")
            return False
        
        if len(self.baseline_delays) < self.min_samples:
            return False
        
        baseline_med = np.median(self.baseline_delays)
        baseline_mad = median_absolute_deviation(self.baseline_delays)
        if baseline_mad < 1e-4:
            baseline_mad = 1e-4
            
        threshold = baseline_med + self.congestion_z * baseline_mad
        recent_med = np.median(recent_delays)
        
        if self.debug:
            self.logger.info(f"[Second {current_second}] Baseline median: {baseline_med:.3f}, MAD: {baseline_mad:.3f}, "
                             f"Threshold: {threshold:.3f}, Recent median: {recent_med:.3f}")
        
        return recent_med > threshold
    
    def _calculate_distribution_estimate(self):
        if not self.uncongested_delays:
            return 0.0, self.min_std
        
        med = np.median(self.uncongested_delays)
        mad_val = median_absolute_deviation(self.uncongested_delays)
        if mad_val < 1e-4:
            mad_val = 1e-4
            
        lower_bound = med - self.outlier_z * mad_val
        upper_bound = med + self.outlier_z * mad_val
        
        filtered_delays = [d for d in self.uncongested_delays if lower_bound <= d <= upper_bound]
        
        if self.debug:
            self.logger.info(f"Uncongested delays: median={med:.3f}, MAD={mad_val:.3f}, "
                             f"bounds=({lower_bound:.3f}, {upper_bound:.3f}), "
                             f"using {len(filtered_delays)} of {len(self.uncongested_delays)} delays.")
        
        if len(filtered_delays) >= len(self.uncongested_delays) * 0.7:
            delays_to_use = filtered_delays
        else:
            delays_to_use = self.uncongested_delays
        
        mean = np.mean(delays_to_use)
        std = max(np.std(delays_to_use), self.min_std)
        
        return mean, std
    
    def get_metrics(self) -> Dict[str, Any]:
        metrics = super().get_metrics()
        return metrics
