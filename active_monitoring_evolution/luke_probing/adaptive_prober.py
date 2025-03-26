import time
import numpy as np
from scipy import stats
from typing import Dict, Any

from base_prober import BaseProber

class AdaptiveProber(BaseProber):
    def __init__(self, simulator, max_probes_per_second=5, sliding_window=10, confidence_level=0.95, min_samples=5, learning_rate=0.1):
        super().__init__(simulator, max_probes_per_second)
        
        self.sliding_window = sliding_window
        self.confidence_level = confidence_level
        self.min_samples = min_samples
        self.learning_rate = learning_rate
        
        self.baseline_delays = []
        self.congestion_detection_results = {}
        self.reference_stats = None
        self.congestion_multiplier = None
        self.known_congestion_ratios = []
        
    def _do_probe(self):
        max_time = self.simulator.max_departure_time
        
        for second in range(int(max_time)):
            start_cpu_time = time.process_time()
            probes_sent = 0
            
            is_congested = self._detect_congestion(second)            
            # Calculate dynamic probe rate - reduce probing during congestion
            if is_congested:
                probe_rate = max(1, int(self.max_probes_per_second * 0.5)) 
            else:
                probe_rate = self.max_probes_per_second
            
            # Send probes
            for i in range(probe_rate):
                t = second + (i / probe_rate)
                self.send_probe(t, second)
                probes_sent += 1

            # Ensure all values are initialized
            new_mean, new_std = self._update_distribution_estimate(second, probes_sent, start_cpu_time)
            if new_mean is None:
                new_mean = 0.0
            if new_std is None:
                new_std = 0.0

            # Update metrics
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
        # Get recent delays
        recent_delays = [d for t, d in self.probe_history[-self.sliding_window:] if d is not None]
        

        if len(recent_delays) < self.min_samples:
            self.congestion_detection_results[current_second] = {
                'detected': False,
                'reason': f'insufficient_samples ({len(recent_delays)})',
                'p_value': None
            }
            return False
            
        # If we don't have baseline stats yet, establish them from first few samples
        if not self.reference_stats and len(self.baseline_delays) < self.min_samples:
            if current_second < 5: 
                self.baseline_delays.extend(recent_delays)
            self.congestion_detection_results[current_second] = {
                'detected': False,
                'reason': f'building_baseline ({len(self.baseline_delays)})',
                'p_value': None
            }
            return False
            
        # Establish baseline if we have enough samples
        if not self.reference_stats and len(self.baseline_delays) >= self.min_samples:
            self.reference_stats = {
                'mean': np.mean(self.baseline_delays),
                'std': np.std(self.baseline_delays)
            }
        
        # Mann-Whitney U test to detect distribution shift
        statistic, p_value = stats.mannwhitneyu(
            self.baseline_delays[-self.min_samples:],
            recent_delays,
            alternative='less'
        )
        
        # Calculate current ratio
        baseline_median = np.median(self.baseline_delays[-self.min_samples:])
        recent_median = np.median(recent_delays)
        current_ratio = recent_median / baseline_median if baseline_median > 0 else 1.0
        
        # Update congestion multiplier if we detect a significant shift
        if p_value < (1 - self.confidence_level) and current_ratio > 1.2:
            self.known_congestion_ratios.append(current_ratio)
            if not self.congestion_multiplier:
                self.congestion_multiplier = current_ratio
            else:
                self.congestion_multiplier = (
                    (1 - self.learning_rate) * self.congestion_multiplier + 
                    self.learning_rate * current_ratio
                )
        
        if self.congestion_multiplier:
            margin = 0.2  # 20% margin around learned multiplier
            is_congested = (
                p_value < (1 - self.confidence_level) and 
                (self.congestion_multiplier * (1 - margin) <= current_ratio <= 
                 self.congestion_multiplier * (1 + margin))
            )
        else:
            is_congested = False
        
        multiplier_str = f"{self.congestion_multiplier:.2f}" if self.congestion_multiplier else "0"
        self.congestion_detection_results[current_second] = {
            'detected': is_congested,
            'reason': (
                f'ratio={current_ratio:.2f}, '
                f'multiplier={multiplier_str}, '
                f'p_value={p_value:.3f}'
            ),
            'p_value': p_value
        }
        
        return is_congested

    # Override get_metrics to include skipped probes
    def get_metrics(self) -> Dict[str, Any]:
        metrics = super().get_metrics()
        
        return metrics