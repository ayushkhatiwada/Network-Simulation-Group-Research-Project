# kl_divergence_prober.py
import time
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from collections import deque

from base_prober import BaseProber # Assuming base_prober.py is in the same directory

# Helper function to calculate KL divergence between two normal distributions
def kl_divergence_normal(mu1: float, std1: float, mu2: float, std2: float) -> float:
    """Calculates KL(N(mu1, std1^2) || N(mu2, std2^2))."""
    if std1 <= 0 or std2 <= 0:
        # Avoid division by zero or log(0)
        return float('inf')
    
    # Ensure variance is used (std^2)
    var1 = std1**2
    var2 = std2**2
    
    # KL divergence formula
    kl = np.log(std2 / std1) + (var1 + (mu1 - mu2)**2) / (2 * var2) - 0.5
    return kl

class KLDivergenceProber(BaseProber):
    def __init__(self, simulator, max_probes_per_second=5, 
                 kl_threshold: float = 0.5, 
                 window_size: int = 15, 
                 min_samples: int = 10, 
                 baseline_window_size: int = 50,
                 congestion_probe_rate_factor: float = 0.2,
                 congestion_confirm: int = 3, # Consecutive signals to confirm congestion
                 normal_confirm: int = 5,     # Consecutive signals to confirm normal
                 min_std: float = 0.01,       # Minimum std dev to avoid numerical issues
                 debug: bool = True, 
                 log_filename: str = "kl_prober_debug.log"):

        super().__init__(simulator, max_probes_per_second)
        
        # KL Divergence parameters
        self.kl_threshold = kl_threshold
        self.window_size = window_size
        self.min_samples = max(min_samples, 2) # Need at least 2 samples for std dev
        self.baseline_window_size = baseline_window_size
        self.min_std = min_std

        # State variables
        self.baseline_delays = []
        self.baseline_mean: Optional[float] = None
        self.baseline_std: Optional[float] = None
        self.recent_delays = deque(maxlen=self.window_size)
        
        self.is_congested = False
        self.congestion_signals = 0 # Counter for consecutive congestion signals
        self.normal_signals = 0     # Counter for consecutive normal signals
        self.congestion_confirm = congestion_confirm
        self.normal_confirm = normal_confirm

        # Probe rate adjustment
        self.congestion_probe_rate = max(1, int(self.max_probes_per_second * congestion_probe_rate_factor))
        self.current_probe_rate = self.max_probes_per_second
        self.probe_rate_history = deque([self.max_probes_per_second] * 3, maxlen=3) # For smoothing

        # Logging setup
        self.debug = debug
        if self.debug:
            # Avoid adding multiple handlers if logger already exists
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                logging.basicConfig(filename=log_filename,
                                    filemode='w',  
                                    level=logging.DEBUG,
                                    format='%(asctime)s %(levelname)s: %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
                # Log initialization parameters
                header = (f"KLDivergenceProber Initialized: "
                          f"max_probes={max_probes_per_second}, kl_thresh={kl_threshold}, "
                          f"window={window_size}, min_samples={min_samples}, "
                          f"baseline_size={baseline_window_size}, "
                          f"cong_factor={congestion_probe_rate_factor}")
                self.logger.info(header)

    def _estimate_normal_params(self, delays: List[float]) -> Optional[Tuple[float, float]]:
        """Estimates mean and std deviation from a list of delays."""
        if len(delays) < self.min_samples:
            return None
        mean = np.mean(delays)
        std = max(np.std(delays), self.min_std) # Ensure minimum std dev
        return mean, std

    def _update_baseline(self, delay: float):
        """Adds a delay to the baseline pool if not yet full."""
        if self.baseline_mean is None and len(self.baseline_delays) < self.baseline_window_size:
            self.baseline_delays.append(delay)
            # Once baseline window is full, calculate its parameters
            if len(self.baseline_delays) == self.baseline_window_size:
                params = self._estimate_normal_params(self.baseline_delays)
                if params:
                    self.baseline_mean, self.baseline_std = params
                    if self.debug:
                        self.logger.info(f"Baseline established: Mean={self.baseline_mean:.3f}, Std={self.baseline_std:.3f}")
                else:
                     # Should not happen if baseline_window_size >= min_samples
                     if self.debug: self.logger.warning("Could not establish baseline parameters.")


    def _detect_change(self, current_second: int) -> bool:
        """Detects distribution change using KL divergence."""
        if self.baseline_mean is None or self.baseline_std is None:
            if self.debug: self.logger.debug(f"[Second {current_second}] Baseline not ready.")
            return False # Baseline not established yet

        valid_recent_delays = list(self.recent_delays)
        if len(valid_recent_delays) < self.min_samples:
            if self.debug: self.logger.debug(f"[Second {current_second}] Not enough recent samples ({len(valid_recent_delays)} < {self.min_samples}).")
            return False # Not enough recent data

        recent_params = self._estimate_normal_params(valid_recent_delays)
        if recent_params is None:
             if self.debug: self.logger.warning(f"[Second {current_second}] Could not estimate recent parameters.")
             return False

        recent_mean, recent_std = recent_params
        
        # Calculate KL divergence: KL(Recent || Baseline)
        # Measures how much information is lost when approximating Recent with Baseline.
        # A high value means Recent is different from Baseline.
        kl_div = kl_divergence_normal(recent_mean, recent_std, self.baseline_mean, self.baseline_std)

        is_diverging = kl_div > self.kl_threshold

        if self.debug:
            self.logger.info(f"[Second {current_second}] Recent Mean={recent_mean:.3f}, Std={recent_std:.3f}. "
                             f"KL(Recent||Baseline)={kl_div:.4f}. Threshold={self.kl_threshold}. Diverging={is_diverging}")

        return is_diverging

    def _do_probe(self):
        max_time = self.simulator.max_departure_time
        
        for second in range(int(max_time)):
            start_cpu_time = time.process_time()
            
            # 1. Detect potential change based on KL divergence
            is_diverging = self._detect_change(second)

            # 2. Update congestion state using hysteresis
            if is_diverging:
                self.congestion_signals += 1
                self.normal_signals = 0 # Reset normal counter
                if self.congestion_signals >= self.congestion_confirm and not self.is_congested:
                    self.is_congested = True
                    if self.debug:
                        self.logger.info(f"[Second {second}] Confirmed CONGESTED state.")
            else:
                self.normal_signals += 1
                self.congestion_signals = 0 # Reset congestion counter
                if self.normal_signals >= self.normal_confirm and self.is_congested:
                    self.is_congested = False
                    if self.debug:
                        self.logger.info(f"[Second {second}] Confirmed NORMAL state.")
            
            # Reset counters if state didn't change but signal flipped
            if self.is_congested and not is_diverging: self.congestion_signals = 0
            if not self.is_congested and is_diverging: self.normal_signals = 0


            # 3. Adjust probe rate (with smoothing)
            target_rate = self.congestion_probe_rate if self.is_congested else self.max_probes_per_second
            self.probe_rate_history.append(target_rate)
            self.current_probe_rate = int(np.mean(list(self.probe_rate_history))) # Use mean of history for smoothing
            
            if self.debug:
                 self.logger.debug(f"[Second {second}] State: {'Congested' if self.is_congested else 'Normal'}, "
                                  f"Target Rate: {target_rate}, Current Rate: {self.current_probe_rate}")
            
            # 4. Send probes for this second
            probes_sent_this_second = 0
            delays_this_second = []
            for i in range(self.current_probe_rate):
                # Avoid exact integer times if simulator caches
                t = second + (i / max(1, self.current_probe_rate)) 
                try:
                    delay = self.send_probe(t, second) # send_probe adds to self.probe_history
                    probes_sent_this_second += 1
                    if delay is not None:
                        delays_this_second.append(delay)
                        self.recent_delays.append(delay) # Add to sliding window
                        # Add to baseline if it's not established yet
                        if self.baseline_mean is None:
                             self._update_baseline(delay)
                except ValueError as e:
                     if self.debug: self.logger.error(f"Error sending probe at t={t}: {e}")
                     # Handle cases where time might exceed simulator limits slightly due to float precision
                     break 

            # 5. Update distribution estimate for metrics (using base class method)
            # This uses the *entire* probe history up to this point.
            # You might want to override this to use only baseline or recent non-congested data.
            estimated_mean, estimated_std = self._update_distribution_estimate(second, probes_sent_this_second, start_cpu_time)

            end_cpu_time = time.process_time()
            cpu_time = end_cpu_time - start_cpu_time

            # 6. Record metrics for this timeslot
            self.metrics_per_timeslot.append((
                second,
                estimated_mean if estimated_mean is not None else 0.0, # Handle potential None from base method if no history
                estimated_std if estimated_std is not None else self.min_std,
                probes_sent_this_second,
                cpu_time
            ))

    # Inherit get_metrics from BaseProber
    def get_metrics(self) -> Dict[str, Any]:
        metrics = super().get_metrics()
        # Add any specific metrics if needed, e.g., KL divergence values per timeslot
        # metrics["kl_values"] = self.kl_history # (Requires storing KL values)
        return metrics
