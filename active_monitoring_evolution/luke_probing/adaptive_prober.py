import time
import numpy as np
import logging
from typing import Dict, Any, List
from collections import deque

from base_prober import BaseProber

def median_absolute_deviation(data):
    # Ensure data is a numpy array for median calculation
    data_array = np.array(data)
    if len(data_array) == 0:
        return 0.0 # Handle empty array case
    med = np.median(data_array)
    return np.median(np.abs(data_array - med))

class AdaptiveProber(BaseProber):
    def __init__(self, simulator, max_probes_per_second=10, sliding_window=10, confidence_level=0.95,
                 min_samples=5, congestion_z=1.5, outlier_z=2.5,
                 # New parameter for re-evaluation sensitivity
                 reevaluation_threshold_factor=0.75,
                 debug=True, log_filename="prober_debug.log"):

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

        # --- New state variables for baseline re-evaluation ---
        self.initial_baseline_median = None
        self.initial_baseline_mad = None
        self.baseline_re_evaluated = False
        self.sustained_normal_count = 0
        self.potential_new_baseline_delays = []
        self.reevaluation_threshold_factor = reevaluation_threshold_factor # How much lower new median must be
        # --- End new state variables ---


        #logging
        self.debug = debug
        if self.debug:
            logging.basicConfig(filename=log_filename,
                                filemode='w',  
                                level=logging.DEBUG,
                                format='%(asctime)s %(levelname)s: %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
            self.logger = logging.getLogger(__name__)
            header = (f"AdaptiveProber Initialized with parameters: "
                      f"max_probes_per_second={max_probes_per_second}, "
                      f"sliding_window={sliding_window}, min_samples={min_samples}, "
                      f"congestion_z={congestion_z}, outlier_z={outlier_z}, "
                      f"reevaluation_threshold_factor={reevaluation_threshold_factor}") # Added new param
            self.logger.info(header)


    def _do_probe(self):
        max_time = self.simulator.max_departure_time
        min_reeval_duration = max_time / 4 

        for second in range(int(max_time)):
            start_cpu_time = time.process_time()
            current_second_uncongested_delays = [] 
            raw_congestion = self._detect_congestion(second)

            # update congestion state.
            if raw_congestion and not self.is_congested:
                self.congestion_count += 1
                if self.congestion_count >= 3:
                    self.is_congested = True
                    self.congestion_count = 0
                    self.normal_count = 0
                    # Reset sustained normal tracking if we become congested
                    self.sustained_normal_count = 0
                    self.potential_new_baseline_delays = []
                    if self.debug:
                        self.logger.info(f"[Second {second}] Switching to CONGESTED state.")
            elif not raw_congestion and self.is_congested:
                self.normal_count += 1
                if self.normal_count >= 5:
                    self.is_congested = False
                    self.normal_count = 0
                    self.congestion_count = 0
                    # Start counting sustained normal seconds 
                    self.sustained_normal_count = 1
                    self.potential_new_baseline_delays = [] 
                    if self.debug:
                        self.logger.info(f"[Second {second}] Switching back to NORMAL state.")
            elif not raw_congestion: 
                self.congestion_count = 0
                if not self.baseline_re_evaluated and self.initial_baseline_median is not None:
                     self.sustained_normal_count += 1


            # adjusting probe rate.
            target_rate = self.congestion_probe_rate if self.is_congested else self.max_probes_per_second
            self.probe_rate_history.append(target_rate)
            self.current_probe_rate = int(np.mean(self.probe_rate_history))

            if self.debug:
                self.logger.info(f"[Second {second}] Congestion: {self.is_congested}, Probe Rate: {self.current_probe_rate}, Sustained Normal: {self.sustained_normal_count}")

            probes_sent = 0
            for i in range(self.current_probe_rate):
                t = second + (i / self.current_probe_rate)
                delay = self.send_probe(t, second) 
                probes_sent += 1

                # use valid delays only if not in congestion for uncongested_delays list
                if delay is not None and not self.is_congested:
                    self.uncongested_delays.append(delay)
                    # add to potential new baseline list if conditions met
                    if not self.baseline_re_evaluated and self.initial_baseline_median is not None:
                        current_second_uncongested_delays.append(delay)

                    # build baseline in early seconds (original logic)
                    if len(self.baseline_delays) < self.min_samples * 2 and self.initial_baseline_median is None:
                         self.baseline_delays.append(delay)

            # Add delays collected this second to the potential list
            if not self.baseline_re_evaluated and \
               self.initial_baseline_median is not None and \
               not self.is_congested:
                 self.potential_new_baseline_delays.extend(current_second_uncongested_delays)


            # --- Check for Baseline Re-evaluation ---
            if not self.baseline_re_evaluated and \
               self.initial_baseline_median is not None and \
               self.sustained_normal_count > min_reeval_duration and \
               len(self.potential_new_baseline_delays) >= self.min_samples:

                potential_median = np.median(self.potential_new_baseline_delays)

                # Check if potential new median is significantly lower than initial
                if potential_median < self.initial_baseline_median * self.reevaluation_threshold_factor:
                    if self.debug:
                        self.logger.warning(f"[Second {second}] *** BASELINE RE-EVALUATION TRIGGERED ***")
                        self.logger.warning(f"    Sustained normal: {self.sustained_normal_count}s > {min_reeval_duration:.1f}s.")
                        self.logger.warning(f"    Potential median {potential_median:.3f} < Initial median {self.initial_baseline_median:.3f} * {self.reevaluation_threshold_factor:.2f}.")
                        self.logger.warning(f"    Replacing baseline ({len(self.baseline_delays)} samples) with potential baseline ({len(self.potential_new_baseline_delays)} samples).")

                    # re-evaluation- replace baseline delays
                    self.baseline_delays = list(self.potential_new_baseline_delays) 

                    # mark as re-evaluated to prevent doing it again
                    self.baseline_re_evaluated = True

                    # clear the potential list as it's now the main baseline
                    self.potential_new_baseline_delays = []
                    self.sustained_normal_count = 0


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
            if self.debug:
                 self.logger.info(f"[Second {current_second}] Not enough recent samples ({len(recent_delays)}) for congestion check.")
            return False 
        
        # check if initial baseline needs to be established
        if self.initial_baseline_median is None and len(self.baseline_delays) >= self.min_samples * 2:
            self.initial_baseline_median = np.median(self.baseline_delays)
            self.initial_baseline_mad = median_absolute_deviation(self.baseline_delays)
            if self.initial_baseline_mad < 1e-4: self.initial_baseline_mad = 1e-4 # Floor MAD
            if self.debug:
                self.logger.info(f"[Second {current_second}] Initial baseline established: Median={self.initial_baseline_median:.3f}, MAD={self.initial_baseline_mad:.3f} using {len(self.baseline_delays)} samples.")
        elif self.initial_baseline_median is None:
             # still collecting initial baseline, cannot detect congestion yet
             if self.debug:
                  self.logger.info(f"[Second {current_second}] Still collecting initial baseline ({len(self.baseline_delays)}/{self.min_samples*2} samples).")
             return False


        # Use the current baseline_delays list (might be initial or re-evaluated)
        # Ensure enough samples in the current baseline list
        if len(self.baseline_delays) < self.min_samples:
             if self.debug:
                  self.logger.info(f"[Second {current_second}] Not enough samples ({len(self.baseline_delays)}) in current baseline list.")
             return False

        # Calculate current baseline stats for thresholding
        baseline_med = np.median(self.baseline_delays)
        baseline_mad = median_absolute_deviation(self.baseline_delays)
        if baseline_mad < 1e-4: baseline_mad = 1e-4 

        # --- Congestion Check ---
        threshold = baseline_med + self.congestion_z * baseline_mad
        recent_med = np.median(recent_delays)

        is_currently_congested = recent_med > threshold

        if self.debug:
            log_level = logging.WARNING if is_currently_congested else logging.INFO
            self.logger.log(log_level, f"[Second {current_second}] Congestion Check: Recent Median={recent_med:.3f} vs Threshold={threshold:.3f} "
                                     f"(Baseline Median={baseline_med:.3f}, MAD={baseline_mad:.3f}, z={self.congestion_z}). Result: {'Congested' if is_currently_congested else 'Normal'}")

        return is_currently_congested

    def _calculate_distribution_estimate(self):
        if not self.uncongested_delays:
            return 0.0, self.min_std

        # Use median/MAD of uncongested delays for outlier filtering
        uncongested_array = np.array(self.uncongested_delays)
        if len(uncongested_array) < self.min_samples:
             if len(uncongested_array) > 0:
                  mean = np.mean(uncongested_array)
                  std = max(np.std(uncongested_array), self.min_std)
                  if self.debug:
                       self.logger.info(f"Calculating estimate with few uncongested samples ({len(uncongested_array)}): mean={mean:.3f}, std={std:.3f}")
                  return mean, std
             else:
                  return 0.0, self.min_std


        med = np.median(uncongested_array)
        mad_val = median_absolute_deviation(uncongested_array)
        if mad_val < 1e-4: mad_val = 1e-4 # Floor MAD

        lower_bound = med - self.outlier_z * mad_val
        upper_bound = med + self.outlier_z * mad_val

        # Filter based on bounds
        filtered_delays = uncongested_array[(uncongested_array >= lower_bound) & (uncongested_array <= upper_bound)]

        if self.debug:
            self.logger.info(f"Estimate Filter: Uncongested median={med:.3f}, MAD={mad_val:.3f}, "
                             f"bounds=({lower_bound:.3f}, {upper_bound:.3f}), "
                             f"using {len(filtered_delays)} of {len(uncongested_array)} uncongested delays.")

        # decide whether to use filtered or all uncongested delays
        # use filtered if it keeps >= 70% of data
        if len(filtered_delays) >= len(uncongested_array) * 0.7 and len(filtered_delays) > 0:
            delays_to_use = filtered_delays
        elif len(uncongested_array) > 0:
             delays_to_use = uncongested_array 
        else:
             return 0.0, self.min_std 


        mean = np.mean(delays_to_use)
        std = max(np.std(delays_to_use), self.min_std)

        return mean, std

    def get_metrics(self) -> Dict[str, Any]:
        metrics = super().get_metrics()
        metrics['baseline_re_evaluated'] = self.baseline_re_evaluated
        if self.initial_baseline_median is not None:
             metrics['initial_baseline_median'] = self.initial_baseline_median
             metrics['initial_baseline_mad'] = self.initial_baseline_mad
        return metrics
