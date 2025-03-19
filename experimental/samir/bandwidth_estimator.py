from active_interface_bandwidth_packetloss import Simulator
import numpy as np
from scipy.stats import gaussian_kde
import math

class BandwidthEstimator:
    def __init__(self):
        self.simulator = Simulator()
        self.initial_step = 1.0  # Initial time step between probes (spaced out)
        self.step_multiplier = 0.95  # Multiplier to decrease the step size
        self.baseline_delay = None  # Baseline delay (most frequent delay)
        self.congestion_threshold = 1.5  # Multiplier to detect congestion (delays > baseline * threshold)
        self.consecutive_over_threshold = 5  # Number of consecutive packets exceeding threshold to stop
        self.max_penalty = None  # Maximum penalty (will be determined when delays plateau)
        self.plateau_threshold = 0.01  # Threshold to detect plateau in delays


    def estimate_baseline_delay(self, num_probes=20):
        """
        Estimate baseline delay by sending spaced-out packets and finding the most frequent delay.
        """
        departure_times = []
        time = 0
        step = self.initial_step

        for _ in range(num_probes):
            departure_times.append(time)
            time += step

        results = self.simulator.send_multiple_probes(departure_times)

        # Filter valid results (where arrival time is not None)
        valid_results = [(d, a, delay) for d, a, delay in results if a is not None]
        if not valid_results:
            raise ValueError("No valid results to estimate baseline delay.")

        # Extract delays
        delays = [delay for _, _, delay in valid_results]
        print(delays)

        # Use Kernel Density Estimation (KDE) to find the most frequent delay
        kde = gaussian_kde(delays)
        delay_range = np.linspace(min(delays), max(delays), 1000)  # Create a range of delay values
        densities = kde(delay_range)  # Compute density at each point in the range
        self.baseline_delay = delay_range[np.argmax(densities)]  # Find the delay with the highest density

        return self.baseline_delay

    def estimate_bandwidth(self):
        """
        Estimate bandwidth by increasing probe rate until delays consistently exceed baseline.
        """
        if self.baseline_delay is None:
            raise ValueError("Baseline delay must be estimated first.")

        time = 100000
        step = self.initial_step
        consecutive_over = 0  # Counter for consecutive packets exceeding threshold
        delays = []

        while True:
            # Send a probe and record departure time
            time += step
            step *= self.step_multiplier  # Decrease step size to increase probe rate

            # Send probes and get results
            result = self.simulator.send_probe_at(time)
            if result[1] is not None:  # arrival_time is not None
                delays.append(result[2])

            if delays != []:
                # Check if the latest delay exceeds the congestion threshold
                latest_delay = delays[-1]
                if latest_delay > self.baseline_delay * self.congestion_threshold:
                    consecutive_over += 1
                else:
                    consecutive_over = 0

                # Stop if delays consistently exceed the threshold
                if consecutive_over >= self.consecutive_over_threshold:
                    estimated_bandwidth = len(delays) - consecutive_over
                    break
        # Second while loop: Continue sending packets until delays plateau
        plateau_delays = []
        while True:
            # Send a probe and record departure time
            time += step
            step *= self.step_multiplier  # Decrease step size to increase probe rate

            # Send probes and get results
            result = self.simulator.send_probe_at(time)
            if result[1] is not None:  # arrival_time is not None
                plateau_delays.append(result[2])

            # Check if delays have plateaued
            if len(plateau_delays) >= 10:  # Need at least 10 samples to check for plateau
                recent_delays = plateau_delays[-10:]
                variance = np.var(recent_delays)
                if variance < self.plateau_threshold:
                    self.max_penalty = np.mean(recent_delays) - baseline_delay  # Plateau reached
                    break
        
        # Estimate k using excess delays and plateau delays
        excess_delays = delays[-self.consecutive_over_threshold:]  # Delays of the excess packets
        k_estimates = []
        for delay in excess_delays:
            penalty = delay - self.baseline_delay
            if penalty > 0:
                k = - (1 / self.consecutive_over_threshold) * math.log(1 - (penalty / self.max_penalty))
                k_estimates.append(k)

        # Average k estimates to account for randomness
        k_avg = np.mean(k_estimates) if k_estimates else 0.0

        return {
            "estimated_bandwidth": estimated_bandwidth,
            "baseline_delay": self.baseline_delay,
            "loss rate k": k_avg,
        }

# Create an instance of the BandwidthEstimator
estimator = BandwidthEstimator()

# Phase 1: Estimate baseline delay
baseline_delay = estimator.estimate_baseline_delay()
print(f"Baseline delay (most frequent): {baseline_delay}")

# Phase 2: Estimate bandwidth
result = estimator.estimate_bandwidth()
print(result)