import math
from ground_truth import GroundTruthNetwork


class ActiveSimulator_v0:
    def __init__(self) -> None:
        """
        Initializes the simulator with a network model.
        """
        self.network = GroundTruthNetwork()
        self.event_log = []                 # Stores tuples: (departure_time, arrival_time, delay)
        self.time_cache = {}                # Caches computed delays for given departure_times
        self.max_departure_time = 100.0     # Allowed departure times: [0, 100] seconds
        self.max_probes_per_second = 10     # Maximum number of probes that can be sent per second
        self.probe_count_per_second = {}    # Tracks number of probes sent for each second

    def measure_end_to_end_delay(self) -> float:
        """
        Measures the delay from SOURCE to DESTINATION.
        
        :return: The delay as a float.
        """
        return self.network.sample_edge_delay(self.network.SOURCE, self.network.DESTINATION)

    def send_probe_at(self, departure_time: float) -> float:
        """
        Sends a probe at a specific departure time.
        The departure time must be within the allowed window [0, 100] seconds,
        and the number of probes per second cannot exceed the fixed max_probes_per_second.
        
        :param departure_time: The time when the probe is sent.
        :return: The delay measured for the probe.
        """
        if departure_time < 0 or departure_time > self.max_departure_time:
            raise ValueError(f"Departure time must be between 0 and {self.max_departure_time} seconds.")

        # Rate limiting: only allow max_probes_per_second in each second
        time_slot = int(departure_time)
        if self.probe_count_per_second.get(time_slot, 0) >= self.max_probes_per_second:
            raise Exception(f"Probe rate limit exceeded for second {time_slot}. "
                            f"Max {self.max_probes_per_second} probe per second allowed.")

        # Increment rate counter for this time slot
        self.probe_count_per_second[time_slot] = self.probe_count_per_second.get(time_slot, 0) + 1

        # Use cached delay if available
        if departure_time in self.time_cache:
            delay = self.time_cache[departure_time]
        else:
            delay = self.measure_end_to_end_delay()
            self.time_cache[departure_time] = delay

        arrival_time = departure_time + delay
        self.event_log.append((departure_time, arrival_time, delay))
        return delay

    def send_multiple_probes(self, departure_times: list[float]) -> list[float]:
        """
        Sends multiple probes at the specified departure times.

        :param departure_times: A list of times at which probes should be sent.
        :return: A list of delays measured for each probe.
        """
        delays = []
        for time_point in sorted(departure_times):
            delays.append(self.send_probe_at(time_point))
        return delays

    def get_event_log(self) -> list[tuple[float, float, float]]:
        """
        Returns the list of past probe events in the form (departure_time, arrival_time, delay).
        """
        return sorted(self.event_log, key=lambda event: event[0])

    def compare_distribution_parameters(self, pred_mean: float, pred_std: float) -> float:
        """
        Compares the predicted delay distribution parameters with the actual network parameters using KL divergence.
        """
        params = self.network.get_distribution_parameters(self.network.SOURCE, self.network.DESTINATION)
        actual_mean = params["mean"]
        actual_std = params["std"]

        kl_div = math.log(pred_std / actual_std) + ((actual_std**2 + (actual_mean - pred_mean)**2) / (2 * pred_std**2)) - 0.5
        
        # Aim for a KL divergence of <= 0.05
        if kl_div <= 0.05:
            print(f"KL divergence: {kl_div:.4f} ✅")
        else:
            print(f"KL divergence: {kl_div:.4f} ❌")
        return kl_div
