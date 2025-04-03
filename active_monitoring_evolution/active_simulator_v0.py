import math
import random

from ground_truth import GroundTruthNetwork


class ActiveSimulator_v0:
    def __init__(self, paths="1", random_seed=42, simulation_duration=100) -> None:
        random.seed(random_seed)
        self.network = GroundTruthNetwork(paths)
        self.paths = paths
        self.event_log = []
        self.time_cache = {}
        self.max_departure_time = simulation_duration
        self.max_probes_per_second = 10
        self.probe_count_per_second = {}

    def measure_end_to_end_delay(self) -> float:
        """
        Measures the delay from SOURCE to DESTINATION.
        
        :return: The delay as a float.
        """
        return self.network.sample_edge_delay(self.network.SOURCE, self.network.DESTINATION)

    def send_probe_at(self, departure_time: float) -> float:
        """
        Sends a probe at a specific departure time.
        Departure time must be within allowed window of [0, 100] seconds,
        Number of probes per second cannot exceed max_probes_per_second.
        
        :param departure_time: The time when the probe is sent.
        :return: The delay measured for the probe.
        """

        # Only allow packets to be sent within [0, 100] seconds
        if departure_time < 0 or departure_time > self.max_departure_time:
            raise ValueError(f"Departure time must be between 0 and {self.max_departure_time} seconds.")

        # Rate limiting: only allow max_probes_per_second in each second
        time_slot = int(departure_time)
        if self.probe_count_per_second.get(time_slot, 0) >= self.max_probes_per_second:
            raise Exception(f"Probe rate limit exceeded for second {time_slot}. "
                            f"Max {self.max_probes_per_second} probe per second allowed.")

        # Increment probe count for this second
        self.probe_count_per_second[time_slot] = self.probe_count_per_second.get(time_slot, 0) + 1

        # Get cached delay for specific time if it exists, otherwise generate new delay
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
        Aim for a KL divergence of <= 0.05
        
        :param pred_mean: Predicted mean of the delay distribution
        :param pred_std: Predicted standard deviation of the delay distribution
        :return: KL divergence score (lower is better)
        """
        # Validate that we're using a single-edge network
        if self.paths != "1" and self.paths != 1:
            raise ValueError("This function can only be used with a single-edge network (paths='1').")
            
        params = self.network.get_single_edge_distribution_parameters(self.network.SOURCE, self.network.DESTINATION)
        actual_mean = params["mean"]
        actual_std = params["std"]

        kl_div = math.log(pred_std / actual_std) + ((actual_std**2 + (actual_mean - pred_mean)**2) / (2 * pred_std**2)) - 0.5
        
        # Aim for a KL divergence of <= 0.05
        if kl_div <= 0.05:
            print(f"KL divergence: {kl_div:.4f} ✅")
        else:
            print(f"KL divergence: {kl_div:.4f} ❌")
        return kl_div

    def compare_distribution_parameters_2(self, mean1: float, mean2: float, std1: float, std2: float) -> list[float]:
        """
        Compare two predicted distributions to two actual edge distributions using KL divergence.
        Returns a list with two KL divergence scores.
        
        :param mean1: Predicted mean for the first edge
        :param mean2: Predicted mean for the second edge
        :param std1: Predicted standard deviation for the first edge
        :param std2: Predicted standard deviation for the second edge
        :return: List containing two KL divergence scores
        """
        # Validate that we're using a multi-edge network
        if self.paths != "2" and self.paths != 2:
            raise ValueError("This function can only be used with a dual-edge network (paths='2').")
            
        actual_params = self.network.get_multi_edge_distribution_parameters(self.network.SOURCE, self.network.DESTINATION)

        if len(actual_params) != 2:
            raise ValueError("Expected exactly two edges between nodes for multi-edge comparison.")

        actual1 = actual_params[0]
        actual2 = actual_params[1]

        kl1 = math.log(std1 / actual1["std"]) + ((actual1["std"] ** 2 + (actual1["mean"] - mean1) ** 2) / (2 * std1 ** 2)) - 0.5

        kl2 = math.log(std2 / actual2["std"]) + ((actual2["std"] ** 2 + (actual2["mean"] - mean2) ** 2) / (2 * std2 ** 2)) - 0.5

        print(f"KL divergence for edge 1: {kl1:.4f} {'✅' if kl1 <= 0.05 else '❌'}")
        print(f"KL divergence for edge 2: {kl2:.4f} {'✅' if kl2 <= 0.05 else '❌'}")

        return [kl1, kl2]
