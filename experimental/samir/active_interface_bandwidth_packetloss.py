import networkx as nx
from collections import deque
import math
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../network')))
from ground_truth import GroundTruthNetwork

class Simulator:
    def __init__(self, max_queue_size=45, max_penalty=100.0, k=0.1, loss_rate=0.01) -> None:
        """
        Initializes the simulator with a network model.

        :param max_queue_size: Maximum number of packets the network can handle before congestion.
        :param max_penalty: Maximum additional delay applied if bandwidth is exceeded.
        :param k: Controls how fast the penalty increases with congestion.
        """
        self.network = GroundTruthNetwork()
        self.event_log = []  # Stores (departure_time, arrival_time, delay)
        self.time_cache = {}  # Caches computed delays
        self.transmission_queue = deque()  # Queue to track packets in transit
        self.max_queue_size = max_queue_size  # Bandwidth limit
        self.max_penalty = max_penalty  # Maximum congestion penalty
        self.k = k  # Growth rate of penalty
        self.loss_rate = loss_rate  # Packet loss probability

    def compute_penalty(self) -> float:
        """
        Computes the delay penalty based on queue size using an exponential model.

        :return: Additional delay penalty (float).
        """
        excess = len(self.transmission_queue) - self.max_queue_size
        if excess <= 0:
            return 0.0  # No congestion

        return self.max_penalty * (1 - math.exp(-self.k * excess))  # Bounded exponential penalty

    def measure_end_to_end_delay(self) -> float:
        """
        Measures the total delay from SOURCE to DESTINATION along the shortest path.

        :return: The total delay as a float.
        """
        path: list[int] = nx.shortest_path(self.network.graph, self.network.SOURCE, self.network.DESTINATION)

        total_delay = 0.0
        for i in range(len(path) - 1):
            total_delay += self.network.sample_edge_delay(path[i], path[i + 1])

        return total_delay

    def send_probe_at(self, departure_time: float) -> tuple[float, float, float]:
        """
        Sends a probe at a specific time.

        :param departure_time: The time when the probe is sent.
        :return: A tuple (departure_time, arrival_time, delay).
        """

        # Simulate packet loss with a probability check
        if random.random() < self.loss_rate:
            # Packet is lost, return None for arrival time
            self.event_log.append((departure_time, None, None))
            return departure_time, None, None

        # Remove packets that have already arrived
        while self.transmission_queue and self.transmission_queue[0] <= departure_time:
            self.transmission_queue.popleft()

        # Use cached delay if available
        if departure_time in self.time_cache:
            delay = self.time_cache[departure_time]
        else:
            delay = self.measure_end_to_end_delay()
            self.time_cache[departure_time] = delay

        # Apply penalty if the queue is full
        if len(self.transmission_queue) >= self.max_queue_size:
            penalty = self.compute_penalty()
            #print(penalty) 
            delay += penalty

        arrival_time: float = departure_time + delay
        self.transmission_queue.append(arrival_time)

        # Log the event
        self.event_log.append((departure_time, arrival_time, delay))

        return departure_time, arrival_time, delay

    def send_multiple_probes(self, departure_times: list[float]) -> list[tuple[float, float, float]]:
        """
        Sends multiple probes at the specified times.

        :param departure_times: A list of times at which probes should be sent.
        :return: A list of (departure_time, arrival_time, delay).
        """
        results = []
        for departure_time in sorted(departure_times):  # Ensure probes are processed in order
            results.append(self.send_probe_at(departure_time))
        return results

    def get_event_log(self) -> list[tuple[float, float, float]]:
        """
        Returns the list of past probe events in the form (departure_time, arrival_time, delay).

        :return: List of tuples (departure_time, arrival_time, delay).
        """
        return sorted(self.event_log, key=lambda x: x[0])  # Sort by departure time

    def reset(self) -> None:
        """
        Resets the simulator's event log and time cache.
        """
        self.event_log = []
        self.time_cache = {}  # Clear cached delays
