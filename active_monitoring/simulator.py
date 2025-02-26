import networkx as nx
from ground_truth import GroundTruthNetwork

class Simulator:
    def __init__(self) -> None:
        """
        Initializes the simulator with a network model.
        """
        self.network = GroundTruthNetwork()
        self.event_log = []  # Stores (departure_time, arrival_time, delay)
        self.time_cache = {}  # Caches computed delays

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
        # Use cached delay if available
        if departure_time in self.time_cache:
            delay = self.time_cache[departure_time]
        else:
            delay = self.measure_end_to_end_delay()
            self.time_cache[departure_time] = delay

        arrival_time: float = departure_time + delay

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
