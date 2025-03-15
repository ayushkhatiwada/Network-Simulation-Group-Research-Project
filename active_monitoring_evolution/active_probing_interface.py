import time
from ground_truth import GroundTruthNetwork  # Assumes this is available in your environment

class NetworkProbeInterface:
    def __init__(self, distribution_type="normal", probe_limit=5, time_window=1.0):
        """
        Initialize the interface with a GroundTruthNetwork instance and probe rate limiting.

        Parameters:
        - distribution_type (str): 'normal' or 'lognormal' to set the network delay distribution.
        - probe_limit (int): Maximum number of allowed probes within a time window.
        - time_window (float): The length of the time window (in seconds) for probe rate limiting.
        """
        self.network = GroundTruthNetwork(distribution_type=distribution_type)
        self.probe_limit = probe_limit
        self.time_window = time_window
        self.probe_timestamps = []  # Stores the time of each probe sent

    def send_probe(self):
        """
        Sends a probe to the network if the probe rate limit has not been exceeded.
        The probe measures the delay from the SOURCE to the DESTINATION.

        Returns:
        - float: The simulated delay (in milliseconds) for the probe.

        Raises:
        - Exception: If the user has exceeded the allowed number of probes in the time window.
        """
        current_time = time.time()
        # Remove any timestamps that are outside the allowed time window.
        self.probe_timestamps = [t for t in self.probe_timestamps if current_time - t < self.time_window]
        
        if len(self.probe_timestamps) >= self.probe_limit:
            raise Exception("Probe limit reached. Please wait before sending more probes.")
        
        # Record the current probe timestamp.
        self.probe_timestamps.append(current_time)
        
        # Simulate sending the probe: sample the delay from the network's source to destination.
        delay = self.network.sample_edge_delay(self.network.SOURCE, self.network.DESTINATION)
        return delay

    def get_probe_history(self):
        """
        Returns the list of timestamps when probes were sent.
        
        Returns:
        - list of float: Timestamps of the recorded probes.
        """
        return self.probe_timestamps

# Example usage:
if __name__ == "__main__":
    probe_interface = NetworkProbeInterface(distribution_type="normal", probe_limit=3, time_window=1.0)
    try:
        for _ in range(5):
            print("Probe delay:", probe_interface.send_probe(), "ms")
            time.sleep(0.3)  # Sleep to simulate probes at different times
    except Exception as e:
        print(e)
