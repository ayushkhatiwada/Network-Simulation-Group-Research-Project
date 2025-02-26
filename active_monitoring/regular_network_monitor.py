import time
import numpy as np
import scipy.stats as stats
from active_monitoring.simulator import Simulator

# Probes sent 
class RegularNetworkMonitor:
    def __init__(self, probe_interval: float, num_probes: int):
        """
        Initializes the regular network monitor. Probes are sent at regular intervals

        :param probe_interval: Time interval between probes.
        :param num_probes: Number of probes to send.
        """
        self.simulator = Simulator()
        self.probe_interval = probe_interval
        self.num_probes = num_probes
        self.delays = []
    
    def run_monitoring(self):
        """
        Runs the monitoring by sending probes at regular intervals.
        """
        departure_times = [i * self.probe_interval for i in range(self.num_probes)]
        results = self.simulator.send_multiple_probes(departure_times)
        
        for departure_time, arrival_time, delay in results:
            self.delays.append(delay)
            print(f"Probe at {departure_time:.2f}s -> Delay: {delay:.2f}s")
        
        self.analyze_delays()
        self.identify_distribution()
    
    def analyze_delays(self):
        """
        Analyzes the collected delays to provide insights.
        """
        if not self.delays:
            print("No data collected.")
            return
        
        avg_delay = np.mean(self.delays)
        std_dev = np.std(self.delays)
        max_delay = np.max(self.delays)
        min_delay = np.min(self.delays)
        
        print("\n--- Network Delay Analysis ---")
        print(f"Average Delay: {avg_delay:.2f}s")
        print(f"Standard Deviation: {std_dev:.2f}s")
        print(f"Max Delay: {max_delay:.2f}s")
        print(f"Min Delay: {min_delay:.2f}s")
        print("-----------------------------")
    
    def identify_distribution(self):
        """
        Attempts to determine the best-fit distribution for the observed delays.
        """
        distributions = ["gamma", "norm", "expon"]
        best_fit = None
        best_p_value = -1
        
        for dist_name in distributions:
            dist = getattr(stats, dist_name)
            params = dist.fit(self.delays)
            _, p_value = stats.kstest(self.delays, dist_name, args=params)
            
            if p_value > best_p_value:
                best_p_value = p_value
                best_fit = dist_name
        
        print(f"Best-fit distribution: {best_fit} (p-value: {best_p_value:.4f})")

monitor = RegularNetworkMonitor(probe_interval=1.0, num_probes=50)
monitor.run_monitoring()