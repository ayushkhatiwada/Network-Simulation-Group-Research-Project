from active_interface_bandwidth_packetloss import Simulator
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# **Base Class NetworkMonitoring**
class NetworkMonitoring:
    def __init__(self, num_probes: int):
        """
        Initializes the base class for network monitoring.
        :param num_probes: Number of probes to send.
        """
        self.num_probes = num_probes
        self.delays = []  # Initialize delays list
        self.simulator = Simulator()

    def run_monitoring(self):
        """
        Placeholder method to be implemented by subclasses.
        """
        raise NotImplementedError("run_monitoring must be implemented by the subclass.")

    def calculate_distribution(self):
        """
        Fits different probability distributions and selects the best fit using the Kolmogorov-Smirnov test.
        """
        # Filter out None values from delays
        temp_delays = [delay for delay in self.delays if delay is not None]
        
        if not temp_delays:
            print("No data is available.")
            return

        distributions = [stats.norm, stats.gamma, stats.expon, stats.lognorm, stats.weibull_min]

        best_fit_distribution = None
        best_params = None
        best_p_value = -1  

        for dist in distributions:
            params = dist.fit(temp_delays)
            _, p_value = stats.kstest(temp_delays, dist.cdf, args=params)

            if p_value > best_p_value:
                best_p_value = p_value
                best_fit_distribution = dist
                best_params = params

        print(f"Best-fit distribution: {best_fit_distribution.name}")
        print(f"Best-fit parameters: {best_params}")

    def plot_distribution(self):
        """
        Plots the histogram and KDE of observed delays.
        """
        # Filter out None values from delays
        temp_delays = [delay for delay in self.delays if delay is not None]
        
        if not temp_delays:
            print("No data is available.")
            return

        plt.figure(figsize=(8, 5))
        sns.histplot(temp_delays, bins=30, kde=True, color="blue", alpha=0.6, label="Histogram + KDE")

        plt.xlabel("Delay")
        plt.ylabel("Frequency")
        plt.title("Observed Delay Distribution")
        plt.legend()
        plt.show()

    def get_delay_statistics(self):
        """
        Returns basic statistics about the delays: range, min, max, and standard deviation.
        """
        # Filter out None values from delays
        temp_delays = [delay for delay in self.delays if delay is not None]
        
        if not temp_delays:
            print("No data is available.")
            return
        
        print(temp_delays)

        delay_range = np.ptp(temp_delays)  # Range (max - min)
        delay_min = np.min(temp_delays)  # Minimum delay
        delay_max = np.max(temp_delays)  # Maximum delay
        delay_std = np.std(temp_delays)  # Standard deviation
        dropped_packets = sum(1 for delay in self.delays if delay is None)

        # Print or return the statistics
        print(f"Delay Statistics:")
        print(f"Range: {delay_range}")
        print(f"Min: {delay_min}")
        print(f"Max: {delay_max}")
        print(f"Standard Deviation: {delay_std}")
        print(f"Number of Dropped Packets: {dropped_packets}")

# **Regular Probing Strategy**
class RegularNetworkMonitoring(NetworkMonitoring):
    def __init__(self, probe_interval: float, num_probes: int):
        """
        Initializes the regular network monitor.
        :param probe_interval: Time interval between probes.
        :param num_probes: Number of probes to send.
        """
        super().__init__(num_probes)
        self.probe_interval = probe_interval  # Initialize the probe interval

    def run_monitoring(self):
        """
        Sends probes at regular intervals and calculates the distribution of delays.
        """
        self.departure_times = []
        start_time = 0
        for _ in range(self.num_probes):
            self.departure_times.append(start_time)
            start_time += self.probe_interval

        self.results = self.simulator.send_multiple_probes(self.departure_times)
        self.delays = [event[2] for event in self.results]  # Extract only the delays
        
        #self.calculate_distribution(self.delays)
        #self.plot_distribution()


# **Adaptive Probing Strategy**
class AdaptiveNetworkMonitoring(NetworkMonitoring):
    def __init__(self, num_probes: int, threshold: float = 0.1, probe_interval: float = 1.0):
        """
        Initializes the adaptive network monitor.
        :param num_probes: Number of probes to send.
        :param threshold: Variance threshold for adjusting the probe interval.
        :param probe_threshold: Initial probing interval.
        """
        super().__init__(num_probes)
        self.threshold = threshold  # Threshold for variance change
        self.probe_interval = probe_interval

    def run_monitoring(self):
        """
        Runs adaptive monitoring where the probe interval may change based on variance of delays.
        After every 5 probes, the probe interval is adjusted based on delay variance.
        """
        start_time = 0
        current_interval = self.probe_interval  # Start with the initial probe interval
        self.delays = []  # Store delays as we collect them

        for i in range(self.num_probes):
            # Send the probe and collect the results
            results = self.simulator.send_probe_at(start_time)
            delay = results[2]
            self.delays.append(delay)
            
            if i % 5 == 4:
                delay_variance = np.var(self.delays[-5:])  # Calculate variance of delays
                
                # Adjust probe interval based on variance
                if delay_variance > self.threshold:
                    # Increase the interval if variance is above threshold
                    current_interval *= 1.2
                    print(f"Variance is high. Increasing probe interval to {current_interval}")
                else:
                    # Decrease the interval if variance is below threshold
                    current_interval *= 0.9
                    print(f"Variance is low. Decreasing probe interval to {current_interval}")
            
            start_time += current_interval

        #self.calculate_distribution(self.delays)
        #self.plot_distribution()

class BurstNetworkMonitoring(NetworkMonitoring):
    def __init__(self, num_bursts: int, probes_per_burst: int, burst_interval: float, idle_time: float):
        """
        :param num_bursts: Number of bursts to send.
        :param probes_per_burst: Number of probes to send per burst.
        :param burst_interval: Time interval between probes within a burst.
        :param idle_time: Time interval between bursts.
        """
        super().__init__(num_bursts * probes_per_burst)  # Total number of probes is num_bursts * probes_per_burst
        self.num_bursts = num_bursts
        self.probes_per_burst = probes_per_burst
        self.burst_interval = burst_interval
        self.idle_time = idle_time

    def run_monitoring(self):
        """
        Sends probes in bursts followed by idle time
        """
        self.delays = [] 
        start_time = 0
        
        for burst in range(self.num_bursts):
            # Send probes in quick succession for the current burst
            for i in range(self.probes_per_burst):
                results = self.simulator.send_probe_at(start_time + i * self.burst_interval)
                delay = results[2] 
                self.delays.append(delay)
            
            # After finishing the burst, wait for idle_time before the next burst
            start_time += self.probes_per_burst * self.burst_interval + self.idle_time
            
            #print(f"Completed burst {burst + 1}/{self.num_bursts} with delays: {self.delays[-self.probes_per_burst:]}")  # Last probes delays
            
        #self.calculate_distribution(self.delays)
        #self.plot_distribution()

class PacketPairNetworkMonitoring(NetworkMonitoring):
    def __init__(self, num_pairs: int, pair_interval: float, probe_interval: float):
        """
        :param num_pairs: Number of probe pairs to send.
        :param pair_interval: Interval between the first and second packet in a pair.
        :param probe_interval: Interval between sending different pairs.
        """
        super().__init__(num_probes=num_pairs * 2)
        self.num_pairs = num_pairs 
        self.pair_interval = pair_interval  # Interval between the two probes in each pair
        self.probe_interval = probe_interval

    def run_monitoring(self):
        """
        Sends packet pairs and calculates the delay differences between them.
        """
        self.delays = []  
        start_time = 0 
        
        for pair in range(self.num_pairs):
            first_probe_time = start_time
            first_probe_arrival_time = (self.simulator.send_probe_at(first_probe_time))[1]
            
            second_probe_time = first_probe_time + self.pair_interval
            second_probe_arrival_time = (self.simulator.send_probe_at(second_probe_time))[1]
                        
            delay_diff = abs(first_probe_arrival_time - second_probe_arrival_time)
            self.delays.append(delay_diff)
            
            #print(f"Completed packet pair {pair + 1}/{self.num_pairs}, delay diff: {delay_diff}")
            
            # Wait before sending the next pair
            start_time += self.probe_interval
        
        # After all pairs, calculate the distribution of delay differences
        #self.calculate_distribution(self.delays)
        #self.plot_distribution()

probe_interval = 1 
num_probes = 100

regular_monitor = RegularNetworkMonitoring(probe_interval, num_probes)
regular_monitor.run_monitoring()
regular_monitor.get_delay_statistics()
regular_monitor.calculate_distribution()
regular_monitor.plot_distribution()