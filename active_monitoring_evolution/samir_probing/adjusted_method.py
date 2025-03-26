import sys  
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))
from active_simulator_v1 import ActiveSimulator_v1
from active_simulator_v2 import ActiveSimulator_v2

import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple

# **Base Class NetworkMonitoring**
class NetworkMonitoring:
    def __init__(self, num_probes: int = 100, simulator = None):
        """
        Initializes the base class for network monitoring.
        :param num_probes: Number of probes to send.
        """
        self.num_probes = num_probes
        self.delays = []
        if simulator is None:
            self.simulator = ActiveSimulator_v1()
        else:
            self.simulator = simulator
        self.packets_dropped = 0

    def run_monitoring(self):
        """
        Placeholder method to be implemented by subclasses.
        """
        raise NotImplementedError("run_monitoring must be implemented by the subclass.")

    def calculate_distribution(self):
        """
        Fits different probability distributions and selects the best fit using the Kolmogorov-Smirnov test.
        """
        if not self.delays:
            print("No data is available.")
            return

        distributions = [stats.norm, stats.gamma, stats.expon, stats.lognorm, stats.weibull_min]

        best_fit_distribution = None
        best_params = None
        best_p_value = -1  

        for dist in distributions:
            params = dist.fit(self.delays)
            _, p_value = stats.kstest(self.delays, dist.cdf, args=params)

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
        if not self.delays:
            print("No data is available.")
            return

        plt.figure(figsize=(8, 5))
        sns.histplot(self.delays, bins=30, kde=True, color="blue", alpha=0.6, label="Histogram + KDE")

        plt.xlabel("Delay")
        plt.ylabel("Frequency")
        plt.title("Observed Delay Distribution")
        plt.legend()
        plt.show()

    def get_delay_statistics(self):
        """
        Returns basic statistics about the delays: range, min, max, and standard deviation.
        """
        
        if not self.delays:
            print("No data is available.")
            return
        
        delay_range = np.ptp(self.delays)  # Range (max - min)
        delay_min = np.min(self.delays)  # Minimum delay
        delay_max = np.max(self.delays)  # Maximum delay
        delay_std = np.std(self.delays)  # Standard deviation

        # Print or return the statistics
        print(f"Delay Statistics:")
        print(f"Range: {delay_range}")
        print(f"Min: {delay_min}")
        print(f"Max: {delay_max}")
        print(f"Standard Deviation: {delay_std}")
        print(f"Number of Dropped Packets: {self.packets_dropped}")
        print(f"Number of Arrived Packets: {len(self.delays)}")

# **Regular Probing Strategy**
class RegularNetworkMonitoring(NetworkMonitoring):
    def __init__(self, probe_interval: float, num_probes: int):
        """
        Initializes the regular network monitor.
        :param probe_interval: Time interval between probes.
        :param num_probes: Number of probes to send.
        """
        super().__init__(num_probes)
        self.probe_interval = probe_interval

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
        self.packets_dropped += sum(1 for event in self.results if event is None)
        self.delays = [event for event in self.results if event is not None]

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
        Ensures a maximum of 10 probes per second.
        """
        start_time = 0
        current_interval = self.probe_interval  # Start with the initial probe interval

        for i in range(self.num_probes):
            # Send the probe and collect the results
            result = self.simulator.send_probe_at(start_time)
            if result:
                self.delays.append(result)
            else:
                self.packets_dropped += 1

            if len(self.delays) % 5 == 4:
                delay_variance = np.var(self.delays[-5:])  # Calculate variance of delays

                # Adjust probe interval based on variance, but ensure the interval doesn't exceed 0.1 seconds
                if delay_variance > self.threshold:
                    # Increase the interval if variance is above threshold
                    current_interval *= 1.2
                    print(f"Variance is high. Increasing probe interval to {current_interval}")
                else:
                    # Decrease the interval if variance is below threshold
                    current_interval *= 0.9
                    print(f"Variance is low. Decreasing probe interval to {current_interval}")
                
                # Ensure the interval does not exceed the max allowed (0.1s for 10 probes/sec)
                current_interval = max(current_interval, 1.0001)

            start_time += current_interval

class BurstNetworkMonitoring(NetworkMonitoring):
    def __init__(self, num_bursts: int = 10, probes_per_burst: int = 10, burst_interval: float = 0.35, idle_time: float = 7.22):
        """
        :param num_bursts: Number of bursts to send.
        :param probes_per_burst: Number of probes to send per burst.
        :param burst_interval: Time interval between probes within a burst.
        :param idle_time: Time interval between bursts.
        """
        super().__init__(num_bursts * probes_per_burst)
        self.num_bursts = num_bursts
        self.probes_per_burst = probes_per_burst
        self.burst_interval = burst_interval
        self.idle_time = idle_time

    def run_monitoring(self):
        """
        Sends probes in bursts followed by idle time
        """
        start_time = 0
        
        for burst in range(self.num_bursts):
            # Send probes in quick succession for the current burst
            for i in range(self.probes_per_burst):
                result = self.simulator.send_probe_at(start_time + i * self.burst_interval)
                if result:
                    self.delays.append(result)
                else:
                    self.packets_dropped += 1
            
            # After finishing the burst, wait for idle_time before the next burst
            start_time += self.probes_per_burst * self.burst_interval + self.idle_time
            
class PacketPairNetworkMonitoring(NetworkMonitoring):
    def __init__(self, num_pairs: int = 100, pair_interval: float = 0.1, probe_interval: int = 1):
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
        start_time = 0 
        
        for pair in range(self.num_pairs):
            # Send first probe
            first_probe_time = start_time
            first_probe_arrival_time = (self.simulator.send_probe_at(first_probe_time))
            
            # Send second probe (after pair interval)
            second_probe_time = first_probe_time + self.pair_interval
            second_probe_arrival_time = (self.simulator.send_probe_at(second_probe_time))

            # Check if either or both probes are dropped
            if first_probe_arrival_time is None and second_probe_arrival_time is None:
                self.packets_dropped += 2  # Both probes dropped
                continue
            elif first_probe_arrival_time is None:
                self.packets_dropped += 1  # Only the first probe dropped
                continue
            elif second_probe_arrival_time is None:
                self.packets_dropped += 1  # Only the second probe dropped
                continue

            # Calculate delay difference between the two probes in the pair
            delay_diff = abs(first_probe_arrival_time - second_probe_arrival_time)
            self.delays.append(delay_diff)
                        
            # Wait before sending the next pair
            start_time += self.probe_interval

def plot_kl_divergence_vs_probes(
    simulator,  # simulator instance (ActiveSimulator_v1)
    min_probes: int = 10,
    max_probes: int = 100,
    step: int = 10,
    probe_interval: float = 1.0,
    target_kl: float = 0.05,
    save_path: str = "kl_divergence_vs_probes_v2.png"
) -> Tuple[List[int], List[float]]:
    """
    Plots KL divergence vs. number of probes for RegularNetworkMonitoring.
    
    Args:
        simulator: Instance of the network simulator.
        min_probes: Minimum number of probes to test.
        max_probes: Maximum number of probes to test.
        step: Step size between probe counts.
        probe_interval: Time interval between probes.
        target_kl: Target KL divergence threshold (default: 0.05).
        save_path: Path to save the plot image.
    
    Returns:
        Tuple of (probe_counts, kl_scores) for further analysis.
    """
    probe_counts = list(range(min_probes, max_probes + 1, step))
    kl_scores = []
    
    for num_probes in probe_counts:
        # Initialize and run regular monitoring
        monitor = AdaptiveNetworkMonitoring(
            probe_interval=probe_interval,
            num_probes=num_probes
        )
        monitor.simulator = simulator  # Use the provided simulator
        monitor.run_monitoring()
        
        if not monitor.delays:
            kl_scores.append(float('inf'))  # No data case
            continue
        
        # Calculate predicted mean and std
        pred_mean = np.mean(monitor.delays)
        pred_std = np.std(monitor.delays)
                
        # Compute KL divergence
        kl_div = simulator.compare_distribution_parameters(pred_mean, pred_std)
        
        kl_scores.append(kl_div)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(probe_counts, kl_scores, 'bo-', label='KL Divergence')
    plt.axhline(y=target_kl, color='r', linestyle='--', label=f'Target ({target_kl})')
    
    plt.xlabel('Number of Probes')
    plt.ylabel('KL Divergence Score')
    plt.title('KL Divergence vs. Number of Probes (adaptive monitoring) For V1')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"KL divergence plot saved to: {save_path}")
    return probe_counts, kl_scores

simulator = ActiveSimulator_v1()
probe_counts, kl_scores = plot_kl_divergence_vs_probes(
    simulator=simulator,
    min_probes=10,
    max_probes=80,
    step=10,
    probe_interval=1.0,
    save_path="adaptive_monitoring_kl_divergence_v1.png"
)