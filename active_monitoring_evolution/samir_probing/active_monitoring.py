import sys  
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from active_probing_algos.active_simulator_v0 import ActiveSimulator_v0
from active_probing_algos.active_simulator_v1 import ActiveSimulator_v1
from active_probing_algos.active_simulator_v2 import ActiveSimulator_v2

import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import deque
from scipy.stats import norm
import random
import statistics

class NetworkMonitoring:
    def __init__(self, simulator, max_probes_per_second, time_limit):
        self.simulator = simulator
        self.max_probes_per_second = max_probes_per_second
        self.time_limit = time_limit
        self.delays = []

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

    def overlay_distributions(self):
        plt.figure(figsize=(8, 5))
        sns.kdeplot(self.delays, color="blue", label="KDE of Observed Delays")
        mean_delay = statistics.mean(self.delays)
        std_delay = statistics.stdev(self.delays)
        kl_score = self.simulator.compare_distribution_parameters(mean_delay, std_delay)
        x = np.linspace(min(self.delays), max(self.delays), 100)
        plt.plot(x, norm.pdf(x, 0.8, 0.15), 'r-', label="True N(0.8, 0.15)")
        plt.xlabel("Delay")
        plt.ylabel("Density")
        plt.title(f"Observed Delays vs. True Normal Distribution\nKL Divergence: {kl_score:.4f}")
        plt.legend()
        plt.show()

class ActiveMonitor(NetworkMonitoring):
    def __init__(self, simulator, max_probes_per_second, time_limit):
        super().__init__(simulator, max_probes_per_second, time_limit)
        self.max_rate = max_probes_per_second
        self.min_rate = 0.5

        self.drop_threshold = 0.20
        self.spike_threshold = 1.5
        self.baseline_delays = []

        self.in_congestion = False
        self.last_delays = deque(maxlen=5)
        self.last_drops = deque(maxlen=5)
        self.baseline_delay = None

    def update_baseline(self):
        print(self.baseline_delay)
        if len(self.last_delays) < 3 or len(self.last_drops) < 3:
            return

        drop_rate = sum(self.last_drops) / len(self.last_drops)
        delay_variance = statistics.variance(self.last_delays)

        if drop_rate < 0.05 and delay_variance < 0.1:
            self.baseline_delay = statistics.mean(self.last_delays)
            self.baseline_delays.append(self.baseline_delay)

    def check_congestion(self, result):
        if result is None:
            self.last_drops.append(1)
        else:
            self.last_drops.append(0)
            if not self.in_congestion:
                self.last_delays.append(result)

        if self.baseline_delay:
            if result and result > self.spike_threshold * statistics.median(self.baseline_delays):
                return "congestion"

        if len(self.last_drops) >= 3:
            drop_rate = sum(self.last_drops) / len(self.last_drops)
            if drop_rate > self.drop_threshold:
                return "congestion"

        self.update_baseline()
        return "stable"

    def adjust_rate(self, status):
        if status == "congestion":
            self.in_congestion = True
            self.current_rate = self.min_rate
        else:
            self.current_rate = self.max_rate
            self.in_congestion = False

    def run_monitoring(self):
        current_time = 0.0
        probes_in_current_second = 0
        last_probe_time = 0

        while current_time < 100:
            if probes_in_current_second < self.max_rate:
                result = self.simulator.send_probe_at(current_time)
                status = self.check_congestion(result)
                self.adjust_rate(status)

                if result is not None:
                    self.delays.append(result)

                probes_in_current_second += 1
                last_probe_time = current_time

            if current_time + (1 / self.current_rate)  > int(last_probe_time) + 1:
                if self.in_congestion == True:
                    current_time += 1.0 / self.current_rate
                else:
                    current_time = int(last_probe_time) + 1
                probes_in_current_second = 0
            else:
                current_time += 1 / self.current_rate

simulator = ActiveSimulator_v2()
monitor = ActiveMonitor(simulator, 10, 100)
monitor.run_monitoring()
monitor.overlay_distributions()