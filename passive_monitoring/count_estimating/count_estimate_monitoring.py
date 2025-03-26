import time
import random
import math
from passive_monitoring.passive_monitoring_interface.general_sketch import Sketch

class CountEstimateSketch(Sketch):
    def __init__(self):
        self.samples = []
    def process_packet(self, packet, switch_id):
        self.samples.append(time.time())
    def record_sample(self):
        self.samples.append(time.time())
    def get_samples(self):
        return self.samples

class CountEstimateMonitor:
    def __init__(self, passive_simulator, min_delay_std=0.05):
        self.passive = passive_simulator
        self.min_delay_std = min_delay_std
        self.source_sketch = CountEstimateSketch()
        self.dest_sketch = CountEstimateSketch()
    def enable_monitoring(self, fluctuation_mean, fluctuation_std, drop_probability=0):
        original_transmit = self.passive.network.transmit_packet
        def modified_transmit(packet):
            self.source_sketch.record_sample()
            if random.random() < drop_probability:
                return
            delay = random.gauss(fluctuation_mean, fluctuation_std)
            if delay < 0:
                delay = 0
            time.sleep(delay)
            original_transmit(packet)
        self.passive.network.transmit_packet = modified_transmit
        self.passive.attach_sketch(self.passive.network.DESTINATION, self.dest_sketch)
    def get_source_samples(self):
        return self.source_sketch.get_samples()
    def get_destination_samples(self):
        return self.dest_sketch.get_samples()
    def compute_delay_distribution(self):
        src = self.get_source_samples()
        dst = self.get_destination_samples()
        n = min(len(src), len(dst))
        if n == 0:
            return None, None, []
        delays = [dst[i] - src[i] for i in range(n)]
        pred_mean = sum(delays) / n
        variance = sum((d - pred_mean) ** 2 for d in delays) / n
        pred_std = math.sqrt(variance)
        if pred_std < self.min_delay_std:
            return None, None, delays
        return pred_mean, pred_std, delays
    @staticmethod
    def compute_kl_divergence(true_mean, true_std, pred_mean, pred_std):
        return math.log(pred_std / true_std) + ((true_std**2 + (true_mean - pred_mean)**2) / (2 * pred_std**2)) - 0.5
    def run_capture_loop_until_convergence(self, true_mean, true_std, kl_threshold=0.05, fluctuation_threshold=5, sim_duration=3, avg_interarrival_ms=100, max_duration=60):
        start_time = time.time()
        prev_source_count = len(self.get_source_samples())
        prev_dest_count = len(self.get_destination_samples())
        while time.time() - start_time < max_duration:
            self.passive.simulate_traffic(duration_seconds=sim_duration, avg_interarrival_ms=avg_interarrival_ms)
            current_source_count = len(self.get_source_samples())
            current_dest_count = len(self.get_destination_samples())
            if (current_source_count - prev_source_count) >= fluctuation_threshold or (current_dest_count - prev_dest_count) >= fluctuation_threshold:
                pred_mean, pred_std, delays = self.compute_delay_distribution()
                if pred_mean is not None:
                    kl_div = self.compute_kl_divergence(true_mean, true_std, pred_mean, pred_std)
                    print("After {} samples, predicted mean = {:.4f}, std = {:.4f}, KL divergence = {:.4f}".format(len(delays), pred_mean, pred_std, kl_div))
                    if kl_div < kl_threshold:
                        print("Convergence reached!")
                        return pred_mean, pred_std, delays
                prev_source_count = current_source_count
                prev_dest_count = current_dest_count
        print("Max duration reached without convergence.")
        return None, None, self.compute_delay_distribution()[2]


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from passive_monitoring.count_estimating.count_estimate_monitoring import CountEstimateMonitor
from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
from active_monitoring_evolution.ground_truth import GroundTruthNetwork

network = GroundTruthNetwork(paths="1")
passive = PassiveSimulator(network)
monitor = CountEstimateMonitor(passive, min_delay_std=0.05)
gt_params = network.get_distribution_parameters(network.SOURCE, network.DESTINATION)
true_mean = gt_params["mean"]
true_std = gt_params["std"]
print("True Delay Distribution: mean = {:.4f}, std = {:.4f}".format(true_mean, true_std))
monitor.enable_monitoring(fluctuation_mean=true_mean, fluctuation_std=true_std, drop_probability=0.2)
pred_mean, pred_std, delays = monitor.run_capture_loop_until_convergence(
    true_mean=true_mean,
    true_std=true_std,
    kl_threshold=0.05,
    fluctuation_threshold=5,
    sim_duration=3,
    avg_interarrival_ms=100,
    max_duration=60
)
if pred_mean is not None:
    print("Final predicted delay distribution: mean = {:.4f}, std = {:.4f}".format(pred_mean, pred_std))
else:
    print("Failed to reach convergence within the maximum allowed time.")
if delays:
    x = np.linspace(min(delays), max(delays), 1000)
    true_pdf = norm.pdf(x, true_mean, true_std)
    pred_pdf = norm.pdf(x, pred_mean, pred_std) if pred_mean is not None else None
    plt.figure(figsize=(10, 6))
    plt.hist(delays, bins=30, density=True, alpha=0.5, label='Predicted Delay Histogram')
    plt.plot(x, true_pdf, 'r-', linewidth=2, label='Ground Truth Distribution')
    if pred_pdf is not None:
        plt.plot(x, pred_pdf, 'b--', linewidth=2, label='Predicted Distribution')
    plt.xlabel("Delay (seconds)")
    plt.ylabel("Density")
    plt.title("Comparison of Delay Distributions")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("No delay samples recorded; skipping plot.")





