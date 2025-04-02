# File: passive_monitoring/count_estimating/count_estimate_monitoring.py
import time
import random
import math
from passive_monitoring.passive_monitoring_interface.general_sketch import Sketch

class CountEstimateSketch(Sketch):
    def __init__(self):
        self.samples = []
        self.seen = set()

    def process_packet(self, packet, switch_id):
        try:
            attr = packet.get_attributes().get('flow_id')
        except AttributeError:
            attr = "unknown"
        self.seen.add(attr)
        self.samples.append((time.time(), len(self.seen)))

    def record_sample(self):
        self.samples.append((time.time(), len(self.seen)))

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
        if not src or not dst:
            return None, None, []
        src_peak = max(src, key=lambda x: x[1])
        dst_peak = max(dst, key=lambda x: x[1])
        delay = dst_peak[0] - src_peak[0]
        if delay < self.min_delay_std:
            return None, None, [delay]
        # For demonstration, we return the delay as both mean and derive std arbitrarily.
        return delay, delay * 0.1, [delay]

    @staticmethod
    def compute_kl_divergence(true_mean, true_std, pred_mean, pred_std):
        return math.log(pred_std / true_std) + ((true_std**2 + (true_mean - pred_mean)**2) / (2 * pred_std**2)) - 0.5

    def run_capture_loop_until_convergence(self, true_mean, true_std, kl_threshold=0.05, fluctuation_threshold=5, sim_duration=1, avg_interarrival_ms=10, max_duration=60):
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
