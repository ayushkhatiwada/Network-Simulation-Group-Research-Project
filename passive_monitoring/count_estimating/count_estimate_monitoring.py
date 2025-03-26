import time
import math
import random
from passive_monitoring.count_estimating.count_estimate_sketch import CountEstimateSketch

class CountEstimateMonitor:
    def __init__(self, passive_simulator, min_delay_std=0.05):
        self.passive = passive_simulator
        self.source_sketch = CountEstimateSketch()
        self.dest_sketch = CountEstimateSketch()
        self.min_delay_std = min_delay_std

    def enable_monitoring(self, fluctuation_mean, fluctuation_std):
        original_transmit = self.passive.network.transmit_packet

        def modified_transmit(packet):
            self.source_sketch.record_sample()
            delay = random.gauss(fluctuation_mean, fluctuation_std)
            if delay < 0:
                delay = 0
            time.sleep(delay)  
            original_transmit(packet)

        self.passive.network.transmit_packet = modified_transmit
        print("Patched transmit_packet to record source events with simulated fluctuations.")

        self.passive.attach_sketch(self.passive.network.DESTINATION, self.dest_sketch)
        print("Attached SimpleCountSketch to destination switch for receive monitoring.")

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
        kl_div = math.log(pred_std / true_std) + ((true_std**2 + (true_mean - pred_mean)**2) / (2 * pred_std**2)) - 0.5
        return kl_div

    def run_capture_loop_until_convergence(self,
                                           true_mean,
                                           true_std,
                                           kl_threshold=0.05,
                                           fluctuation_threshold=5,
                                           sim_duration=1,
                                           avg_interarrival_ms=10,
                                           max_duration=60):
        start_time = time.time()
        prev_source_count = len(self.get_source_samples())
        prev_dest_count = len(self.get_destination_samples())

        while time.time() - start_time < max_duration:
            self.passive.simulate_traffic(duration_seconds=sim_duration, avg_interarrival_ms=avg_interarrival_ms)
            current_source_count = len(self.get_source_samples())
            current_dest_count = len(self.get_destination_samples())

            if ((current_source_count - prev_source_count) >= fluctuation_threshold or 
                (current_dest_count - prev_dest_count) >= fluctuation_threshold):
                
                pred_mean, pred_std, delays = self.compute_delay_distribution()
                if pred_mean is not None:
                    kl_div = self.compute_kl_divergence(true_mean, true_std, pred_mean, pred_std)
                    print(f"After {len(delays)} samples, predicted mean = {pred_mean:.4f}, "
                          f"std = {pred_std:.4f}, KL divergence = {kl_div:.4f}")
                    if kl_div < kl_threshold:
                        print("Convergence reached!")
                        return pred_mean, pred_std, delays
                prev_source_count = current_source_count
                prev_dest_count = current_dest_count
        print("Max duration reached without convergence.")
        return None, None, self.compute_delay_distribution()[2]