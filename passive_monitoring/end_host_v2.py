import time
import random
import numpy as np
from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
from active_monitoring_evolution.ground_truth import GroundTruthNetwork
from passive_monitoring.end_host_latency_measurement import EndHostEstimation
from passive_monitoring.passive_monitoring_interface.switch_and_packet import Packet

def evolution_2_with_congestion_and_drops():
    print("=== Evolution 2: Estimating Under Congestion + Drops (Normal Distribution) ===")

    target_kl = 0.05
    min_window_size = 100
    max_window_size = 1000
    window_increment = 100
    trials_per_window = 5
    simulation_duration = 5  # seconds

    # Ground truth delay params
    true_mean = 0.8
    true_std = 0.15

    results = []

    for window_size in range(min_window_size, max_window_size + 1, window_increment):
        print(f"\nTesting window size: {window_size}")
        kl_scores = []

        for trial in range(trials_per_window):
            # Setup network + simulator
            network = GroundTruthNetwork()
            simulator = PassiveSimulator(network)

            # Enable congestion + drop behavior
            simulator.enable_congestion_simulation(network.DESTINATION)
            simulator.congested_drop_probability = 0.4  # Drop rate during congestion

            # Create estimator
            monitor = EndHostEstimation(
                window_size=window_size,
                apply_filtering=False,
                true_mean=true_mean,
                true_std=true_std
            )
            monitor.report_interval = float('inf')
            simulator.attach_sketch(network.DESTINATION, monitor)

            # Simulate manual packet sending
            start_time = time.time()
            while time.time() - start_time < simulation_duration:
                packet = Packet(network.SOURCE, network.DESTINATION)
                delay = network.sample_edge_delay(packet.source, packet.destination)

                sim_time = time.time() - simulator.simulation_start_time
                in_congestion = any(start <= sim_time <= end for start, end in simulator.congestion_intervals)
                
                # Apply congestion delay factor
                if in_congestion:
                    delay += (simulator.congestion_delay_factor - 1.0) * 10  # ms
                    drop_prob = simulator.congested_drop_probability
                else:
                    drop_prob = simulator.normal_drop_probability

                # Simulate packet drop
                if random.random() < drop_prob:
                    continue  # Packet dropped, do not deliver

                # Otherwise, record and deliver
                packet.true_delay = delay
                time.sleep(0.001)
                simulator.network.destination_switch.receive(packet)

            # Estimate + evaluate
            params = monitor.estimate_parameters()
            est_mean = params['estimated_mean']
            est_std = params['estimated_std']
            if est_mean is None or est_std is None:
                continue

            kl_score = simulator.compare_distribution_parameters(est_mean, est_std)
            kl_scores.append(kl_score)

        if kl_scores:
            avg_kl = np.mean(kl_scores)
            success_rate = np.mean([kl <= target_kl for kl in kl_scores])
            results.append((window_size, avg_kl, success_rate))

            print(f"  Window size: {window_size}")
            print(f"  Average KL score: {avg_kl:.6f}")
            print(f"  Success rate: {success_rate:.2%}")

            # if avg_kl <= target_kl and success_rate >= 0.8:
            if avg_kl <= target_kl:
                print(f"\n*** Optimal window size found under congestion + drops: {window_size} ***")
                break

    print("\n=== Evolution 2 Results (Congestion + Drops) ===")
    print("Window Size | Avg KL Score | Success Rate")
    print("----------------------------------------")
    for window_size, avg_kl, success_rate in results:
        print(f"{window_size:11d} | {avg_kl:12.6f} | {success_rate:11.2%}")

    return results

if __name__ == "__main__":
    evolution_2_with_congestion_and_drops()
