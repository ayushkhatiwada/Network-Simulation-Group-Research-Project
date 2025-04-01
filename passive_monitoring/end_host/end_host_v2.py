import time
import random
import numpy as np
import matplotlib.pyplot as plt
from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
from active_monitoring_evolution.ground_truth import GroundTruthNetwork
from passive_monitoring.end_host.end_host_latency_measurement import EndHostEstimation
from passive_monitoring.passive_monitoring_interface.switch_and_packet import Packet

def evolution_2_with_congestion(min_window_size=100, max_window_size=1000, window_increment=100, apply_filtering=None, discard_method="median_filter"):
    # print(f"\n=== Evolution 3: Estimating Latency Parameters with Congestion === filtering:")

    target_kl = 0.05
    trials_per_window = 10
    simulation_duration = 5

    true_mean = 0.8
    true_std = 0.15

    results = []

    for window_size in range(min_window_size, max_window_size + 1, window_increment):
        print(f"\nTesting window size: {window_size}")
        kl_scores = []

        for trial in range(trials_per_window):
            network = GroundTruthNetwork()
            simulator = PassiveSimulator(network, 42)
            simulator.normal_drop_probability = 0.1
            simulator.congested_drop_probability = 0.4
            simulator.enable_congestion_simulation(network.DESTINATION)

            monitor = EndHostEstimation(
                window_size=window_size,
                apply_filtering=apply_filtering,
                discard_method=discard_method,
                true_mean=true_mean,
                true_std=true_std
            )

            monitor.report_interval = float('inf')
            simulator.attach_sketch(network.DESTINATION, monitor)

            # Custom traffic simulation
            start_time = time.time()
            while time.time() - start_time < simulation_duration:
                packet = Packet(network.SOURCE, network.DESTINATION)
                packet.true_delay = network.sample_edge_delay(packet.source, packet.destination)

                # Simulate dropped packets manually to match congestion behavior
                drop_chance = simulator.congested_drop_probability if simulator.simulation_start_time and any(
                    start <= time.time() - simulator.simulation_start_time <= end for start, end in simulator.congestion_intervals
                ) else simulator.normal_drop_probability

                if random.random() < drop_chance:
                    continue  # Drop the packet

                time.sleep(0.001)  # send spacing
                simulator.network.destination_switch.receive(packet)

            # Evaluate estimation
            params = monitor.estimate_parameters()
            est_mean = params['estimated_mean']
            est_std = params['estimated_std']
            if est_mean is None or est_std is None:
                continue

            kl_score = simulator.compare_distribution_parameters(est_mean, est_std)
            kl_scores.append(kl_score)

        if kl_scores:
            avg_kl = np.mean(kl_scores)
            results.append((window_size, avg_kl))
            print(f"  Window size: {window_size}")
            print(f"  Average KL score: {avg_kl:.6f}")
            if avg_kl <= target_kl:
                print(f"\n*** Optimal window size found: {window_size} ***")
                break

    print(f"\n=== Evolution 3 Results (filtering: {discard_method}) ===")
    print("Window Size | Avg KL Score")
    print("--------------------------")
    for window_size, avg_kl in results:
        print(f"{window_size:11d} | {avg_kl:12.6f}")
    optimal_windows = [w for w, kl in results if kl <= target_kl]
    if optimal_windows:
        print(f"\nSmallest window size for achieving KL score ≤ {target_kl}: {min(optimal_windows)}")
    else:
        print(f"\nNo window size achieved the target KL score of {target_kl} consistently.")

    # Plot
    window_sizes = [w for w, kl in results]
    kl_scores = [kl for w, kl in results]

    plt.figure(figsize=(10, 6))
    plt.plot(window_sizes, kl_scores, marker='o', label='KL Divergence')
    plt.axhline(y=0.05, color='gray', linestyle='--', label='Target KL Threshold (0.05)')
    plt.title("Evolution 3: KL Divergence vs. Window Size (With Congestion)")
    plt.xlabel("Window Size")
    plt.ylabel("Average KL Divergence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


evolution_2_with_congestion(20, 1000, 20, False, None)