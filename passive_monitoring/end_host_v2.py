import time
import numpy as np
from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
from active_monitoring_evolution.ground_truth import GroundTruthNetwork
from passive_monitoring.end_host_latency_measurement import EndHostEstimation

def evolution_2_congestion():
    """
    Evolution 2: Congestion scenario with both delay noise and dropped probes.
    Goal: Estimate the true exponential distribution parameters under challenging conditions.
    """
    print("=== Evolution 2: Congestion (Noise + Dropped Probes) ===")

    # Experiment parameters
    target_kl = 0.05
    drop_probability = 0.2
    min_window_size = 50
    max_window_size = 500
    window_increment = 50
    trials_per_window = 5
    simulation_duration = 5 
    avg_interarrival_ms = 10

    # filter guidelines
    apply_filter = "True"
    discard_method = "threshold"

    # ground truth params
    true_mean = 0.01
    true_congested_mean = 0.015
    true_lambda = 1 / true_mean

    results = []

    for window_size in range(min_window_size, max_window_size + 1, window_increment):
        print(f"\nTesting window size: {window_size}")
        kl_scores = []

        for trial in range(trials_per_window):
            network = GroundTruthNetwork(paths=1)  # 2 nodes, 1 edge
            simulator = PassiveSimulator(network)

            # Enable congestion simulation and drop probability
            simulator.enable_congestion_simulation(network.DESTINATION)
            simulator.set_drop_probability(network.DESTINATION, drop_probability)

            # Create estimator
            monitor = EndHostEstimation(
                window_size=window_size,
                apply_filtering=apply_filter,
                discard_method="trimmed",
                true_mean=true_mean,
                true_congested_mean=true_congested_mean
            )
            monitor.report_interval = float('inf')  # silence frequent logs

            simulator.attach_sketch(network.DESTINATION, monitor)
            simulator.simulate_traffic(duration_seconds=simulation_duration, avg_interarrival_ms=avg_interarrival_ms)

            # Estimate after simulation
            params = monitor.estimate_parameters()
            if params["estimated_lambda"] is None:
                continue

            kl_score = monitor.kl_divergence(true_lambda, params["estimated_lambda"])
            kl_scores.append(kl_score)

        # Aggregate KL divergence results
        if kl_scores:
            avg_kl = np.mean(kl_scores)
            success_rate = np.mean([kl <= target_kl for kl in kl_scores])
            results.append((window_size, avg_kl, success_rate))

            print(f"  Avg KL: {avg_kl:.6f} | Success Rate: {success_rate:.2%}")

            if avg_kl <= target_kl and success_rate >= 0.8:
                print(f"\n✅ Optimal window size under congestion: {window_size}")
                break

    # Final Results
    print(f"\n=== Evolution 2 Results (drop probability: {drop_probability}, filter: {discard_method}) ===")
    print("Window Size | Avg KL Score | Success Rate")
    print("------------------------------------------")
    for w, kl, r in results:
        print(f"{w:11d} | {kl:12.6f} | {r:11.2%}")

    optimal_windows = [w for w, kl, r in results if kl <= target_kl and r >= 0.8]
    if optimal_windows:
        best = min(optimal_windows)
        print(f"\n✅ Smallest window meeting goal: {best}")
    else:
        print(f"\n❌ No window size consistently achieved KL < {target_kl}")

    return results


if __name__ == "__main__":
    evolution_2_congestion()
