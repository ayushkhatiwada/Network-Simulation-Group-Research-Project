import time
import numpy as np
from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
from active_monitoring_evolution.ground_truth import GroundTruthNetwork
from passive_monitoring.end_host_latency_measurement import EndHostEstimation

def evolution_1_probe_drops():
    """
    Evolution 1: Estimate delay distribution when probes can be dropped.
    Still exponential distribution between 2 nodes.
    Goal: Estimate lambda using only arrived packets. KL divergence < 0.05.
    """
    print("=== Evolution 1: Estimating Under Packet Drop ===")
    
    # Parameters
    target_kl = 0.05
    drop_probability = 0.2  # 20% packet loss
    min_window_size = 10
    max_window_size = 500
    window_increment = 10
    trials_per_window = 5
    simulation_duration = 5  # seconds

    apply_filtering = True
    discard_method = "trimmed"

    # Ground truth distribution (same as Evolution 0)
    true_mean = 0.01  # seconds
    true_lambda = 1 / true_mean

    results = []

    for window_size in range(min_window_size, max_window_size + 1, window_increment):
        print(f"\nTesting window size: {window_size}")
        kl_scores = []

        for trial in range(trials_per_window):
            network = GroundTruthNetwork(paths=1)  # one edge
            simulator = PassiveSimulator(network)

            # Enable packet drops at destination switch
            simulator.set_drop_probability(network.DESTINATION, drop_probability)

            monitor = EndHostEstimation(
                window_size=window_size,
                apply_filtering=apply_filtering,
                discard_method=discard_method,
                true_mean=true_mean  # used for internal KL calc
            )
            monitor.report_interval = float('inf')  # silence per-second print

            simulator.attach_sketch(network.DESTINATION, monitor)

            # Run simulation
            simulator.simulate_traffic(duration_seconds=simulation_duration, avg_interarrival_ms=10)

            # Get estimated λ
            params = monitor.estimate_parameters()
            if params['estimated_lambda'] is None:
                continue

            kl_score = monitor.kl_divergence(true_lambda, params['estimated_lambda'])
            kl_scores.append(kl_score)

        if kl_scores:
            avg_kl = np.mean(kl_scores)
            success_rate = np.mean([kl <= target_kl for kl in kl_scores])
            results.append((window_size, avg_kl, success_rate))

            print(f"  Avg KL: {avg_kl:.6f} | Success Rate: {success_rate:.2%}")

            if avg_kl <= target_kl and success_rate >= 0.8:
                print(f"\n*** Optimal window size under drop conditions - window size: {window_size}, drop probability: {drop_probability} ***")
                break

    # Print summary
    print(f"\n=== Evolution 1 Results (drop probability: {drop_probability}, filter: {discard_method}) ===")
    print("Window Size | Avg KL Score | Success Rate")
    print("------------------------------------------")
    for w, kl, r in results:
        print(f"{w:11d} | {kl:12.6f} | {r:11.2%}")

    good_windows = [w for w, kl, r in results if kl <= target_kl and r >= 0.8]
    if good_windows:
        best = min(good_windows)
        print(f"\n✅ Smallest window meeting goal: {best}")
    else:
        print(f"\n❌ No window consistently achieved KL < {target_kl}")

    return results

if __name__ == "__main__":
    evolution_1_probe_drops()
