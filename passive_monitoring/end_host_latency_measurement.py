import random
import numpy as np
from collections import deque

class LatencyEstimator:
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.flow_latency = {}  # stores latencies to each flow_id

    def update(self, flow_id, latency):
        if flow_id not in self.flow_latency:
            self.flow_latency[flow_id] = deque(maxlen=self.window_size)

        self.flow_latency[flow_id].append(latency)
        
        return self.estimate_parameters(flow_id)

    def estimate_parameters(self, flow_id):
        """
        Estimates the mean and variance of latencies, assuming it follows the normal distribution
        """
        if (flow_id not in self.flow_latency) or len(self.flow_latency[flow_id]) == 0:
            return {'estimated_mean': None, 'estimated_variance': None}

        latencies = np.array(self.flow_latency[flow_id])

        estimated_mean = np.mean(latencies)
        estimated_variance = np.var(latencies, ddof=1)

        return {'estimated_mean': estimated_mean, 'estimated_variance': estimated_variance}

def find_optimal_window_size(target_accuracy=0.95, max_window_size=500, trials_per_window=50):
    """
    Finds the smallest window size that achieves at least 80% accuracy when estimating the mean and variance.
    """
    true_mean = 50  # ground truth mean
    true_variance = 25  # ground truth variance

    min_window = 50
    increments = 50
    optimal_window_size = None

    for window_size in range(min_window, max_window_size + 1, increments):
        print(f"\n=== Testing Window Size: {window_size} ===")
        results = []

        for _ in range(trials_per_window):
            # generate ground truth latency samples
            latencies = [random.gauss(true_mean, np.sqrt(true_variance)) for _ in range(window_size)]

            estimator = LatencyEstimator(window_size)
            for latency in latencies:
                estimator.update("flow_1", latency)

            result = estimator.estimate_parameters("flow_1")

            if result['estimated_mean'] is not None and result['estimated_variance'] is not None:
                mse_mean = (result['estimated_mean'] - true_mean) ** 2
                mse_variance = (result['estimated_variance'] - true_variance) ** 2
                results.append((mse_mean, mse_variance))

        avg_mse_mean = np.mean([mse[0] for mse in results])
        avg_mse_variance = np.mean([mse[1] for mse in results])
        mean_accuracy = 1 - (avg_mse_mean / true_variance)  # normalization
        variance_accuracy = 1 - (avg_mse_variance / true_variance)

        print(f"  Accuracy (Mean): {mean_accuracy:.2%}")
        print(f"  Accuracy (Variance): {variance_accuracy:.2%}")

        # stop early if 80% accuracy reached
        if mean_accuracy >= target_accuracy and variance_accuracy >= target_accuracy:
            optimal_window_size = window_size
            break 

    if optimal_window_size:
        print(f"\n Smallest window size for 80% accuracy in mean and variance: {optimal_window_size}")
    else:
        print("\n No window size could achieve 80% accuracy rate.")

    return optimal_window_size

# === RUN TESTS ===
if __name__ == "__main__":
    target_accuracy = 0.8
    max_window = 500
    trials = 100

    optimal_w = find_optimal_window_size(target_accuracy, max_window, trials)
