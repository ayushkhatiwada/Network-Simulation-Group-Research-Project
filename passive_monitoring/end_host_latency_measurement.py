import random
import numpy as np
from collections import deque

class LatencyEstimator:
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.flow_latency = {}  # stores latencies to each flow_id

    # keep most recent values
    def update(self, flow_id, latency):
        if flow_id not in self.flow_latency:
            self.flow_latency[flow_id] = deque(maxlen=self.window_size)

        self.flow_latency[flow_id].append(latency)
        
        return self.estimate_parameters(flow_id)
    
    # estimates mean and variance of normal distribution
    def estimate_parameters(self, flow_id, apply_filtering=False, discard_method="trimmed", discard_fraction=0.1):

        if (flow_id not in self.flow_latency) or len(self.flow_latency[flow_id]) == 0:
            return {'estimated_mean': None, 'estimated_variance': None}

        latencies = np.array(self.flow_latency[flow_id])

        # filtering to improve variance estimation
        if apply_filtering:
            if discard_method == "trimmed":
                lower_bound = int(len(latencies) * discard_fraction)
                upper_bound = len(latencies) - lower_bound
                latencies = np.sort(latencies)[lower_bound:upper_bound]  # cut off extreme values
            elif discard_method == "median_filter":
                median_value = np.median(latencies)
                deviation = np.abs(latencies - median_value)
                threshold = np.median(deviation) * 2 
                latencies = latencies[deviation < threshold]
            elif discard_method == "threshold":
                mean = np.mean(latencies)
                std_dev = np.std(latencies)
                latencies = latencies[(latencies >= mean - 2 * std_dev) & (latencies <= mean + 2 * std_dev)]

        estimated_mean = np.mean(latencies)
        estimated_variance = np.var(latencies, ddof=1)  # sample variance

        return {'estimated_mean': estimated_mean, 'estimated_variance': estimated_variance}


# normal latency generation
def generate_normal_latencies(size, true_mean=50, true_variance=25):
    return [random.gauss(true_mean, np.sqrt(true_variance)) for _ in range(size)]

# noisy latency generation (temporary shift with certain probability)
def generate_noisy_latencies(size, true_mean=50, true_variance=25, shift_probability=0.2, shift_amount=10):
    latencies = []
    for _ in range(size):
        if random.random() < shift_probability:
            mean_shift = random.choice([-shift_amount, shift_amount])  # shifts up or down randomly
            latencies.append(random.gauss(true_mean + mean_shift, np.sqrt(true_variance)))
        else:
            latencies.append(random.gauss(true_mean, np.sqrt(true_variance)))
    return latencies

def find_optimal_window_size(target_accuracy=0.80, max_window_size=500, trials_per_window=50, noise=False, apply_filtering=False, discard_method="trimmed"):
    """
    Finds the smallest window size that achieves at least 80% accuracy
    when estimating the mean and variance.
    
    - noise: If True, uses noisy latency generation.
    - apply_filtering: If True, applies a filtering method to remove extreme values.
    """
    true_mean = 50  # ground truth mean
    true_variance = 25  # ground truth variance

    min_window = 50
    increments = 50
    optimal_window_size = None

    for window_size in range(min_window, max_window_size + 1, increments):  # Incrementally increase window size
        print(f"\n=== Testing Window Size: {window_size} | Noise: {noise} | Filtering: {apply_filtering} ({discard_method}) ===")
        mse_results = []

        for _ in range(trials_per_window):
            if noise:
                latencies = generate_noisy_latencies(window_size)
            else:
                latencies = generate_normal_latencies(window_size)

            estimator = LatencyEstimator(window_size)
            for latency in latencies:
                estimator.update("flow_1", latency)

            result = estimator.estimate_parameters("flow_1", apply_filtering=apply_filtering, discard_method=discard_method)

            if result['estimated_mean'] is not None and result['estimated_variance'] is not None:
                mse_mean = (result['estimated_mean'] - true_mean) ** 2

                variance_error = abs(result['estimated_variance'] - true_variance)
                variance_accuracy = max(0, 1 - (variance_error / true_variance))  # variance accuracy counded between 0, 1

                mse_results.append((mse_mean, variance_accuracy))

        avg_mse_mean = np.mean([mse[0] for mse in mse_results])
        avg_mse_variance = np.mean([mse[1] for mse in mse_results])
        print(avg_mse_variance, true_variance)
        mean_accuracy = 1 - (avg_mse_mean / true_variance)  # normalize accuracy

        print(f"  Accuracy (Mean): {mean_accuracy:.2%}")
        print(f"  Accuracy (Variance): {variance_accuracy:.2%}")

        if mean_accuracy >= target_accuracy and variance_accuracy >= target_accuracy:
            optimal_window_size = window_size
            break  

    if optimal_window_size:
        print(f"\n Smallest window size for 80% accuracy in mean and variance: {optimal_window_size}")
    else:
        print("\n No window size could achieve 80% accuracy rate.")

    return optimal_window_size

if __name__ == "__main__":
    TARGET_ACCURACY = 0.80
    MAX_WINDOW_SIZE = 500
    TRIALS_PER_WINDOW = 50

    print("\n=== Running Round 1: Standard Normal Latency (No Filtering) ===")
    optimal_w_normal = find_optimal_window_size(TARGET_ACCURACY, MAX_WINDOW_SIZE, TRIALS_PER_WINDOW, noise=False, apply_filtering=False)

    print("\n=== Running Round 1.5: Standard Normal Latency (With Filtering) ===")
    optimal_w_normal = find_optimal_window_size(TARGET_ACCURACY, MAX_WINDOW_SIZE, TRIALS_PER_WINDOW, noise=False, apply_filtering=True, discard_method="trimmed")

    print("\n=== Running Round 2: Noisy Latency (No Filtering) ===")
    optimal_w_noisy_raw = find_optimal_window_size(TARGET_ACCURACY, MAX_WINDOW_SIZE, TRIALS_PER_WINDOW, noise=True, apply_filtering=False)

    print("\n=== Running Round 2.5: Noisy Latency (With Filtering) ===")
    optimal_w_noisy_filtered = find_optimal_window_size(TARGET_ACCURACY, MAX_WINDOW_SIZE, TRIALS_PER_WINDOW, noise=True, apply_filtering=True, discard_method="trimmed")
