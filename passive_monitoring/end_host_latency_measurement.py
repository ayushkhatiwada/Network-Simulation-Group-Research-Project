from collections import deque
import random
import numpy as np
import scipy.stats as stats

class EndHostLatencyMeasurement:
    """
    This sketch keeps track of the latest delays measured, returns avg and variance
    """
    def __init__(self, selection_criterion, window_size=10):
        # window_size: max num of most recent latency vals stored
        self.window_size = window_size
        self.flow_latency = {} # flow_id: deque of x most recent latencies
        self.selection_criterion = selection_criterion

    def update(self, flow_id, latency) -> dict:
        if flow_id not in self.flow_latency:
            self.flow_latency[flow_id] = deque(maxlen=self.window_size)  # Create a new sliding window for new flow
        self.flow_latency[flow_id].append(latency)
        return self.estimate_delay(flow_id)

    def estimate_delay(self, flow_id) -> dict: # Returns latency average and variance
        if (flow_id not in self.flow_latency) or len(self.flow_latency[flow_id]) == 0:
            return {'average': None, 'variance': None}  # No data available
        
        latencies = list(self.flow_latency[flow_id])
        # print(f'num of latency measurements: {len(latencies)}')
        n = len(latencies)
        avg = sum(latencies) / n
        variance = sum((x - avg) ** 2 for x in latencies) / n
        best_fit, p_value = None, None

        result = self.fit_best_distribution(latencies)
        
        if result is not None and result is not (None, None):
            best_fit, p_value = result
            best_fit = best_fit['distribution']
      
        return {'average': avg,
                'variance': variance,
                'best_fit': best_fit, 
                'p_value': p_value}
    
    def fit_best_distribution(self, latencies):
        """
        Attempts to fit multiple distributions and returns the best-fitting one.
        """
        distributions = [stats.norm, stats.expon, stats.lognorm]
        best_dist = None
        best_params = None
        best_p_value = -1  # Track the highest p-value
        best_ks_stat = float("inf")


        latencies = np.array(latencies)

        for distribution in distributions:
            try: 
                params = distribution.fit(latencies)  # Fit distribution to data
                ks_stat, p_value = stats.kstest(latencies, lambda x: distribution.cdf(x, *params))

                # Selection logic based on the chosen criterion
                if self.selection_criterion == 'p_value' and p_value > best_p_value:
                    best_p_value = p_value
                    best_dist = distribution
                    best_params = params
                    best_ks_stat = ks_stat  # Keep track for reference

                elif self.selection_criterion == 'ks_stat' and ks_stat < best_ks_stat:
                    best_ks_stat = ks_stat
                    best_dist = distribution
                    best_params = params
                    best_p_value = p_value  # Keep track for reference
                
            except (stats._warnings_errors.FitError, ValueError, RuntimeError) as e:
                print(f"Skipping {distribution.name} due to fitting error: {e}")
                continue  # Skip this distribution and try the next one

        if best_dist:
            return {'distribution': best_dist.name, 'parameters': best_params}, best_p_value
        return None

def test_latency_estimation(window_sizes, distributions, selection_criterion, trials_per_window=50):
    """
    Runs tests with different latency distributions across different windows.
    """
    results = {}

    for window_size in window_sizes: 
        print(f"\n=== Testing Window Size: {window_size} === | Criterion: {selection_criterion} ===")
        accuracy_results = {}

        for true_dist_name, generator in distributions.items():
            correct_predictions = 0

            for _ in range(trials_per_window):
                # Generate latency samples using the current distribution
                latencies = generator(window_size)

                estimator = EndHostLatencyMeasurement(selection_criterion, window_size)
                for latency in latencies:
                    estimator.update("flow_1", latency)

                result = estimator.estimate_delay("flow_1")

                if result['best_fit'] == true_dist_name:
                    correct_predictions += 1

            accuracy = correct_predictions / trials_per_window
            accuracy_results[true_dist_name] = accuracy
            print(f"  {true_dist_name}: Accuracy = {accuracy:.2%}")

        results[window_size] = accuracy_results

    return results

# === RUN TESTS ===
if __name__ == "__main__":
    # WINDOW_SIZES = [10, 20, 50, 100, 150, 200]  # Different window sizes to test
    WINDOW_SIZES = list(range(50, 501, 50))
    TRIALS_PER_WINDOW = 50  # Number of trials per window size
    SELECTION_CRITERION = 'ks_stat' # p_value or ks_stat
    DISTRIBUTIONS = {
        "norm": lambda size: [max(0, random.gauss(50, 10)) for _ in range(size)],
        "expon": lambda size: [random.expovariate(1/50) for _ in range(size)],
        # "gamma": lambda size: [random.gammavariate(2, 25) for _ in range(size)],
        "lognorm": lambda size: [random.lognormvariate(3.5, 0.5) for _ in range(size)],
        # "weibull_min": lambda size: [random.weibullvariate(50, 1.5) for _ in range(size)],
        # "pareto": lambda size: [random.paretovariate(2) * 10 for _ in range(size)]
    }

    test_results = test_latency_estimation(WINDOW_SIZES, DISTRIBUTIONS, SELECTION_CRITERION, TRIALS_PER_WINDOW)

    print("\n=== Final Accuracy Results ===")
    for window_size, accuracies in test_results.items():
        print(f"\nWindow Size: {window_size}")
        for dist, acc in accuracies.items():
            print(f"  {dist}: {acc:.2%}")


# === SIMULATION OF REAL-TIME UPDATES ===
# if __name__ == "__main__":
#     window_size = 50  # Maximum number of latencies to track
#     num_updates = 100  # Number of updates to simulate
#     plot_every = 20  # Plot every 20 updates

#     sketch = EndHostLatencyMeasurement(window_size=window_size)
#     flow_id = "flow_1"

#     print("\n=== Real-Time Latency Distribution Estimation ===\n")

#     for i in range(1, num_updates + 1):
#         # Simulate receiving a new latency measurement
#         latency = max(0, random.gauss(50, 10))  # True underlying distribution is Normal
#         stats_out = sketch.update(flow_id, latency)

#         # Print real-time update
#         print(f"[Update {i}] Latency: {latency:.2f} ms | Best Fit: {stats_out['best_fit']} | p-value: {stats_out['p_value']:.3f}")

