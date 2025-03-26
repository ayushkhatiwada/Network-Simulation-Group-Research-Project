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
monitor.enable_monitoring(fluctuation_mean=true_mean, fluctuation_std=true_std, drop_probability=0.2)  #TODO: ADD DROPS
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





