import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
from active_monitoring_evolution.ground_truth import GroundTruthNetwork
from experimental.savani.wave_sketch_monitoring.wave_monitor import WaveMonitor  # Assumes WaveSketch and WaveMonitor are in separate files

def extract_delays(event_log):
    """
    Extract per-packet delays from the simulator's event log.
    Each event is a tuple (arrival_time, processed_time, delay).
    Only packets that were processed (not dropped) have a non-None delay.
    """
    delays = []
    for entry in event_log:
        if entry[2] is not None:
            delays.append(entry[2])
    return delays

if __name__ == '__main__':
    # -------------------------------
    # Setup network and simulator.
    # -------------------------------
    network = GroundTruthNetwork(paths="1")
    passive = PassiveSimulator(network)
    
    # Even in drop-free mode, we want to patch the destination receive function so that delays are logged.
    # Set drop probability to 0 (i.e. no drops) to ensure logging.
    passive.set_drop_probability(network.DESTINATION, 0.0)
    
    # -------------------------------
    # Configure WaveSketch-based monitoring.
    # -------------------------------
    # Use a fine bin size to capture microsecond-level variations.
    bin_size = 0.00001  # seconds per bin (e.g., 10 µs)
    start_time = time.time()
    wave_monitor = WaveMonitor(passive, bin_size, start_time)
    wave_monitor.enable_monitoring()
    
    # -------------------------------
    # Simulate traffic.
    # -------------------------------
    simulation_duration = 10  # seconds
    avg_interarrival_ms = 10  # ms
    passive.simulate_traffic(duration_seconds=simulation_duration, avg_interarrival_ms=avg_interarrival_ms)
    
    # -------------------------------
    # Retrieve WaveSketch signal (for flow rate analysis).
    # -------------------------------
    signal = wave_monitor.get_signal()
    time_axis = np.arange(len(signal)) * bin_size
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, signal, marker='o', linestyle='-', label='WaveSketch Signal')
    plt.xlabel("Time (s)")
    plt.ylabel("Flow Rate (count per bin)")
    plt.title("WaveSketch Flow Rate Signal (Normal Network)")
    plt.legend()
    plt.show()
    
    # -------------------------------
    # Extract per-packet delays from the event log.
    # -------------------------------
    delays = extract_delays(passive.event_log)  # delays in seconds
    if not delays:
        print("No delays recorded! Check your simulation or logging setup.")
        exit(0)
    delays_ms = [d * 1000 for d in delays]  # convert to milliseconds
    
    # -------------------------------
    # Fit a normal distribution to the measured delays.
    # -------------------------------
    est_mu, est_std = norm.fit(delays_ms)
    print("Estimated normal fit parameters (ms):", est_mu, est_std)
    
    # Retrieve the true underlying delay distribution from the network.
    # (Assume network.get_distribution_parameters()[0][2] returns a dict with keys "mean" and "std", in ms.)
    true_params = network.get_distribution_parameters()[0][2]
    true_mu = true_params["mean"]
    true_std = true_params["std"]
    print("True underlying distribution (ms):", true_mu, true_std)
    
    # Compute KL divergence.
    kl_div = passive.compare_distribution_parameters(est_mu, est_std)
    print("KL divergence:", kl_div)
    
    # -------------------------------
    # Plot both PDFs on the same graph.
    # -------------------------------
    lower_bound = min(est_mu - 4*est_std, true_mu - 4*true_std)
    upper_bound = max(est_mu + 4*est_std, true_mu + 4*true_std)
    x = np.linspace(lower_bound, upper_bound, 200)
    
    est_pdf = norm.pdf(x, est_mu, est_std)
    true_pdf = norm.pdf(x, true_mu, true_std)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x, est_pdf, 'r-', linewidth=2,
             label=f"Estimated Normal\nμ={est_mu:.4f} ms, σ={est_std:.4f} ms")
    plt.plot(x, true_pdf, 'b--', linewidth=2,
             label=f"True Normal\nμ={true_mu:.4f} ms, σ={true_std:.4f} ms")
    plt.xlabel("Delay (ms)")
    plt.ylabel("Probability Density")
    plt.title("Estimated vs. True Delay Distributions (Normal Network)")
    plt.legend()
    plt.show()
