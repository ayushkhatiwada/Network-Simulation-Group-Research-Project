# import time
# import random
# import matplotlib.pyplot as plt
# import numpy as np
# from passive_monitoring.time_binning.time_bin_sketch import TimeBinSketch
# from passive_monitoring.time_binning.time_bin_monitoring import TimeBinMonitor
# from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
# from active_monitoring_evolution.ground_truth import GroundTruthNetwork


# if __name__ == '__main__':
#     network = GroundTruthNetwork(paths="1")
#     passive = PassiveSimulator(network)
    
#     tb_monitor = TimeBinMonitor(passive, bin_size=0.001)
#     tb_monitor.enable_monitoring()
    
#     passive.simulate_traffic(duration_seconds=10, avg_interarrival_ms=100)
    
#     source_hist = tb_monitor.get_source_histogram()
#     dest_hist = tb_monitor.get_destination_histogram()
#     print("Source (sent) histogram (bin -> count):", source_hist)
#     print("Destination (received) histogram (bin -> count):", dest_hist)

    
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from passive_monitoring.time_binning.time_bin_sketch import TimeBinSketch
from passive_monitoring.time_binning.time_bin_monitoring import TimeBinMonitor
from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
from active_monitoring_evolution.ground_truth import GroundTruthNetwork

def compute_delay_distribution(source_hist, dest_hist, bin_size):
    """
    Expand the histograms into lists of bin indices for each event,
    pair up the events, and compute delays as the difference in bin indices times the bin size.
    """
    source_events = []
    for bin_index in sorted(source_hist.keys()):
        source_events.extend([bin_index] * source_hist[bin_index])
    
    dest_events = []
    for bin_index in sorted(dest_hist.keys()):
        dest_events.extend([bin_index] * dest_hist[bin_index])
    
    # In case some packets are dropped, only pair up to the minimum count.
    n = min(len(source_events), len(dest_events))
    delays = [(dest_events[i] - source_events[i]) * bin_size for i in range(n)]
    return delays

if __name__ == '__main__':
    # Set up the network and simulator.
    network = GroundTruthNetwork(paths="1")
    passive = PassiveSimulator(network)
    
    tb_monitor = TimeBinMonitor(passive, bin_size=0.00001, start_time=time.time())
    tb_monitor.enable_monitoring()
    
    # Simulate traffic for 10 seconds with an average interarrival time of 10 ms.
    passive.simulate_traffic(duration_seconds=10, avg_interarrival_ms=10)
    
    # Retrieve the source and destination histograms.
    source_hist = tb_monitor.get_source_histogram()
    dest_hist = tb_monitor.get_destination_histogram()
    print("Source (sent) histogram (bin -> count):", source_hist)
    print("Destination (received) histogram (bin -> count):", dest_hist)
    
    # Compute the delay distribution in seconds.
    delays = compute_delay_distribution(source_hist, dest_hist, tb_monitor.bin_size)
    print("Computed delays (in seconds):", delays)
    
    # Convert delays to milliseconds.
    delays_ms = [d * 1000 for d in delays]
    
    # Plot the delay distribution in milliseconds.
    plt.figure(figsize=(8, 5))
    n, bins, patches = plt.hist(delays_ms, bins=20, density=True, alpha=0.6, edgecolor='black')
    plt.xlabel('Delay (ms)')
    plt.ylabel('Probability Density')
    plt.title('Delay Distribution from Source and Destination Histograms (ms)')
    
    # Fit a normal distribution to the delays in ms and plot it.
    if delays_ms:
        mu_ms, std_ms = norm.fit(delays_ms)
        # Compare distribution parameters, now in ms:
        print("Comparing distribution parameters in ms:")
        kl_div = passive.compare_distribution_parameters(mu_ms, std_ms)
        print(f"KL divergence: {kl_div}")
        
        # Plot the fitted normal curve
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu_ms, std_ms)
        plt.plot(x, p, 'r', linewidth=2, 
                 label=f'Normal Fit\nμ={mu_ms:.4f} ms\nσ={std_ms:.4f} ms')
        plt.legend()
    
    plt.show()
