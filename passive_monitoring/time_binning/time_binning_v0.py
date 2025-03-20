import time
import random
import matplotlib.pyplot as plt
from passive_monitoring.time_binning.time_bin_sketch import TimeBinSketch
from passive_monitoring.time_binning.time_bin_monitoring import TimeBinMonitor
from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
from active_monitoring_evolution.ground_truth import GroundTruthNetwork

network = GroundTruthNetwork(paths="1")
passive = PassiveSimulator(network)

tb_monitor = TimeBinMonitor(passive, bin_size=0.1)
tb_monitor.enable_monitoring()

passive.simulate_traffic(duration_seconds=10, avg_interarrival_ms=100)

# After simulation, retrieve the histograms.
source_hist = tb_monitor.get_source_histogram()
dest_hist = tb_monitor.get_destination_histogram()
print("Source (sent) histogram (bin -> count):", source_hist)
print("Destination (received) histogram (bin -> count):", dest_hist)

source_bins = sorted(source_hist.keys())
source_time_bins = [b * tb_monitor.bin_size for b in source_bins]
source_counts = [source_hist[b] for b in source_bins]

dest_bins = sorted(dest_hist.keys())
dest_time_bins = [b * tb_monitor.bin_size for b in dest_bins]
dest_counts = [dest_hist[b] for b in dest_bins]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(source_time_bins, source_counts, width=tb_monitor.bin_size*0.8, color='skyblue', align='edge')
plt.xlabel("Time (seconds)")
plt.ylabel("Packets Sent")
plt.title("Source (Sent) ")

plt.subplot(1, 2, 2)
plt.bar(dest_time_bins, dest_counts, width=tb_monitor.bin_size*0.8, color='salmon', align='edge')
plt.xlabel("Time (seconds)")
plt.ylabel("Packets Received")
plt.title("Destination (Received) ")

plt.tight_layout()
plt.show()
