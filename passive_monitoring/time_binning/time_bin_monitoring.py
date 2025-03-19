from passive_monitoring.time_binning.time_bin_sketch import TimeBinSketch
from passive_monitoring.passive_monitoring_evolution.passive_simulator import PassiveSimulator
from active_monitoring_evolution.ground_truth import GroundTruthNetwork

class TimeBinMonitor:
    def __init__(self, passive_simulator, bin_size=1.0):
        self.passive = passive_simulator
        self.bin_size = bin_size
        # Create two sketches: one for the source switch and one for the destination switch.
        self.source_sketch = TimeBinSketch(bin_size=bin_size)
        self.dest_sketch = TimeBinSketch(bin_size=bin_size)

    def enable_monitoring(self):
        """
        Enable time binning by attaching the sketches to the switches via the passive simulator.
        """
        self.passive.attach_sketch(self.passive.network.SOURCE, self.source_sketch)
        self.passive.attach_sketch(self.passive.network.DESTINATION, self.dest_sketch)
        print("TimeBinMonitor: Attached TimeBinSketch to both source and destination switches.")

    def get_source_histogram(self):
        return self.source_sketch.get_histogram()

    def get_destination_histogram(self):
        return self.dest_sketch.get_histogram()




if __name__ == '__main__':
    # Create the ground truth network.
    network = GroundTruthNetwork(paths="1")
    
    # Instantiate the passive simulator.
    passive = PassiveSimulator(network)
    
    # Create the time bin monitor (with a bin size of 1 second, for example).
    tb_monitor = TimeBinMonitor(passive, bin_size=1.0)
    tb_monitor.enable_monitoring()
    
    # Run the traffic simulation via the passive simulator.
    passive.simulate_traffic(duration_seconds=10, avg_interarrival_ms=100)
    
    # Retrieve and print the histograms.
    print("Source (sent) histogram (bin -> count):", tb_monitor.get_source_histogram())
    print("Destination (received) histogram (bin -> count):", tb_monitor.get_destination_histogram())
