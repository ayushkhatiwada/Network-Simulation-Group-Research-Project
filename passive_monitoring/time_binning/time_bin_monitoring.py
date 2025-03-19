from passive_monitoring.time_binning.time_bin_sketch import TimeBinSketch
from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
from active_monitoring_evolution.ground_truth import GroundTruthNetwork

class TimeBinMonitor:
    def __init__(self, passive_simulator, bin_size=1.0):
        self.passive = passive_simulator
        self.bin_size = bin_size
        self.source_sketch = TimeBinSketch(bin_size=bin_size)
        self.dest_sketch = TimeBinSketch(bin_size=bin_size)

    def enable_monitoring(self):
        original_transmit = self.passive.network.transmit_packet
        def modified_transmit(packet):
            self.source_sketch.record_event()
            original_transmit(packet)
        self.passive.network.transmit_packet = modified_transmit
        print("Patched transmit_packet to record send events for source monitoring.")

        self.passive.attach_sketch(self.passive.network.DESTINATION, self.dest_sketch)
        print("Attached TimeBinSketch to destination switch for receive monitoring.")

    def get_source_histogram(self):
        return self.source_sketch.get_histogram()

    def get_destination_histogram(self):
        return self.dest_sketch.get_histogram()


if __name__ == '__main__':
    network = GroundTruthNetwork(paths="1")
    passive = PassiveSimulator(network)
    
    tb_monitor = TimeBinMonitor(passive, bin_size=1.0)
    
    tb_monitor.enable_monitoring()
    
    # Run the traffic simulation via the passive simulator, simulate for 10 seconds with an average interarrival time of 100 ms.
    passive.simulate_traffic(duration_seconds=10, avg_interarrival_ms=100)
    
    # After simulation, retrieve and print the histograms.
    print("Source (sent) histogram (bin -> count):", tb_monitor.get_source_histogram())
    print("Destination (received) histogram (bin -> count):", tb_monitor.get_destination_histogram())
