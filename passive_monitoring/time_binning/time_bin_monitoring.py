import time
import random
from passive_monitoring.time_binning.time_bin_sketch import TimeBinSketch
from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
from active_monitoring_evolution.ground_truth import GroundTruthNetwork

class TimeBinMonitor:
    def __init__(self, passive_simulator, bin_size, start_time):
        self.passive = passive_simulator
        self.bin_size = bin_size
        self.source_sketch = TimeBinSketch(start_time, bin_size)
        self.dest_sketch = TimeBinSketch(start_time, bin_size)

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

