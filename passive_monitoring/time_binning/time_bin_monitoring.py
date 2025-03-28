import time
from passive_monitoring.time_binning.time_bin_sketch import TimeBinSketch
from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator

class TimeBinMonitor:
    def __init__(self, passive_simulator: PassiveSimulator, bin_size, start_time):
        self.passive = passive_simulator
        self.bin_size = bin_size
        self.start_time = start_time
        
        # Discrete (fixed) sketches.
        self.source_sketch = TimeBinSketch(start_time, bin_size)
        self.dest_sketch = TimeBinSketch(start_time, bin_size)
        
        # Sliding sketches: we use an offset (here half the bin size).
        self.sliding_offset = bin_size / 2
        self.source_sliding_sketch = TimeBinSketch(start_time + self.sliding_offset, bin_size)
        self.dest_sliding_sketch = TimeBinSketch(start_time + self.sliding_offset, bin_size)
    
    def enable_monitoring(self):
        # Patch the source transmit method to record events in both discrete and sliding sketches.
        original_transmit = self.passive.network.transmit_packet
        def modified_transmit(packet):
            self.source_sketch.record_event()
            self.source_sliding_sketch.record_event()
            original_transmit(packet)
        self.passive.network.transmit_packet = modified_transmit
        print("Patched transmit_packet to record send events for source monitoring.")
        
        # Attach both destination sketches to the destination switch.
        self.passive.attach_sketch(self.passive.network.DESTINATION, self.dest_sketch)
        self.passive.attach_sketch(self.passive.network.DESTINATION, self.dest_sliding_sketch)
        print("Attached TimeBinSketches (discrete & sliding) to destination switch for receive monitoring.")
    
    def get_source_histogram(self):
        return self.source_sketch.get_histogram()
    
    def get_destination_histogram(self):
        return self.dest_sketch.get_histogram()
    
    def get_source_sliding_histogram(self):
        return self.source_sliding_sketch.get_histogram()
    
    def get_destination_sliding_histogram(self):
        return self.dest_sliding_sketch.get_histogram()
