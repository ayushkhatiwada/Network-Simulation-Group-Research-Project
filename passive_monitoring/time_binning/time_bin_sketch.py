import time
from passive_monitoring.passive_monitoring_interface.general_sketch import Sketch

class TimeBinSketch(Sketch):
    def __init__(self, bin_size=0.1):
        self.bin_size = bin_size
        self.start_time = time.time()
        self.bins = {}  # Dictionary mapping bin index to count

    def process_packet(self, packet, switch_id):
        now = time.time()
        bin_index = int((now - self.start_time) // self.bin_size)
        self.bins[bin_index] = self.bins.get(bin_index, 0) + 1

    def record_event(self):
        now = time.time()
        bin_index = int((now - self.start_time) // self.bin_size)
        self.bins[bin_index] = self.bins.get(bin_index, 0) + 1


    def get_histogram(self):
        return self.bins
