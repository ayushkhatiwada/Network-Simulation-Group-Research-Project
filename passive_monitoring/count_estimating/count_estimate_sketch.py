import time
from passive_monitoring.passive_monitoring_interface.general_sketch import Sketch

class CountEstimateSketch(Sketch):
    def __init__(self):
        self.samples = []  

    def process_packet(self, packet, switch_id):
        self.samples.append(time.time())

    def record_sample(self):
        self.samples.append(time.time())

    def get_samples(self):
        return self.samples
