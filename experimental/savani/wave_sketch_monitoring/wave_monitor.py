from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
from experimental.savani.wave_sketch_monitoring.wave_sketch import WaveSketch

class WaveMonitor:
    """
    A simple monitoring system based on WaveSketch.
    
    This monitor instantiates a WaveSketch and attaches it to the specified switch
    (e.g., the source switch). It can then retrieve the flow rate signal.
    """
    def __init__(self, passive_simulator: PassiveSimulator, bin_size, start_time):
        self.passive = passive_simulator
        self.bin_size = bin_size
        self.start_time = start_time
        self.wave_sketch = WaveSketch(start_time, bin_size)
    
    def enable_monitoring(self):
        # Attach the WaveSketch to the source switch.
        self.passive.attach_sketch(self.passive.network.SOURCE, self.wave_sketch)
    
    def get_signal(self):
        return self.wave_sketch.get_signal()
