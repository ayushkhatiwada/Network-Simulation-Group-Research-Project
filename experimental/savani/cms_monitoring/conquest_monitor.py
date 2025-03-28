from passive_monitoring.passive_monitoring_interface.passive_simulator import PassiveSimulator
from experimental.savani.cms_monitoring.conquest_sketch import ConQuestSketch

class ConQuestMonitor:
    """
    A monitoring system based on a simplified ConQuest sketch.
    
    This monitor instantiates a ConQuestSketch and attaches it to both the source and destination.
    For demonstration purposes, we patch the source transmit function so that each transmitted packet
    also records an event in the source sketch. (In a full ConQuest system, one typically uses
    ingress and egress measurements to reconstruct queuing state.)
    """
    def __init__(self, passive_simulator, T, start_time):
        self.passive = passive_simulator
        self.T = T  # snapshot time window length in seconds
        self.start_time = start_time
        
        self.source_sketch = ConQuestSketch(start_time, T)
        self.dest_sketch = ConQuestSketch(start_time, T)
    
    def enable_monitoring(self):
        # Patch the source transmit function so that each packet is recorded in the source sketch.
        original_transmit = self.passive.network.transmit_packet
        def modified_transmit(packet):
            self.source_sketch.record_event()
            original_transmit(packet)
        self.passive.network.transmit_packet = modified_transmit
        print("Patched transmit_packet to record source events.")
        
        # Attach the destination sketch to the destination switch.
        self.passive.attach_sketch(self.passive.network.DESTINATION, self.dest_sketch)
        print("Attached ConQuestSketch to destination switch.")
    
    def get_source_histogram(self):
        return self.source_sketch.get_histogram()
    
    def get_destination_histogram(self):
        return self.dest_sketch.get_histogram()
