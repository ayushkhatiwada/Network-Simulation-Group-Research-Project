# passive_monitoring_evolution/monitoring_interface.py

import time
import random
from active_monitoring_evolution.ground_truth import GroundTruthNetwork

class PassiveMonitoringInterface:
    def __init__(self, ground_truth_network: GroundTruthNetwork):
        self.network = ground_truth_network
        self.switches = {
            self.network.SOURCE: self.network.source_switch,
            self.network.DESTINATION: self.network.destination_switch
        }
        # Event log: each entry is a tuple (arrival_time, processed_time, delay and for dropped packets both are None.
        self.event_log = []

    def attach_sketch(self, node_id, sketch):
        if node_id in self.switches:
            self.switches[node_id].add_sketch(sketch)
            print(f"Attached {sketch.__class__.__name__} to switch {node_id}.")
        else:
            raise ValueError(f"No switch found for node id {node_id}.")

    def set_drop_probability(self, node_id, drop_probability: float):
        if node_id not in self.switches:
            raise ValueError(f"No switch found for node id {node_id}.")
        
        switch = self.switches[node_id]
        original_receive = switch.receive

        def modified_receive(packet):
            arrival_time = time.time()
            if random.random() < drop_probability:
                # Log the drop event (arrival_time, None, None) and print a message.
                self.event_log.append((arrival_time, None, None))
                print(f"[Drop] Packet {packet} dropped at switch {node_id} at {arrival_time:.2f} s.")
            else:
                processed_time = time.time()
                delay = processed_time - arrival_time
                self.event_log.append((arrival_time, processed_time, delay))
                original_receive(packet)
        
        switch.receive = modified_receive
        print(f"Set drop probability to {drop_probability*100:.1f}% for switch {node_id}.")

    def simulate_traffic(self, duration_seconds=10, avg_interarrival_ms=50):
        self.network.simulate_traffic(duration_seconds, avg_interarrival_ms)
