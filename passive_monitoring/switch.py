#TODO:store arrival times for tcp flows, record the arrival of the initiating packet and then match the corresponding response and find delays accordingly
# get the average, variance and percentiles for the network path
# use aggregated delay on a particular graph to be able to understand its performance (leads into making traffic decisions)

import sketch
import time

class Switch:
    def __init__(self, sketch):
        self.sketch = Sketch(2, 2, 42)
        self.flow_table = {}

    def calculate_flow_id(self, packet):
        flow_fields = (
            packet.get('src_ip'),
            packet.get('dst_ip'),
            packet.get('src_port'),
            packet.get('dst_port'),
            packet.get('protocol')
        )
        return hash(flow_fields)

    def process_packet(self, packet, ingress=True):
        flow_id = self.compute_flow_id(packet)
        if flow_id in self.flow_table:
            ingress_timestamp = self.flow_table[flow_id]
            egress_timestamp = time.time() * 1000  
            delay = egress_timestamp - ingress_timestamp
            self.sketch.update(flow_id, delay)
            print(f"Egress: Flow {flow_id} delay computed as {delay:.2f} ms")
            del self.flow_table[flow_id]
        else:
            ingress_timestamp = time.time() * 1000  
            self.flow_table[flow_id] = ingress_timestamp
            packet['ingress_timestamp'] = ingress_timestamp
            print(f"Ingress: Flow {flow_id} recorded at {ingress_timestamp:.2f} ms")
