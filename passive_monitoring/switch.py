#TODO:store arrival times for tcp flows, record the arrival of the initiating packet and then match the corresponding response and find delays accordingly
# get the average, variance and percentiles for the network path
# use aggregated delay on a particular graph to be able to understand its performance (leads into making traffic decisions)

from sketch import Sketch
import time

class Switch:
    def __init__(self):
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
        flow_id = self.calculate_flow_id(packet)
        if role == "ingress":
            ingress_timestamp = time.time() * 1000 
            self.flow_table[flow_id] = ingress_timestamp
            packet['global_ingress_timestamp'] = ingress_timestamp
            print(f"Switch {id(self)} - Ingress: Flow {flow_id} stamped at {ingress_timestamp:.2f} ms")
        elif role == "forward":
            print(f"Switch {id(self)} - Forwarding packet for flow {flow_id}.")
        elif role == "egress":
            if 'global_ingress_timestamp' in packet:
                egress_timestamp = time.time() * 1000  # in ms
                delay = egress_timestamp - packet['global_ingress_timestamp']
                self.sketch.update(flow_id, delay)
                print(f"Switch {id(self)} - Egress: Flow {flow_id} delay computed as {delay:.2f} ms")
