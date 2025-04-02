import time
import random
import string

ALLOWED_FLOW_IDS = ["A", "B", "C", "D"]

def get_flow_id(probabilities=None):
    if probabilities is None:
        probabilities = [1/len(ALLOWED_FLOW_IDS)] * len(ALLOWED_FLOW_IDS)
    return random.choices(ALLOWED_FLOW_IDS, weights=probabilities, k=1)[0]

class Packet:
    def __init__(self, source, destination, probabilities=None):
        self.source = source
        self.destination = destination
        self.flow_id = get_flow_id(probabilities)
        self.packet_id = self.generate_packet_id()
        self.payload = self.generate_payload()
    def generate_packet_id(self):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    def generate_payload(self):
        return random.randint(100, 1000)
    def get_attributes(self):
        return {
            "packet_id": self.packet_id,
            "flow_id": self.flow_id,
            "source": self.source,
            "destination": self.destination,
            "payload": self.payload
        }
    def __repr__(self):
        return f"Packet({self.source}->{self.destination}, id={self.packet_id}, flow={self.flow_id})"

class Switch:
    def __init__(self, switch_id):
        self.switch_id = switch_id
        self.sketches = []

    def add_sketch(self, sketch):
        self.sketches.append(sketch)

    def receive(self, packet):
        for s in self.sketches:
            s.process_packet(packet, self.switch_id)
        print(f"[{time.strftime('%H:%M:%S')}] Switch {self.switch_id} received {packet}")