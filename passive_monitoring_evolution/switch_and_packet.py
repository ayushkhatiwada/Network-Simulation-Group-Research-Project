import time

class Packet:
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination

    def __repr__(self):
        return f"Packet({self.source} -> {self.destination})"

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