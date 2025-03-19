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

    def receive(self, packet):
        print(f"[{time.strftime('%H:%M:%S')}] Switch {self.switch_id} received {packet}")