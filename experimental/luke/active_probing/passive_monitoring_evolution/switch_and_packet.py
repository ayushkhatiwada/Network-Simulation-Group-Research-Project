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

    def receive(self, packet, virtual_time=None):
        """
        Process the received packet.
        
        Parameters:
        -----------
        packet: Packet
            The packet to be processed
        virtual_time: float, optional
            The virtual time at which the packet is received
        """
        for s in self.sketches:
            s.process_packet(packet, self.switch_id)
        print(f"[{time.strftime('%H:%M:%S')}] Switch {self.switch_id} received {packet} at virtual time {virtual_time}")