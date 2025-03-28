import time
from passive_monitoring.passive_monitoring_interface.general_sketch import Sketch


class ConQuestSketch(Sketch):
    """
    A very simplified ConQuest-style sketch.
    
    This sketch divides time (using the local clock) into fixed-length snapshots of duration T.
    For each departing packet (as processed in the data plane), it increments a counter in the
    corresponding snapshot (based on the current time). In a full ConQuest design, each snapshot
    would be a compact data structure (e.g., a Count-Min Sketch) indexed by flow IDs.
    
    Here, for simplicity, we simply record the total packet count per snapshot.
    """
    def __init__(self, start_time, T):
        self.start_time = start_time  # local clock start time
        self.T = T  # snapshot time window length (seconds)
        self.snapshots = {}  # mapping: snapshot index -> count
    
    def process_packet(self, packet, switch_id):
        now = time.time()
        # Determine which snapshot the packet belongs to.
        snapshot_idx = int((now - self.start_time) / self.T)
        # Update the counter for this snapshot.
        if snapshot_idx not in self.snapshots:
            self.snapshots[snapshot_idx] = 0
        self.snapshots[snapshot_idx] += 1

    def record_event(self):
        # For compatibility if needed.
        self.process_packet(None, None)

    def get_histogram(self):
        return self.snapshots
