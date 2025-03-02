from collections import deque

class SlidingWindowSketch:
    """
    This sketch keeps track of the latest delays measured
    """
    def __init__(self, window_size=10): 
        # window_size: max num of most recent latency vals stored
        self.window_size = window_size
        self.flow_latency = {} # flow_id: deque of x most recent latencies

    def update(self, flow_id, latency):
        if flow_id not in self.flow_latency:
            self.flow_latency[flow_id] = deque(maxlen=self.window_size)  # Create a new sliding window for new flow
        self.flow_latency[flow_id].append(latency)

