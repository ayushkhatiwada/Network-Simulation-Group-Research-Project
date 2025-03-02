from collections import deque
import random

class SlidingWindowSketch:
    """
    This sketch keeps track of the latest delays measured
    """
    def __init__(self, window_size=10):
        # window_size: max num of most recent latency vals stored
        self.window_size = window_size
        self.flow_latency = {} # flow_id: deque of x most recent latencies

    def update(self, flow_id, latency) -> None:
        if flow_id not in self.flow_latency:
            self.flow_latency[flow_id] = deque(maxlen=self.window_size)  # Create a new sliding window for new flow
        self.flow_latency[flow_id].append(latency)

    def estimate_delay(self, flow_id) -> list[float, float]: # Returns latency average and variance
        if (flow_id not in self.flow_latency) or len(self.flow_latency[flow_id]) == 0:
            return {'average': None, 'variance': None}  # No data available
        
        latencies = list(self.flow_latency[flow_id])
        n = len(latencies)
        avg = sum(latencies) / n
        variance = sum((x - avg) ** 2 for x in latencies) / n
      
        return [avg, variance]

# Test with only one
latency_sketch = SlidingWindowSketch(window_size=10)
flow_id = "flow_1"
    
# Simulate incoming latency measurements (random normal distribution with mean 50ms, std deviation 10ms)
for _ in range(20):  # Simulate 20 incoming packets
    latency = max(0, random.gauss(50, 10))  # Ensure latency is not negative
    latency_sketch.update(flow_id, latency)
    print(f"Added latency: {latency:.2f} ms")
    
# Retrieve latency statistics
stats = latency_sketch.estimate_delay(flow_id)
print("\nLatency Statistics:")
print(f"Average Latency: {stats[0]:.2f} ms")
print(f"Variance: {stats[1]:.2f} ms")
