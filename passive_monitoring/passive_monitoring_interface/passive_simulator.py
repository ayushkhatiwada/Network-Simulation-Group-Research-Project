import time
import random
import math
from active_monitoring_evolution.ground_truth import GroundTruthNetwork

class PassiveSimulator:
    def __init__(self, ground_truth_network: GroundTruthNetwork):
        self.network = ground_truth_network
        self.switches = {
            self.network.SOURCE: self.network.source_switch,
            self.network.DESTINATION: self.network.destination_switch
        }
        # Event log: each entry is a tuple (arrival_time, processed_time, delay), for dropped packets both are None.
        self.event_log = []
        
        # Congestion simulation parameters.
        self.max_simulation_time = 100.0  
        self.normal_drop_probability = 0.1
        self.congested_drop_probability = 0.4
        self.congestion_delay_factor = 1.5 
        
        self.congestion_intervals = []  # list of (start, end) times (relative seconds)
        self.simulation_start_time = None

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
                self.event_log.append((arrival_time, None, None))
                print(f"[Drop] Packet {packet} dropped at switch {node_id} at {arrival_time:.2f} s.")
            else:
                processed_time = time.time()
                delay = processed_time - arrival_time
                self.event_log.append((arrival_time, processed_time, delay))
                original_receive(packet)
        
        switch.receive = modified_receive
        print(f"Set drop probability to {drop_probability*100:.1f}% for switch {node_id}.")

    def _generate_congestion_intervals(self):
        """
        Generate congestion intervals within the total simulation time.
        Randomly partitions 50 seconds of congested time into several intervals.
        """
        congested_durations = [5, 10, 15, 20]  
        random.shuffle(congested_durations)
        total_congested_time = sum(congested_durations)
        total_simulation_time = self.max_simulation_time
        total_non_congested_time = total_simulation_time - total_congested_time
        num_non_congested_intervals = len(congested_durations) + 1
        
        # Randomly partition non-congested time.
        weights = [random.random() for _ in range(num_non_congested_intervals)]
        weight_sum = sum(weights)
        non_congested_durations = [(w / weight_sum) * total_non_congested_time for w in weights]
        
        congestion_intervals = []
        current_time = non_congested_durations[0]
        for i, duration in enumerate(congested_durations):
            start = current_time
            end = start + duration
            congestion_intervals.append((start, end))
            current_time = end + non_congested_durations[i + 1]
        return congestion_intervals

    def enable_congestion_simulation(self, node_id):
        if node_id not in self.switches:
            raise ValueError(f"No switch found for node id {node_id}.")
        
        # Set simulation start time and generate congestion intervals.
        self.simulation_start_time = time.time()
        self.congestion_intervals = self._generate_congestion_intervals()
        print("Generated congestion intervals (relative to simulation start):")
        for start, end in self.congestion_intervals:
            print(f"  {start:.2f} s to {end:.2f} s")
        
        switch = self.switches[node_id]
        original_receive = switch.receive

        def modified_receive(packet):
            rel_time = time.time() - self.simulation_start_time
            in_congestion = any(start <= rel_time <= end for start, end in self.congestion_intervals)
            if in_congestion:
                drop_prob = self.congested_drop_probability
                delay_factor = self.congestion_delay_factor
                state = "congested"
            else:
                drop_prob = self.normal_drop_probability
                delay_factor = 1.0
                state = "normal"

            arrival_time = time.time()
            if random.random() < drop_prob:
                self.event_log.append((rel_time, None, None))
                print(f"[Drop] Packet {packet} dropped at switch {node_id} at {rel_time:.2f} s ({state}).")
            else:
                # If congested, simulate additional delay.
                if delay_factor > 1.0:
                    extra_delay = (delay_factor - 1.0) * 0.01
                    time.sleep(extra_delay)
                processed_time = time.time()
                measured_delay = processed_time - arrival_time
                self.event_log.append((rel_time, processed_time - self.simulation_start_time, measured_delay))
                original_receive(packet)
        
        switch.receive = modified_receive
        print(f"Enabled congestion simulation on switch {node_id}.")

    def simulate_traffic(self, duration_seconds=10, avg_interarrival_ms=10):
        self.network.simulate_traffic(duration_seconds, avg_interarrival_ms)

    def compare_distribution_parameters(self, pred_mean: float, pred_std: float) -> float:
        params = self.network.get_distribution_parameters()[0][2]
        actual_mean = params["mean"]
        actual_std = params["std"]

        kl_div = math.log(pred_std / actual_std) + ((actual_std**2 + (actual_mean - pred_mean)**2) / (2 * pred_std**2)) - 0.5
        
        # Aim for a KL divergence of <= 0.05
        if kl_div <= 0.05:
            print(f"KL divergence: {kl_div:.4f} ✅")
        else:
            print(f"KL divergence: {kl_div:.4f} ❌")
        return kl_div
