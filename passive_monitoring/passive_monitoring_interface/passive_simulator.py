import time
import random
import math
from active_monitoring_evolution.ground_truth import GroundTruthNetwork

class PassiveSimulator:
    def __init__(self, ground_truth_network: GroundTruthNetwork, seed=None):
        self.network = ground_truth_network
        self.switches = {
            self.network.SOURCE: self.network.source_switch,
            self.network.DESTINATION: self.network.destination_switch
        }
        self.event_log = []
        
        # Congestion simulation parameters.
        self.max_simulation_time = 100.0  
        self.normal_drop_probability = 0.1
        self.congested_drop_probability = 0.4
        self.congestion_delay_factor = 1.5 
        
        self.congestion_intervals = []  # List of (start, end) times (relative seconds)
        self.simulation_start_time = None
        
        # For variable congestion, we add intensities.
        self.congestion_intensities = {}  # Dict mapping congestion interval index to intensity
        
        # Seed handling: use a local RNG instance for reproducibility
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else random

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
            if self.rng.random() < drop_probability:
                self.event_log.append((arrival_time, None, None))
            else:
                processed_time = time.time()
                delay = processed_time - arrival_time
                self.event_log.append((arrival_time, processed_time, delay))
                original_receive(packet)
        
        switch.receive = modified_receive
        print(f"Set drop probability to {drop_probability*100:.1f}% for switch {node_id}.")

    def set_congestion_parameters(self, node_id, congestion_level=0.5, buffer_size=16):
        print(f"Congestion parameters set: level={congestion_level}, buffer_size={buffer_size}")
        # Currently these parameters aren't used, but the method exists if we want to build on evolution v3 which was abandoned

    def _generate_congestion_intervals(self, simulation_duration=None):
        # If no specific duration provided, use the default max simulation time
        if simulation_duration is None:
            simulation_duration = self.max_simulation_time
        
        # Calculate what proportion of the simulation should be congested
        congestion_ratio = 0.5  # 50% congestion time
        
        # Scale the total congested time proportionally to the actual simulation duration
        total_congested_time = simulation_duration * congestion_ratio
        
        # Scale the congestion segment durations proportionally
        if simulation_duration < 10:
            congested_durations = [total_congested_time]
        elif simulation_duration < 20:
            congested_durations = [total_congested_time * 0.4, total_congested_time * 0.6]
        else:
            original_durations = [5, 10, 15, 20]
            scale_factor = total_congested_time / sum(original_durations)
            congested_durations = [d * scale_factor for d in original_durations]
        
        self.rng.shuffle(congested_durations)
        
        total_non_congested_time = simulation_duration - total_congested_time
        num_non_congested_intervals = len(congested_durations) + 1
        
        weights = [self.rng.random() for _ in range(num_non_congested_intervals)]
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

    def enable_congestion_simulation(self, node_id, simulation_duration=None):
        if node_id not in self.switches:
            raise ValueError(f"No switch found for node id {node_id}.")
        
        self.simulation_start_time = time.time()
        self.congestion_intervals = self._generate_congestion_intervals(simulation_duration)
        print("Generated congestion intervals (fixed intensity):")
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
            if self.rng.random() < drop_prob:
                self.event_log.append((rel_time, None, None))
            else:
                if delay_factor > 1.0:
                    extra_delay = (delay_factor - 1.0) * 0.01
                    time.sleep(extra_delay)
                processed_time = time.time()
                measured_delay = processed_time - arrival_time
                self.event_log.append((rel_time, processed_time - self.simulation_start_time, measured_delay))
                original_receive(packet)
        switch.receive = modified_receive
        print(f"Enabled congestion simulation on switch {node_id}.")

    def enable_variable_congestion_simulation(self, node_id, max_intensity=5.0, seed=42, simulation_duration=None):
        if node_id not in self.switches:
            raise ValueError(f"No switch found for node id {node_id}.")
        
        self.simulation_start_time = time.time()
        self.congestion_intervals = self._generate_congestion_intervals(simulation_duration)
        
        # Generate congestion intensities for each interval.
        intensity_rng = random.Random(seed)
        self.congestion_intensities = {i: intensity_rng.uniform(1.5, max_intensity) for i in range(len(self.congestion_intervals))}
        
        print("Generated congestion intervals with variable intensity:")
        for i, (start, end) in enumerate(self.congestion_intervals):
            intensity = self.congestion_intensities[i]
            print(f"  Interval {i}: {start:.2f} s to {end:.2f} s, Intensity: {intensity:.2f}")
        
        switch = self.switches[node_id]
        original_receive = switch.receive

        def modified_receive(packet):
            rel_time = time.time() - self.simulation_start_time
            congested = False
            intensity = 1.0
            # Determine if we are in a congestion interval and get its intensity.
            for i, (start, end) in enumerate(self.congestion_intervals):
                if start <= rel_time <= end:
                    congested = True
                    intensity = self.congestion_intensities[i]
                    break
            if congested:
                state = f"congested (intensity: {intensity:.2f})"
                # Scale drop probability by intensity, but cap at 95%
                drop_prob = min(0.95, self.congested_drop_probability * intensity)
                delay_factor = self.congestion_delay_factor
            else:
                state = "normal"
                drop_prob = self.normal_drop_probability
                delay_factor = 1.0
            arrival_time = time.time()
            if self.rng.random() < drop_prob:
                self.event_log.append((rel_time, None, None))
            else:
                if delay_factor > 1.0:
                    extra_delay = (delay_factor - 1.0) * 0.01
                    time.sleep(extra_delay)
                processed_time = time.time()
                measured_delay = processed_time - arrival_time
                self.event_log.append((rel_time, processed_time - self.simulation_start_time, measured_delay))
                original_receive(packet)
        switch.receive = modified_receive
        print(f"Enabled variable congestion simulation on switch {node_id}.")

    def simulate_traffic(self, duration_seconds=10, avg_interarrival_ms=10):
        self.network.simulate_traffic(duration_seconds, avg_interarrival_ms)

    def compare_distribution_parameters(self, pred_mean: float, pred_std: float) -> float:
        params = self.network.get_distribution_parameters()[0][2]
        actual_mean = params["mean"]
        actual_std = params["std"]
        kl_div = math.log(pred_std / actual_std) + ((actual_std**2 + (actual_mean - pred_mean)**2) / (2 * pred_std**2)) - 0.5
        if kl_div <= 0.05:
            print(f"KL divergence: {kl_div:.4f} ✅")
        else:
            print(f"KL divergence: {kl_div:.4f} ❌")
        return kl_div

    def get_network(self):
        return self.network