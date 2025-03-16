import random

from active_simulator_v1 import ActiveSimulator_v1


class ActiveSimulator_v2(ActiveSimulator_v1):
    """
    Simulates congestion

    - Congestion occurs at 4 randomly placed time intervals within the 100 seconds
    - 4 intervals of lengths 5, 10, 15, 20 seconds
    
    - Total congested time is 50 sec; non-congested time is 50 sec.
    - Probes sent during congestion intervals have higher drop rate and increased delay.
    """
    
    def __init__(self) -> None:
        super().__init__()

        # Congestion parameters
        self.normal_drop_probability = 0.1
        self.congested_drop_probability = 0.4
        self.congestion_delay_factor = 1.5

        # Generate congestion intervals and store them.
        self.congestion_intervals = self._generate_congestion_intervals()

    def _generate_congestion_intervals(self):
        """
        Generate congestion intervals
        Don't worry about how this function works
        
        Returns list of (start, end) tuples representing the congestion time periods
        """

        fixed_lengths = [5, 10, 15, 20]
        random.shuffle(fixed_lengths)
        self.fixed_interval_lengths = fixed_lengths  # e.g., [15, 5, 20, 10]

        total_congested_time = sum(fixed_lengths)  # 50 secs
        total_simulation_time = self.max_departure_time  # 100 secs
        total_non_congested_time = total_simulation_time - total_congested_time  # 50 secs

        num_of_non_congestion_intervals = len(fixed_lengths) + 1
        weights = [random.random() for _ in range(num_of_non_congestion_intervals)]
        weight_sum = sum(weights)
        non_congested_durations = [(w / weight_sum) * total_non_congested_time for w in weights]

        congestion_intervals = []
        current_time = non_congested_durations[0]

        for i, length in enumerate(fixed_lengths):
            start = current_time
            end = start + length
            congestion_intervals.append((start, end))
            current_time = end + non_congested_durations[i + 1]

        return congestion_intervals

    def print_congestion_intervals(self):
        """
        Only use for debugging 
        Your algorithm should NOT know when the congestion intervals occur 
        It should try to guess where they occur by sampling
        """

        print("Congestion Intervals:")
        for start, end in self.congestion_intervals:
            print(f"  {start:.2f} s to {end:.2f} s")

    def send_probe_at(self, departure_time: float) -> float:
        """
        Send a probe at a given time.
        
        - Enforces rate limit.
        - Checks if time is in a congestion interval.
        - Applies drop probability and delay factor accordingly.
        - Returns delay or None if dropped.
        """

        # Only allow packets to be sent within [0, 100] seconds
        if departure_time < 0 or departure_time > self.max_departure_time:
            raise ValueError(f"Time must be in [0, {self.max_departure_time}] sec.")

        # Rate limiting: only allow max_probes_per_second in each second
        time_slot = int(departure_time)
        if self.probe_count_per_second.get(time_slot, 0) >= self.max_probes_per_second:
            raise Exception(f"Rate limit exceeded for second {time_slot}.")
        self.probe_count_per_second[time_slot] = self.probe_count_per_second.get(time_slot, 0) + 1

        # Check if we are in a congested time period, increase drop probability if we are
        congested = any(start <= departure_time <= end for start, end in self.congestion_intervals)
        if congested:
            state = "congested"
            drop_prob = self.congested_drop_probability
        else:
            state = "normal"
            drop_prob = self.normal_drop_probability

        # Decide if prob should be dropped depending on self.drop_probability
        if random.random() < drop_prob:
            self.event_log.append((departure_time, None, None))
            print(f"[Drop] Probe at {departure_time:.2f} s dropped ({state}).")
            return None

        # Get cached delay for specific time if it exists, otherwise generate new delay
        if departure_time in self.time_cache:
            base_delay = self.time_cache[departure_time]
        else:
            base_delay = self.measure_end_to_end_delay()
            self.time_cache[departure_time] = base_delay

        # Increase dealy by congestion_delay_factor if we are in a congestion period
        final_delay = base_delay * self.congestion_delay_factor if congested else base_delay

        arrival_time = departure_time + final_delay
        self.event_log.append((departure_time, arrival_time, final_delay))
        return final_delay
