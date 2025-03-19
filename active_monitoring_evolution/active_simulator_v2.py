import random

from active_simulator_v1 import ActiveSimulator_v1


class ActiveSimulator_v2(ActiveSimulator_v1):
    """
    Simulates congestion

    - Congestion occurs at randomly placed time intervals within the 100 seconds
    - Total congested time is 50 sec; non-congested time is 50 sec.
    - Probes sent during congestion intervals have higher drop rate and increased delay.
    """

    def __init__(self, paths="1") -> None:
        super().__init__(paths)

        # Congestion parameters
        self.normal_drop_probability = 0.1
        self.congested_drop_probability = 0.4
        self.congestion_delay_factor = 1.5

        # Generate time intervals where congestion will occur
        self.congestion_intervals = self._generate_congestion_intervals()


    def send_probe_at(self, departure_time: float) -> float:
        """
        Send a probe at a given time.
        
        - Checks if depature time is in a congestion interval.
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
        
        # Increment probe count for this second
        self.probe_count_per_second[time_slot] = self.probe_count_per_second.get(time_slot, 0) + 1

        # Check if we are in a congested time period, increase drop probability if we are
        congested = any(start <= departure_time <= end for start, end in self.congestion_intervals)
        if congested:
            state = "congested"
            drop_prob = self.congested_drop_probability
        else:
            state = "normal"
            drop_prob = self.normal_drop_probability

        # Decide if probe should be dropped depending on self.drop_probability
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


    # Don't read this function, don't worry about how it works
    # You should write an algorithms that initally knows nothing about where the congestion occurs
    # Having knowledge about when the congestion occurs gives you an unfair advantage
    # potentially leading to your algorithm overfitting for this specific congestion scenario.
    def _generate_congestion_intervals(self):
        """
        Generate congestion intervals
        
        Returns list of (start, end) tuples representing the congestion time periods
        """
        something = [0x5, 0xA, 0xF, 0x14]
        random.shuffle(something)
        total_congested_time = sum(something)
        total_simulation_time = self.max_departure_time
        total_non_congested_time = total_simulation_time - total_congested_time
        num_of_non_congestion_intervals = len(something) + 1
        weights = [random.random() for _ in range(num_of_non_congestion_intervals)]
        weight_sum = sum(weights)
        non_congested_durations = [(w / weight_sum) * total_non_congested_time for w in weights]
        congestion_intervals = []
        current_time = non_congested_durations[0]
        for i, length in enumerate(something):
            start = current_time
            end = start + length
            congestion_intervals.append((start, end))
            current_time = end + non_congested_durations[i + 1]
        return congestion_intervals


    def print_congestion_intervals(self):
        """
        Only use for debugging or checking
        Your algorithm should NOT know when the congestion intervals occur 
        It should try to guess where they occur by sampling
        """

        print("Congestion Intervals:")
        for start, end in self.congestion_intervals:
            print(f"  {start:.2f} s to {end:.2f} s")
