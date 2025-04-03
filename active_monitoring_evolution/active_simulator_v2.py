import random

from active_simulator_v1 import ActiveSimulator_v1


class ActiveSimulator_v2(ActiveSimulator_v1):

    def __init__(self, paths="1", random_seed=None, simulation_duration=100) -> None:
        super().__init__(paths=paths, random_seed=random_seed, simulation_duration=simulation_duration)

        self.normal_drop_probability = 0.1
        self.congested_drop_probability = 0.4
        self.congestion_delay_factor = 1.5
        self.congestion_rng = random.Random(random_seed)
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

        # Check if we are in a congested time period
        congested = any(start <= departure_time <= end for start, end in self.congestion_intervals)
        if congested:
            state = "congested"
            drop_prob = self.congested_drop_probability
        else:
            state = "normal"
            drop_prob = self.normal_drop_probability

        # Decide if probe should be dropped using local drop_rng
        if self.drop_rng.random() < drop_prob:
            self.event_log.append((departure_time, None, None))
            #(f"[Drop] Probe at {departure_time:.2f} s dropped ({state}).")
            return None

        # Get cached delay for specific time if it exists, otherwise generate new delay
        if departure_time in self.time_cache:
            base_delay = self.time_cache[departure_time]
        else:
            base_delay = self.measure_end_to_end_delay()
            self.time_cache[departure_time] = base_delay

        # Increase delay by congestion_delay_factor if we are in a congestion period
        final_delay = base_delay * self.congestion_delay_factor if congested else base_delay

        arrival_time = departure_time + final_delay
        self.event_log.append((departure_time, arrival_time, final_delay))
        return final_delay

    def _generate_congestion_intervals(self):
        """
        Generate congestion intervals for half the total simulation time.
        Returns a list of (start, end) tuples representing congested time periods.
        """
        total_simulation_time = self.max_departure_time
        total_congested_time = total_simulation_time / 2.0  # half of the overall sim time
        total_non_congested_time = total_simulation_time - total_congested_time

        # We'll break congestion time intervals into random segments.
        # For example, you can either keep the original logic for random intervals
        # or simplify to a single half block. Below we'll keep a similar approach
        # but scale "something" so it sums to total_congested_time.
        # ---------------------------------------------------------------
        # Example with just one big chunk: 
        # return [(0, total_congested_time)]  # Entire first half is congested
        # You can then separate it from the second half, etc.
        # ---------------------------------------------------------------

        # Original approach: fixed list of interval lengths
        base_intervals = [5, 10, 15, 20]  # sums to 50
        # We scale them to match total_congested_time
        scale_factor = total_congested_time / sum(base_intervals)
        intervals = [i * scale_factor for i in base_intervals]

        # Shuffle intervals to randomize the positions
        self.congestion_rng.shuffle(intervals)
        num_of_non_congestion_intervals = len(intervals) + 1

        # Randomly partition the total non-congested time among these intervals
        weights = [self.congestion_rng.random() for _ in range(num_of_non_congestion_intervals)]
        weight_sum = sum(weights)
        non_congested_durations = [(w / weight_sum) * total_non_congested_time for w in weights]

        # Finally, build the congestion intervals
        congestion_intervals = []
        current_time = non_congested_durations[0]

        for i, length in enumerate(intervals):
            start = current_time
            end = start + length
            congestion_intervals.append((start, end))
            if i + 1 < len(non_congested_durations):
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
