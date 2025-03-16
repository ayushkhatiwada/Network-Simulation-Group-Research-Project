import random

from active_simulator_v1 import ActiveSimulator_v1


class ActiveSimulator_v2(ActiveSimulator_v1):
    """
    Extends ActiveSimulator_v1 by simulating congestion via fixed congestion intervals.

    This version uses four fixed congestion interval lengths (5, 10, 15, and 20 seconds)
    that sum to 50 seconds, but they are randomly ordered and then spread over the 100 second
    simulation period by randomly partitioning the remaining 50 seconds into gaps. When a probe
    is sent within a congestion interval, a higher drop probability and a delay multiplier are applied.
    """
    def __init__(self) -> None:
        super().__init__()

        # Fixed congestion interval lengths that sum to 50 seconds.
        fixed_lengths = [5, 10, 15, 20]
        # Randomize the order of the congestion intervals.
        random.shuffle(fixed_lengths)
        self.fixed_interval_lengths = fixed_lengths

        total_congested_time = sum(self.fixed_interval_lengths)                 # 50 seconds
        total_simulation_time = self.max_departure_time                         # 100 seconds
        total_non_congested_time = total_simulation_time - total_congested_time # 50 seconds

        # There will be (number of intervals + 1) gaps.
        num_non_congestion_intervals = len(self.fixed_interval_lengths) + 1
        random_gaps = [random.random() for _ in range(num_non_congestion_intervals)]
        gap_sum = sum(random_gaps)
        gap_lengths = [(gap / gap_sum) * total_non_congested_time for gap in random_gaps]

        # Build congestion intervals sequentially.
        self.congestion_intervals = []
        current_time = gap_lengths[0]
        for i, interval_length in enumerate(self.fixed_interval_lengths):
            start = current_time
            end = start + interval_length
            self.congestion_intervals.append((start, end))
            current_time = end + gap_lengths[i + 1]

        # Print the generated congestion intervals.
        print("Generated Congestion Intervals:  ")
        for interval in self.congestion_intervals:
            print(f"{interval[0]:.2f} s to {interval[1]:.2f} s")

        # Set congestion-specific parameters.
        self.normal_drop_probability = 0.1      # Drop probability under normal conditions.
        self.congested_drop_probability = 0.4   # Drop probability during congestion.
        self.congestion_delay_factor = 1.5      # Delay multiplier during congestion.

    def send_probe_at(self, departure_time: float) -> float:
        """
        Sends a probe at the specified departure time, applying congestion effects if within an interval.

        If the probe is dropped, the event is logged with None values and None is returned.
        Probes in congestion intervals have a higher chance of being dropped and their delay is multiplied.

        :param departure_time: Time (in seconds) when the probe is sent.
        :return: The measured (possibly congested) delay, or None if the packet is dropped.
        """
        if departure_time < 0 or departure_time > self.max_departure_time:
            raise ValueError(f"Departure time must be between 0 and {self.max_departure_time} seconds.")

        # Enforce rate limiting: each time slot (integer part of departure_time) has a maximum.
        time_slot = int(departure_time)
        if self.probe_count_per_second.get(time_slot, 0) >= self.max_probes_per_second:
            raise Exception(f"Probe rate limit exceeded for second {time_slot}.")
        self.probe_count_per_second[time_slot] = self.probe_count_per_second.get(time_slot, 0) + 1

        # Check if departure_time falls within any congestion interval.
        congested = any(start <= departure_time <= end for start, end in self.congestion_intervals)
        if congested:
            state = "congested"
            drop_probability = self.congested_drop_probability
        else:
            state = "normal"
            drop_probability = self.normal_drop_probability

        # Simulate packet drop.
        if random.random() < drop_probability:
            self.event_log.append((departure_time, None, None))
            print(f"[Drop] Probe at {departure_time:.2f} s dropped due to {state} conditions.")
            return None

        # Retrieve or measure the base delay.
        if departure_time in self.time_cache:
            base_delay = self.time_cache[departure_time]
        else:
            base_delay = self.measure_end_to_end_delay()
            self.time_cache[departure_time] = base_delay

        # If congested, apply the congestion delay factor.
        final_delay = base_delay * self.congestion_delay_factor if congested else base_delay
        arrival_time = departure_time + final_delay
        self.event_log.append((departure_time, arrival_time, final_delay))
        return final_delay
