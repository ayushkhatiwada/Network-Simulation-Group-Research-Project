import random

from active_simulator_v0 import ActiveSimulator_v0


class ActiveSimulator_v1(ActiveSimulator_v0):
    """
    Extends ActiveSimulator_v0 to simulate packet drops using a fixed drop probability.
    """
    def __init__(self) -> None:
        super().__init__()
        self.drop_probability = 0.1  # 10% chance of a packet being dropped

    def send_probe_at(self, departure_time: float) -> float:
        """
        Sends a probe at the given departure time with packet drop simulation.

        If a packet is dropped, logs the event with None values and returns None.
        
        :param departure_time: The time when the probe is sent.
        :return: The delay measured for the probe, or None if the probe is dropped.
        """

        # Only allow packets to be sent within [0, 100] seconds
        if departure_time < 0 or departure_time > self.max_departure_time:
            raise ValueError(f"Departure time must be between 0 and {self.max_departure_time} seconds.")

        # Rate limiting: only allow max_probes_per_second in each second
        time_slot = int(departure_time)
        if self.probe_count_per_second.get(time_slot, 0) >= self.max_probes_per_second:
            raise Exception(f"Probe rate limit exceeded for second {time_slot}. "
                            f"Max {self.max_probes_per_second} probe per second allowed.")

        # Increment rate counter for this time slot
        self.probe_count_per_second[time_slot] = self.probe_count_per_second.get(time_slot, 0) + 1

        # Decide if probe should be dropped depending on self.drop_probability
        if random.random() < self.drop_probability:
            self.event_log.append((departure_time, None, None))
            print(f"[Drop] Probe sent at {departure_time:.2f} s was dropped")
            return None

        # Get cached delay for specific time if it exists, otherwise generate new delay
        if departure_time in self.time_cache:
            delay = self.time_cache[departure_time]
        else:
            delay = self.measure_end_to_end_delay()
            self.time_cache[departure_time] = delay

        arrival_time = departure_time + delay
        self.event_log.append((departure_time, arrival_time, delay))
        return delay
