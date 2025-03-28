import random

from active_simulator_v2 import ActiveSimulator_v2

class ActiveSimulator_v3(ActiveSimulator_v2):
    """
    Simulates congestion with variable intensity
    
    - Each congestion interval has a fixed intensity value
    - Congestion intensity directly scales the drop probability
    - Maintains the same congestion intervals from v2
    """

    def __init__(self, max_departure_time=100, paths="1", max_intensity=5.0, seed=None) -> None:
        # Pass seed to parent class to ensure consistent congestion intervals
        super().__init__(paths=paths, seed=seed)
        
        # Store max_departure_time if provided
        if max_departure_time != 100:
            self.max_departure_time = max_departure_time
            
        # Congestion intensity parameters
        self.max_intensity = max_intensity
        
        # Generate intensities for congestion intervals
        self.congestion_intensities = self._generate_congestion_intensities()

    def _generate_congestion_intensities(self):
        """
        Generate intensities for each congestion interval using the seeded RNG.
        """
        # Use the local RNG (set up in parent class) for consistent results
        return {i: self.rng.uniform(1.5, self.max_intensity) 
                for i in range(len(self.congestion_intervals))}
    
    def send_probe_at(self, departure_time: float) -> float:
        """
        Send a probe with congestion intensity effects.
        
        - Determines which congestion interval we're in (if any)
        - Applies intensity-based drop probability
        - Returns delay or None if dropped
        """
        # Basic validation as in v2
        if departure_time < 0 or departure_time > self.max_departure_time:
            raise ValueError(f"Time must be in [0, {self.max_departure_time}] sec.", departure_time)

        # Rate limiting check
        time_slot = int(departure_time)
        if self.probe_count_per_second.get(time_slot, 0) >= self.max_probes_per_second:
            raise Exception(f"Rate limit exceeded for second {time_slot}.")
        
        # Increment probe count for this second
        self.probe_count_per_second[time_slot] = self.probe_count_per_second.get(time_slot, 0) + 1

        # Check if we are in a congested interval and get its intensity
        congested = False
        intensity = 1.0
        
        for i, (start, end) in enumerate(self.congestion_intervals):
            if start <= departure_time <= end:
                congested = True
                intensity = self.congestion_intensities[i]
                break
        
        # Calculate drop probability based on congestion state and intensity
        if congested:
            state = f"congested (intensity: {intensity:.2f})"
            # Scale drop probability by intensity, cap at 0.95
            drop_prob = min(0.95, self.congested_drop_probability * intensity)
        else:
            state = "normal"
            drop_prob = self.normal_drop_probability

        # Check for packet drop using the local RNG
        if self.rng.random() < drop_prob:
            self.event_log.append((departure_time, None, None))
            print(f"[Drop] Probe at {departure_time:.2f} s dropped ({state}).")
            return None

        # Get or calculate base delay
        if departure_time in self.time_cache:
            base_delay = self.time_cache[departure_time]
        else:
            base_delay = self.measure_end_to_end_delay()
            self.time_cache[departure_time] = base_delay

        # Apply congestion delay if in congested period
        # Using a fixed delay factor as in v2
        final_delay = base_delay * self.congestion_delay_factor if congested else base_delay

        arrival_time = departure_time + final_delay
        self.event_log.append((departure_time, arrival_time, final_delay))
        return final_delay
    
    def print_congestion_details(self):
        """Display congestion intervals and their intensities"""
        print("Congestion Details:")
        for i, (start, end) in enumerate(self.congestion_intervals):
            intensity = self.congestion_intensities[i]
            print(f"  {start:.2f}s to {end:.2f}s - Intensity: {intensity:.2f}")
            print(f"  Drop probability: {min(0.95, self.congested_drop_probability * intensity):.2f}") 