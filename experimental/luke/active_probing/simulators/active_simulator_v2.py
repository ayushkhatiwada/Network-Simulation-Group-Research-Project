import random
import math
import threading
import time
import logging
import inspect
import copy
from .active_simulator_v1 import ActiveSimulator_v1
from experimental.luke.active_probing.passive_monitoring_evolution.switch_and_packet import Packet

class ActiveSimulator_v2(ActiveSimulator_v1):
    """
    Extends ActiveSimulator_v1 to simulate network congestion with:
    - Time-varying congestion levels
    - Modified distribution parameters during congestion
    - Higher packet drop rates during congestion
    - Actual background traffic generation during congestion
    - Support for passive monitoring via sketches
    """
    def __init__(self, paths="1", congestion_duration=30.0, congestion_intensity=0.7, 
                 num_congestion_windows=3, max_simulation_time=60.0) -> None:
        """
        Initialize the congestion-aware simulator.
        
        Parameters:
        -----------
        paths: str or int
            Path configuration for the network
        congestion_duration: float
            Total duration of congestion in seconds
        congestion_intensity: float
            Maximum intensity of congestion (0.0 to 1.0)
        num_congestion_windows: int
            Number of congestion windows to generate
        max_simulation_time: float
            Maximum simulation time - ALL congestion windows will be within this time
        """
        super().__init__(paths)
        
        # Congestion parameters
        self.congestion_duration = congestion_duration
        self.congestion_intensity = congestion_intensity
        self.num_congestion_windows = num_congestion_windows
        self.max_simulation_time = max_simulation_time  # Default to 60 seconds
        
        # Generate congestion windows
        self.congestion_windows = self._generate_congestion_windows()
        
        # Base drop probability from v1
        self.base_drop_probability = self.drop_probability
        self.congested_drop_probability = min(0.4, self.drop_probability * 4)
        
        # Track background traffic rate
        self.base_traffic_rate = 5.0  # packets per second
        self.background_traffic_active = False
        self.background_thread = None
        
        # Store original distribution parameters for each edge
        self.original_edge_params = {}
        for u, v in self.network.graph.edges():
            # For MultiGraph, we need to handle multiple edges
            edges = self.network.graph[u][v]
            for key in edges:
                edge_data = edges[key]
                # Create a copy of the original parameters
                self.original_edge_params[(u, v, key)] = edge_data.copy()
        
        # Passive monitoring components
        self.switches = {
            self.network.SOURCE: self.network.source_switch,
            self.network.DESTINATION: self.network.destination_switch
        }
        self.passive_event_log = []
        
        # Storage for distribution predictions
        self.predictions = {}
        
        print(f"Initialized congestion model with {len(self.congestion_windows)} windows")
        for start, end, intensity in self.congestion_windows:
            print(f"  Congestion window: {start:.1f}s - {end:.1f}s, intensity: {intensity:.2f}")
        
        # Start background traffic thread
        self.background_thread = threading.Thread(target=self._background_traffic_generator)
        self.background_thread.daemon = True
        self.background_thread.start()
        
        # Configure switches with drop probabilities
        self._configure_switches()

    def _generate_congestion_windows(self):
        """Generate strictly non-overlapping congestion windows"""
        windows = []
        
        # Calculate the total time we can use for congestion windows
        total_simulation_time = self.max_simulation_time
        
        # First, check if we have enough time for all windows plus gaps
        # At minimum, we need: window_count * duration + (window_count-1) * min_gap
        min_gap_between_windows = 10.0  # At least 10 seconds between congestion periods
        
        total_required_time = (self.num_congestion_windows * self.congestion_duration + 
                               (self.num_congestion_windows - 1) * min_gap_between_windows)
        
        if total_required_time > total_simulation_time:
            original_windows = self.num_congestion_windows
            original_duration = self.congestion_duration
            
            # Adjust parameters to fit the simulation time
            if self.num_congestion_windows > 1:
                # Try reducing window count first
                self.num_congestion_windows = max(1, int(total_simulation_time / 
                                                 (self.congestion_duration + min_gap_between_windows)))
            
            # If still not enough, adjust duration
            if ((self.num_congestion_windows * self.congestion_duration + 
                (self.num_congestion_windows - 1) * min_gap_between_windows) > total_simulation_time):
                max_duration = (total_simulation_time - (self.num_congestion_windows - 1) * min_gap_between_windows) / self.num_congestion_windows
                self.congestion_duration = max_duration
            
            logging.warning(f"Adjusted congestion parameters to fit simulation time: "
                           f"windows {original_windows} → {self.num_congestion_windows}, "
                           f"duration {original_duration} → {self.congestion_duration}")
        
        # Now create evenly spaced windows
        if self.num_congestion_windows == 1:
            # With one window, center it in the simulation time
            start_time = (total_simulation_time - self.congestion_duration) / 2
            end_time = start_time + self.congestion_duration
            windows.append((start_time, end_time, self.congestion_intensity))
        else:
            # With multiple windows, evenly space them
            total_congestion_time = self.num_congestion_windows * self.congestion_duration
            total_gap_time = total_simulation_time - total_congestion_time
            
            # Calculate gap between windows
            gap_between_windows = total_gap_time / (self.num_congestion_windows + 1)
            
            # Initial gap
            current_time = gap_between_windows
            
            # Create each window
            for i in range(self.num_congestion_windows):
                start_time = current_time
                end_time = start_time + self.congestion_duration
                windows.append((start_time, end_time, self.congestion_intensity))
                
                # Move to the next window position
                current_time = end_time + gap_between_windows
        
        logging.info(f"Generated {len(windows)} congestion windows:")
        for i, (start, end, intensity) in enumerate(windows):
            logging.info(f"  Window {i+1}: {start:.1f}s - {end:.1f}s (duration: {end-start:.1f}s, intensity: {intensity:.2f})")
        
        return windows

    def _configure_switches(self):
        """Configure switches with drop probabilities and congestion effects"""
        self.switches = {
            self.network.SOURCE: self.network.source_switch,
            self.network.DESTINATION: self.network.destination_switch
        }
        
        for node_id, switch in self.switches.items():
            # Store the original receive function before modification
            if not hasattr(switch, '_original_receive'):
                switch._original_receive = switch.receive
            else:
                # If resetting, restore the original first
                switch.receive = switch._original_receive
            
            # Create a new function that captures the original
            def create_modified_receive(original_fn, node_id):
                def modified_receive(packet, vt=None):
                    current_time = time.time()  # REAL TIME
                    rel_time = current_time - self.start_time if hasattr(self, 'start_time') else 0
                    
                    # Get congestion at this time
                    congestion = self._get_congestion_at_time(rel_time)
                    
                    # Determine drop probability based on congestion
                    drop_prob = self.base_drop_probability * (1.0 + 4.0 * congestion)
                    
                    if random.random() < drop_prob:
                        self.passive_event_log.append((rel_time, None, None))
                        logging.info(f"[Drop] Packet dropped at switch {node_id} at {rel_time:.2f}s (congestion: {congestion:.2f})")
                        return None
                    else:
                        # If congested, simulate additional delay
                        if congestion > 0:
                            delay_factor = 1.0 + (1.5 * congestion)
                            extra_delay = (delay_factor - 1.0) * 0.01
                            time.sleep(extra_delay)
                        
                        processed_time = time.time()
                        measured_delay = processed_time - current_time
                        self.passive_event_log.append((rel_time, rel_time + measured_delay, measured_delay))
                        
                        # Call the original function with appropriate arguments
                        if vt is not None:
                            return original_fn(packet, vt)
                        else:
                            return original_fn(packet)
                
                return modified_receive
            
            # Assign the new function to the switch
            switch.receive = create_modified_receive(switch._original_receive, node_id)

    def attach_sketch(self, node_id, sketch):
        """
        Attach a sketch to a switch for passive monitoring.
        
        Parameters:
        -----------
        node_id: int
            ID of the switch
        sketch: object
            Sketch object to attach
        """
        if node_id in self.switches:
            self.switches[node_id].add_sketch(sketch)
            print(f"Attached {sketch.__class__.__name__} to switch {node_id}.")
        else:
            raise ValueError(f"No switch found for node id {node_id}.")

    def _get_congestion_at_time(self, time):
        """
        Get the congestion intensity at a specific time.
        
        Parameters:
        -----------
        time: float
            The time to check
            
        Returns:
        --------
        float: Congestion intensity (0.0 to 1.0)
        """
        for start, end, intensity in self.congestion_windows:
            if start <= time < end:
                return intensity
        return 0.0  # No congestion outside defined windows

    def _update_edge_parameters(self, current_time):
        """
        Update edge parameters based on current congestion level.
        This directly modifies the distribution parameters in the network graph.
        """
        congestion = self._get_congestion_at_time(current_time)
        
        if congestion > 0:
            # Update each edge's parameters
            for (u, v, key), original_params in self.original_edge_params.items():
                edge_data = self.network.graph[u][v][key]
                
                # Modify parameters based on congestion level
                if self.network.distribution_type == "normal":
                    # Real-world congestion effects:
                    # 1. Mean delay increases non-linearly with congestion
                    # 2. Jitter (std) increases more dramatically at high congestion
                    
                    # Delay increases exponentially with congestion
                    # At 0.5 congestion: ~2x delay, at 0.9 congestion: ~5x delay
                    mean_multiplier = math.exp(2.0 * congestion) - 0.5
                    
                    # Jitter increases even more dramatically with congestion
                    # At 0.5 congestion: ~2.5x jitter, at 0.9 congestion: ~8x jitter
                    std_multiplier = math.exp(2.5 * congestion) - 0.6
                    
                    edge_data["mean"] = original_params["mean"] * max(1.0, mean_multiplier)
                    edge_data["std"] = original_params["std"] * max(1.0, std_multiplier)
                
                elif self.network.distribution_type == "gamma":
                    # For gamma, we adjust the scale parameter to increase delay
                    scale_multiplier = math.exp(2.0 * congestion) - 0.5
                    edge_data["scale"] = original_params["scale"] * max(1.0, scale_multiplier)
                
                elif self.network.distribution_type == "lognormal":
                    # For lognormal, we adjust both mu and sigma
                    mu_increase = math.log(max(1.0, math.exp(2.0 * congestion) - 0.5))
                    sigma_multiplier = max(1.0, math.exp(congestion) - 0.5)
                    
                    edge_data["mu"] = original_params["mu"] + mu_increase
                    edge_data["sigma"] = original_params["sigma"] * sigma_multiplier
        else:
            # Reset to original parameters when no congestion
            for (u, v, key), original_params in self.original_edge_params.items():
                for param, value in original_params.items():
                    self.network.graph[u][v][key][param] = value

    def _background_traffic_generator(self):
        """
        Generate actual background traffic based on congestion levels using virtual time.
        """
        self.background_traffic_active = True
        self.virtual_time = 0.0
        
        while self.background_traffic_active and self.virtual_time < self.max_simulation_time:
            # Get current congestion level
            congestion = self._get_congestion_at_time(self.virtual_time)
            
            # Calculate traffic rate based on congestion
            current_rate = self.base_traffic_rate * math.exp(3.0 * congestion)
            
            # Generate packets based on current rate
            if current_rate > 0:
                # Calculate interarrival time (exponential distribution)
                interarrival = random.expovariate(current_rate)
                
                # Create and send packet
                packet = Packet(self.network.SOURCE, self.network.DESTINATION)
                self.network.transmit_packet(packet, self.virtual_time)
                
                # Update virtual time
                self.virtual_time += interarrival
            else:
                # No traffic, just advance time
                self.virtual_time += 0.1

    def measure_end_to_end_delay(self, at_time=None) -> float:
        """
        Measures the delay from SOURCE to DESTINATION, using the current edge parameters.
        
        Parameters:
        -----------
        at_time: float
            The time at which to measure the delay
            
        Returns:
        --------
        float: The delay as a float
        """
        # Update edge parameters based on current congestion
        if at_time is not None:
            self._update_edge_parameters(at_time)
        
        # Use the network's sample_edge_delay with the updated parameters
        return self.network.sample_edge_delay(self.network.SOURCE, self.network.DESTINATION)

    def send_probe_at(self, departure_time: float) -> float:
        """
        Sends a probe at the given departure time with congestion-aware packet drop simulation.
        
        Parameters:
        -----------
        departure_time: float
            The time when the probe is sent
            
        Returns:
        --------
        float or None: The delay measured for the probe, or None if the probe is dropped
        """
        # Add debug logging
        logging.debug(f"send_probe_at called with departure_time={departure_time}")
        
        # Apply rate limiting checks from parent class
        if departure_time < 0 or departure_time > self.max_simulation_time:
            logging.warning(f"Departure time {departure_time} out of bounds (0-{self.max_simulation_time})")
            raise ValueError(f"Departure time must be between 0 and {self.max_simulation_time} seconds.")

        time_slot = int(departure_time)
        if self.probe_count_per_second.get(time_slot, 0) >= self.max_probes_per_second:
            logging.warning(f"Probe rate limit exceeded for second {time_slot}")
            raise ValueError(f"Probe rate limit exceeded for second {time_slot}. Max {self.max_probes_per_second} probe per second allowed.")
        
        # Update probe count
        self.probe_count_per_second[time_slot] = self.probe_count_per_second.get(time_slot, 0) + 1
        
        # Get congestion at this time
        congestion = self._get_congestion_at_time(departure_time)
        logging.debug(f"Congestion at {departure_time}: {congestion}")
        
        # Adjust drop probability based on congestion
        adjusted_drop_prob = self.base_drop_probability * (1.0 + 4.0 * congestion)
        
        # Decide if probe should be dropped (with deterministic seed for debugging)
        drop_decision = random.random() < adjusted_drop_prob
        if drop_decision:
            logging.debug(f"Probe at {departure_time} dropped (drop_prob={adjusted_drop_prob})")
            self.event_log.append((departure_time, None, None))
            print(f"[Drop] Probe sent at {departure_time:.2f}s was dropped (congestion: {congestion:.2f})")
            return None

        # Get cached delay for specific time if it exists, otherwise generate new delay
        if departure_time in self.time_cache:
            delay = self.time_cache[departure_time]
        else:
            delay = self.measure_end_to_end_delay(at_time=departure_time)
            self.time_cache[departure_time] = delay
        
        logging.debug(f"Probe at {departure_time} received with delay {delay}")
        arrival_time = departure_time + delay
        self.event_log.append((departure_time, arrival_time, delay))
        return delay
    
    def get_congestion_windows(self):
        """Return the congestion windows for visualization"""
        return self.congestion_windows
    
    def get_current_distribution_parameters(self, source, destination, at_time=None):
        """
        Get the current distribution parameters, accounting for congestion.
        
        Parameters:
        -----------
        source: int
            Source node
        destination: int
            Destination node
        at_time: float
            Time at which to get parameters
            
        Returns:
        --------
        dict: Distribution parameters
        """
        # Update edge parameters based on current congestion
        if at_time is not None:
            self._update_edge_parameters(at_time)
        
        # Get parameters from the network
        return self.network.get_distribution_parameters(source, destination)
    
    def compare_distribution_parameters(self, pred_mean, pred_std, at_time=None):
        """
        Compare predicted distribution parameters with ground truth.
        
        Parameters:
        -----------
        pred_mean: float
            Predicted mean
        pred_std: float
            Predicted standard deviation
        at_time: float
            Time at which to compare (default: current virtual time)
        
        Returns:
        --------
        dict: Comparison metrics
        """
        if at_time is None:
            at_time = self.virtual_time
        
        # Get ground truth parameters
        params = self.get_current_distribution_parameters(
            self.network.SOURCE, 
            self.network.DESTINATION, 
            at_time=at_time
        )
        
        actual_mean = params["mean"]
        actual_std = params["std"]
        
        # Calculate mean absolute percentage error
        if actual_mean != 0:
            mean_perc_error = abs((pred_mean - actual_mean) / actual_mean)
        else:
            mean_perc_error = abs(pred_mean) if pred_mean != 0 else 0
        
        if actual_std != 0:
            std_perc_error = abs((pred_std - actual_std) / actual_std)
        else:
            std_perc_error = abs(pred_std) if pred_std != 0 else 0
        
        # Calculate KL divergence safely (avoid division by zero and log of negative numbers)
        # KL(p||q) where p is the true distribution and q is the predicted
        try:
            # Ensure standard deviations are positive
            safe_pred_std = max(pred_std, 1e-8)
            safe_actual_std = max(actual_std, 1e-8)
            
            # Calculate KL divergence for Gaussian distributions
            kl_div = math.log(safe_pred_std / safe_actual_std) + \
                    ((safe_actual_std**2 + (actual_mean - pred_mean)**2) / (2 * safe_pred_std**2)) - 0.5
            
            # KL divergence should be non-negative
            kl_div = max(0, kl_div)
        except (ValueError, ZeroDivisionError, OverflowError):
            logging.warning(f"Error calculating KL divergence at time {at_time}. Using fallback error metric.")
            kl_div = abs(pred_mean - actual_mean) + abs(pred_std - actual_std)
        
        # Save comparison result
        if at_time not in self.predictions:
            self.predictions[at_time] = []
        
        comparison = {
            "time": at_time,
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "actual_mean": actual_mean,
            "actual_std": actual_std,
            "mean_error": abs(pred_mean - actual_mean),
            "std_error": abs(pred_std - actual_std),
            "mean_perc_error": mean_perc_error,
            "std_perc_error": std_perc_error,
            "kl_divergence": kl_div
        }
        
        self.predictions[at_time].append(comparison)
        
        return comparison
    
    def simulate_traffic(self, duration_seconds=10, avg_interarrival_ms=50):
        """
        Simulate traffic using the network's built-in method.
        """
        self.network.simulate_traffic(duration_seconds, avg_interarrival_ms)
    
    def reset(self):
        """Reset the simulator to its initial state"""
        # Reset congestion windows
        self.congestion_windows = self._generate_congestion_windows()
        
        # Reset time tracking
        self.virtual_time = 0.0
        self.start_time = time.time()
        
        # Reset passive monitoring
        self.passive_event_log = []
        
        # Reset switches if needed
        self._configure_switches()
        
        # Reset edge parameters to original values
        for edge_id, params in self.original_edge_params.items():
            u, v, key = edge_id
            # For MultiGraph, update the specific edge
            self.network.graph[u][v][key].update(params)
        
        print("Simulator reset to initial state.")

    def __del__(self):
        """Cleanup when the object is destroyed"""
        self.background_traffic_active = False
        if self.background_thread and self.background_thread.is_alive():
            self.background_thread.join(timeout=1.0)

    def __deepcopy__(self, memo):
        """Handle proper deep copying of the simulator"""
        # Create a shallow copy first
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        # Save state of background traffic
        was_background_active = self.background_traffic_active
        thread_existed = self.background_thread is not None
        
        # Temporarily disable threading for copy
        self.background_traffic_active = False
        self.background_thread = None
        
        # Copy all attributes except thread-related ones
        for k, v in self.__dict__.items():
            if k != 'background_thread':
                setattr(result, k, copy.deepcopy(v, memo))
        
        # Restore original state
        self.background_traffic_active = was_background_active
        if thread_existed and was_background_active:
            self.background_thread = threading.Thread(target=self._background_traffic_generator)
            self.background_thread.daemon = True
            self.background_thread.start()
        
        # Initialize the new simulator's background thread if needed
        result.background_thread = None
        if was_background_active:
            result.background_thread = threading.Thread(target=result._background_traffic_generator)
            result.background_thread.daemon = True
            result.background_thread.start()
        
        return result