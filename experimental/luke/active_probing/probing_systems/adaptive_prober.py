import logging
from .base_prober import BaseProber
import numpy as np
import psutil
import time  # Add this import

class AdaptiveProber(BaseProber):
    """
    An adaptive prober that adjusts its probing rate based on congestion.
    """
    def __init__(self, simulator, base_probes_per_second=5, max_probes_per_second=50, max_allowed_rate=10):
        super().__init__(simulator, name="AdaptiveProber")
        self.base_probes_per_second = base_probes_per_second
        self.max_probes_per_second = max_probes_per_second
        self.current_probes_per_second = base_probes_per_second
        self.max_allowed_rate = max_allowed_rate
        self.recent_probes = []  # Store recent probes
        # Add counters for probe rate limiting
        self.last_second = 0
        self.probes_this_second = 0
    
    def probe_strategy(self, current_time):
        """
        Adapt probing rate based on network conditions.
        
        Parameters:
        -----------
        current_time: float
            Current simulation time
            
        Returns:
        --------
        list of float: Times at which to send probes
        """
        # Default to base rate
        probes_to_send = self.base_probes_per_second
        
        # If we have enough recent measurements, analyze them for congestion
        if len(self.recent_probes) >= 5:
            # Get the most recent delays
            recent_delays = [delay for _, delay in self.recent_probes[-10:]]
            
            # Calculate delay statistics
            mean_delay = np.mean(recent_delays)
            std_delay = np.std(recent_delays) if len(recent_delays) > 1 else 0
            
            # Calculate coefficient of variation (normalized std)
            cv = std_delay / mean_delay if mean_delay > 0 else 0
            
            # Check for signs of congestion (high delay variance or increasing delays)
            if cv > 0.2 or (len(recent_delays) > 3 and np.mean(recent_delays[-3:]) > 1.5 * np.mean(recent_delays[:-3])):
                # Detected potential congestion - increase probing rate
                probes_to_send = min(self.max_probes_per_second, self.current_probes_per_second * 2)
                logging.info(f"Congestion detected at {current_time:.1f}s - increasing probe rate to {probes_to_send}/s")
            else:
                # Network seems stable - gradually decrease probing rate
                probes_to_send = max(self.base_probes_per_second, 
                                    int(self.current_probes_per_second * 0.8))
                
        # Update current probing rate
        self.current_probes_per_second = probes_to_send
        
        # Ensure we respect the rate limit
        actual_probes = min(probes_to_send, self.max_allowed_rate)
        
        logging.info(f"AdaptiveProber at {current_time:.1f}s: sending {actual_probes} probes")
        
        # Space probes evenly within the second
        if actual_probes == 1:
            return [current_time + 0.5]  # Send one probe in the middle of the second
        else:
            return [current_time + (i+1)/(actual_probes+1) for i in range(actual_probes)]
    
    def run(self, duration):
        """
        Run the adaptive probing strategy for the specified duration.
        
        Parameters:
        -----------
        duration: float
            Duration in seconds to run the probing
            
        Returns:
        --------
        dict: Results and metrics
        """
        # Ensure we respect the max simulation time
        effective_duration = min(duration, getattr(self.simulator, 'max_simulation_time', duration))
        
        logging.info(f"AdaptiveProber running for {effective_duration}s")
        
        current_time = self.simulator.virtual_time
        end_time = current_time + effective_duration
        
        self.probes_sent = 0
        self.probes_received = 0
        self.results = []
        self.cpu_usage = []
        self.memory_usage = []
        self.recent_probes = []  # Store recent probes for rate adaptation
        
        # For tracking distribution estimates over time
        self.distribution_estimates = []
        window_size = 5.0  # Time window for distribution estimation (seconds)
        window_measurements = []
        last_estimate_time = 0
        
        while current_time < end_time:
            try:
                probe_times = self.probe_strategy(current_time)
                
                for probe_time in probe_times:
                    # Skip probes beyond our effective duration
                    if probe_time >= end_time:
                        continue
                        
                    try:
                        # Use send_probe_at to be consistent with HighRateProber
                        delay = self.simulator.send_probe_at(probe_time)
                        self.probes_sent += 1
                        
                        if delay is not None:
                            self.probes_received += 1
                            self.results.append((current_time, delay, 0, 0))
                            self.recent_probes.append((current_time, delay))
                            window_measurements.append((current_time, delay))
                            
                            # Limit the number of recent probes to avoid memory issues
                            if len(self.recent_probes) > 100:
                                self.recent_probes.pop(0)
                        else:
                            logging.debug(f"Probe dropped at {probe_time}")
                    except Exception as e:
                        logging.error(f"Error sending probe at time {probe_time}: {e}")
                
                # Track CPU and memory usage
                self.cpu_usage.append(psutil.cpu_percent())
                self.memory_usage.append(psutil.virtual_memory().used / (1024 * 1024))  # Convert to MB
                
                # Calculate distribution parameters every window_size seconds
                if current_time - last_estimate_time >= window_size and window_measurements:
                    # Get only measurements within the current window
                    recent_measurements = [m for m in window_measurements 
                                         if m[0] > current_time - window_size]
                    
                    if recent_measurements:
                        delays = [m[1] for m in recent_measurements]
                        mean_delay = np.mean(delays)
                        std_delay = np.std(delays) if len(delays) > 1 else 0.1
                        
                        # Store the distribution estimate with timestamp
                        self.distribution_estimates.append((current_time, mean_delay, std_delay))
                        
                        last_estimate_time = current_time
                
                self.simulator.virtual_time += 1.0
                current_time += 1.0
            except Exception as e:
                logging.error(f"Unexpected error in AdaptiveProber run: {e}", exc_info=True)
                self.simulator.virtual_time += 1.0
                current_time += 1.0
        
        # Add final distribution estimate if needed
        if window_measurements and (not self.distribution_estimates or 
                                   self.distribution_estimates[-1][0] < current_time - window_size):
            delays = [m[1] for m in window_measurements[-int(window_size*self.base_probes_per_second):]]
            if delays:
                mean_delay = np.mean(delays)
                std_delay = np.std(delays) if len(delays) > 1 else 0.1
                self.distribution_estimates.append((current_time, mean_delay, std_delay))
        
        logging.info(f"AdaptiveProber completed: sent {self.probes_sent} probes, received {self.probes_received}")
        
        return {
            "name": self.name,
            "results": self.results,
            "distribution_estimates": self.distribution_estimates,
            "probes_sent": self.probes_sent,
            "probes_received": self.probes_received,
            "loss_rate": 1.0 - (self.probes_received / max(1, self.probes_sent)),
            "runtime": effective_duration,  # Use effective duration instead of requested
            "avg_cpu": np.mean(self.cpu_usage) if self.cpu_usage else 0,
            "avg_memory": np.mean(self.memory_usage) if self.memory_usage else 0,
            "max_memory": max(self.memory_usage) if self.memory_usage else 0,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage
        }