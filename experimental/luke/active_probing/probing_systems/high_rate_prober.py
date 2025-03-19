import logging
from .base_prober import BaseProber
import psutil  # Import psutil for resource tracking
import numpy as np  # Import numpy for mean calculation

class HighRateProber(BaseProber):
    """
    A high-rate prober that sends a large number of probes per second.
    """
    def __init__(self, simulator, probes_per_second=10):
        super().__init__(simulator, name="HighRateProber")
        self.probes_per_second = probes_per_second
    
    def probe_strategy(self, current_time):
        """
        Send a large number of probes per second.
        
        Parameters:
        -----------
        current_time: float
            Current simulation time
            
        Returns:
        --------
        list of float: Times at which to send probes
        """
        logging.debug(f"HighRateProber sending probes at {self.probes_per_second} probes per second")
        return [current_time + i/self.probes_per_second for i in range(self.probes_per_second)]
    
    def run(self, duration):
        """
        Run the high-rate probing strategy for the specified duration.
        
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
        
        logging.info(f"HighRateProber running for {effective_duration}s")
        
        current_time = self.simulator.virtual_time
        end_time = current_time + effective_duration
        
        self.probes_sent = 0
        self.probes_received = 0
        self.results = []
        self.cpu_usage = []
        self.memory_usage = []
        
        # For tracking distribution estimates over time
        self.distribution_estimates = []
        window_size = 1.0  # Reduced from 5.0 to 1.0 for more responsive estimates
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
                        delay = self.simulator.send_probe_at(probe_time)
                        self.probes_sent += 1
                        
                        if delay is not None:
                            self.probes_received += 1
                            self.results.append((current_time, delay, 0, 0))
                            window_measurements.append((current_time, delay))
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
                logging.error(f"Unexpected error in HighRateProber run: {e}", exc_info=True)
                self.simulator.virtual_time += 1.0
                current_time += 1.0
        
        # Add final distribution estimate if needed
        if window_measurements and (not self.distribution_estimates or 
                                   self.distribution_estimates[-1][0] < current_time - window_size):
            delays = [m[1] for m in window_measurements[-int(window_size*self.probes_per_second):]]
            if delays:
                mean_delay = np.mean(delays)
                std_delay = np.std(delays) if len(delays) > 1 else 0.1
                self.distribution_estimates.append((current_time, mean_delay, std_delay))
        
        logging.info(f"HighRateProber completed: sent {self.probes_sent} probes, received {self.probes_received}")
        
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