import numpy as np
from collections import deque
from .base_prober import BaseProber
import time
import threading
import logging
import psutil

class MemoryEfficientProber(BaseProber):
    """
    A memory-efficient prober that uses online statistics calculation.
    """
    def __init__(self, simulator, analysis_window=10.0, probes_per_second=2):
        super().__init__(simulator, name="MemoryEfficientProber")
        self.analysis_window = analysis_window
        self.probes_per_second = probes_per_second
        
        # Use fixed-size circular buffer for recent delays
        self.delay_buffer = deque(maxlen=int(analysis_window * probes_per_second))
        
        # Online statistics
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # For variance calculation
    
    def probe_strategy(self, current_time):
        """
        Send a fixed number of probes per second.
        
        Parameters:
        -----------
        current_time: float
            Current simulation time
            
        Returns:
        --------
        list of float: Times at which to send probes
        """
        # Distribute probes evenly within the second
        return [current_time + i/self.probes_per_second for i in range(self.probes_per_second)]
    
    def analyze_recent_probes(self, current_time):
        """
        Use online algorithm to calculate mean and variance.
        This is more memory efficient than storing all probes.
        
        Based on Welford's online algorithm.
        """
        if not self.delay_buffer:
            return None, None
        
        # Calculate mean and std from buffer
        mean = self.mean
        std = np.sqrt(self.M2 / max(1, self.count - 1)) if self.count > 1 else 0
        
        # Compare with ground truth
        if hasattr(self.simulator, 'compare_distribution_parameters'):
            self.simulator.compare_distribution_parameters(mean, std, at_time=current_time)
        
        # Store in distribution estimates
        self.distribution_estimates.append((current_time, mean, std))
        
        return mean, std
    
    def _track_resources(self):
        """Track CPU and memory usage in a separate thread."""
        self.cpu_usage = []
        self.memory_usage = []
        
        while self.tracking:
            self.cpu_usage.append(psutil.cpu_percent())
            self.memory_usage.append(psutil.virtual_memory().used / (1024 * 1024))  # MB
            time.sleep(1.0)  # Sample every second
    
    def run(self, duration):
        """
        Run the memory-efficient probing strategy for the specified duration.
        
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
        
        logging.info(f"MemoryEfficientProber running for {effective_duration}s")
        
        current_time = self.simulator.virtual_time
        end_time = current_time + effective_duration
        
        self.probes_sent = 0
        self.probes_received = 0
        self.results = []
        self.cpu_usage = []
        self.memory_usage = []
        
        # For distribution estimation
        self.distribution_estimates = []
        window_size = 5.0  # Time window for distribution estimation
        window_measurements = []
        last_estimate_time = current_time
        
        # Process in 1-second increments for consistent CPU/memory measurements
        while current_time < end_time:
            try:
                # Get probe times for this second
                probe_times = self.probe_strategy(current_time)
                
                for probe_time in probe_times:
                    # Skip probes beyond our effective duration
                    if probe_time >= end_time:
                        continue
                        
                    try:
                        # CRITICAL CHANGE: Use send_probe_at instead of send_probe_packet
                        delay = self.simulator.send_probe_at(probe_time)
                        self.probes_sent += 1
                        
                        if delay is not None:
                            self.probes_received += 1
                            self.results.append((probe_time, delay))
                            window_measurements.append((probe_time, delay))
                        else:
                            logging.debug(f"Probe dropped at {probe_time}")
                    except Exception as e:
                        logging.error(f"Error sending probe at time {probe_time}: {e}")
                
                # Track system resources
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
                        std_delay = np.std(delays) if len(delays) > 1 else 0.1  # Avoid zero std
                        
                        # Store the distribution estimate with timestamp
                        self.distribution_estimates.append((current_time, mean_delay, std_delay))
                        last_estimate_time = current_time
                
                # Increment virtual time by 1 second
                self.simulator.virtual_time += 1.0
                current_time += 1.0
                
            except Exception as e:
                logging.error(f"Unexpected error in MemoryEfficientProber run: {e}", exc_info=True)
                current_time += 1.0
                self.simulator.virtual_time += 1.0
        
        # Add final distribution estimate if we have measurements
        if window_measurements and (not self.distribution_estimates or 
                                   self.distribution_estimates[-1][0] < current_time - window_size):
            delays = [m[1] for m in window_measurements[-int(window_size*self.probes_per_second):]]
            if delays:
                mean_delay = np.mean(delays)
                std_delay = np.std(delays) if len(delays) > 1 else 0.1
                self.distribution_estimates.append((current_time, mean_delay, std_delay))
        
        logging.info(f"MemoryEfficientProber completed: sent {self.probes_sent}, received {self.probes_received}")
        
        return {
            "name": self.name,
            "results": self.results,
            "distribution_estimates": self.distribution_estimates,
            "probes_sent": self.probes_sent,
            "probes_received": self.probes_received,
            "loss_rate": 1.0 - (self.probes_received / max(1, self.probes_sent)),
            "runtime": effective_duration,  # Use the effective duration, not the actual runtime
            "avg_cpu": np.mean(self.cpu_usage) if self.cpu_usage else 0,
            "avg_memory": np.mean(self.memory_usage) if self.memory_usage else 0,
            "max_memory": max(self.memory_usage) if self.memory_usage else 0,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage
        }