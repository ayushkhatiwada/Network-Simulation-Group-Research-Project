import logging
import time
import numpy as np
import psutil
import threading
from abc import ABC, abstractmethod
from collections import deque

class BaseProber(ABC):
    def __init__(self, simulator, analysis_window=10.0, name="BaseProber"):
        """
        Initialize the prober.
        
        Parameters:
        -----------
        simulator: ActiveSimulator
            The simulator to probe
        analysis_window: float
            Time window for analyzing results in seconds
        name: str
            Name of the prober for identification
        """
        self.simulator = simulator
        self.analysis_window = analysis_window
        self.name = name
        
        # Statistics
        self.probes_sent = 0
        self.probes_received = 0
        self.start_time = None
        self.end_time = None
        
        # Memory-efficient storage using deque with max length
        self.recent_probes = deque(maxlen=1000)  # Store recent probe results
        self.results = []  # Store analysis results (timestamp, mean, std, kl_div)
        
        # Resource tracking
        self.cpu_usage = []
        self.memory_usage = []
        self.resource_tracker = None
        self.tracking = False
    
    @abstractmethod
    def probe_strategy(self, current_time):
        """
        Define the probing strategy.
        
        Parameters:
        -----------
        current_time: float
            Current simulation time
            
        Returns:
        --------
        list of float: Times at which to send probes
        """
        pass
    
    def _track_resources(self):
        """Track CPU and memory usage during probing"""
        process = psutil.Process()
        
        while self.tracking:
            # Get CPU and memory usage
            cpu_percent = process.cpu_percent(interval=0.5)
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
            
            # Store measurements
            self.cpu_usage.append(cpu_percent)
            self.memory_usage.append(memory_mb)
            
            # Sleep to avoid excessive measurements
            time.sleep(0.5)
    
    def run(self, duration):
        """
        Run the prober for the specified duration.
        
        Parameters:
        -----------
        duration: float
            Duration to run in seconds
            
        Returns:
        --------
        dict: Results and statistics
        """
        self.start_time = time.time()
        current_time = 0.0
        
        # Start resource tracking
        self.tracking = True
        self.resource_tracker = threading.Thread(target=self._track_resources)
        self.resource_tracker.daemon = True
        self.resource_tracker.start()
        
        # Previous distribution estimates
        prev_mean = None
        prev_std = None
        
        # Run for the specified duration
        while current_time < duration:
            # Get probe times from strategy
            probe_times = self.probe_strategy(current_time)
            
            # Send probes
            for probe_time in probe_times:
                try:
                    delay = self.simulator.send_probe_at(probe_time)
                    self.probes_sent += 1
                    
                    if delay is not None:
                        self.probes_received += 1
                        # Store probe result with timestamp
                        self.recent_probes.append((probe_time, delay))
                except Exception as e:
                    logging.warning(f"Error sending probe: {e}")
            
            # Analyze results periodically
            if current_time % self.analysis_window < 0.001:
                mean, std, kl_div = self.analyze_recent_probes(current_time, prev_mean, prev_std)
                if mean is not None and std is not None:
                    prev_mean, prev_std = mean, std
            
            # Advance time
            time.sleep(1.0)  # Simulate passage of time
            current_time += 1.0
        
        # Final analysis
        self.analyze_recent_probes(current_time, prev_mean, prev_std)
        
        # Stop resource tracking
        self.tracking = False
        if self.resource_tracker.is_alive():
            self.resource_tracker.join(timeout=1.0)
        
        self.end_time = time.time()
        
        return {
            "name": self.name,
            "results": self.results,
            "probes_sent": self.probes_sent,
            "probes_received": self.probes_received,
            "loss_rate": 1.0 - (self.probes_received / max(1, self.probes_sent)),
            "runtime": self.end_time - self.start_time,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "avg_cpu": np.mean(self.cpu_usage) if self.cpu_usage else 0,
            "avg_memory": np.mean(self.memory_usage) if self.memory_usage else 0,
            "max_memory": max(self.memory_usage) if self.memory_usage else 0
        }
    
    def analyze_recent_probes(self, current_time, prev_mean=None, prev_std=None):
        """
        Analyze recent probe results to estimate distribution parameters.
        
        Parameters:
        -----------
        current_time: float
            Current simulation time
        prev_mean: float
            Previous mean estimate
        prev_std: float
            Previous std estimate
            
        Returns:
        --------
        tuple: (mean, std, kl_div)
        """
        # Get recent probes within analysis window
        recent_probes = [p for p in self.recent_probes 
                        if current_time - self.analysis_window <= p[0] < current_time]
        
        if not recent_probes:
            return None, None, None
        
        # Extract delays
        delays = [p[1] for p in recent_probes]
        
        # Calculate mean and std
        mean = np.mean(delays)
        std = np.std(delays)
        
        # Compare with ground truth
        kl_div = self.simulator.compare_distribution_parameters(mean, std, at_time=current_time)
        
        # Store results
        self.results.append((current_time, mean, std, kl_div))
        
        return mean, std, kl_div