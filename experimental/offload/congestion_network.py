import time
import random
import threading
import numpy as np
from network.ground_truth import GroundTruthNetwork
import networkx as nx
from typing import List
from .active_simulator_v2 import ActiveSimulator_v2


class CongestionNetwork(ActiveSimulator_v2):
    """
    alt congestion model;
    - Exponential delay scaling based on congestion intensity
    - Dynamic packet generation during congestion
    this means we can experiment with hybrid systems (packet generation at higher congestion)
    and also profile the system under different congestion levels (i.e find the distribution when we are congested)
    """
    
    def __init__(self, paths="1"):
        super().__init__(paths)

        try:
            self.path = nx.shortest_path(self.network.graph, 
                                   source=self.network.SOURCE,
                                   target=self.network.DESTINATION)
            self.edge_delays = {edge: [] for edge in zip(self.path[:-1], self.path[1:])}
            
        except nx.NetworkXNoPath:
            print("No path found between specified nodes!")
            print("Available nodes:", sorted(list(self.network.graph.nodes())))
            
            nodes = sorted(list(self.network.graph.nodes()))
            if len(nodes) >= 2:
                self.network.SOURCE = nodes[0]
                self.network.DESTINATION = nodes[-1]
                print(f"Using nodes {self.network.SOURCE} and {self.network.DESTINATION} instead")
                self.path = nx.shortest_path(self.network.graph, 
                                       source=self.network.SOURCE,
                                       target=self.network.DESTINATION)
                self.edge_delays = {edge: [] for edge in zip(self.path[:-1], self.path[1:])}
        
        # higher intensity -> more packet drops, more delay, more passive generation
        self.congestion_intensity = 0.0  
        self.traffic_multiplier = 1.0
        self.delay_exponent = 2.5
        self.base_probe_rate = self.max_probes_per_second
        
        # Background traffic simulation
        self.background_traffic = {
            'active': False,
            'rate_multiplier': 1.0,
            'last_generated': time.time()
        }


    def update_congestion(self, intensity: float):
        self.congestion_intensity = max(0.0, min(1.0, intensity))
        
        # Exponential delay scaling
        self.congestion_delay_factor = 1 + (self.congestion_intensity ** self.delay_exponent)
        
        # Dynamic traffic generation
        self.traffic_multiplier = 1 + 2 * self.congestion_intensity
        self.max_probes_per_second = int(self.base_probe_rate * self.traffic_multiplier)
        
        # Adaptive drop probability
        self.congested_drop_probability = 0.1 + 0.6 * (self.congestion_intensity ** 1.8)

    def send_probe_at(self, departure_time: float) -> float:
        # Generate background traffic during congestion
        if self.congestion_intensity > 0.3 and not self.background_traffic['active']:
            self._start_background_traffic()
            
        # Apply exponential delay scaling
        try:
            delay = super().send_probe_at(departure_time)
            if delay is None:
                return None
                
            jitter = delay * np.random.uniform(0, 0.2 * self.congestion_intensity)
            return delay + jitter if random.random() > 0.5 else delay - jitter
            
        except Exception as e:
            if "Rate limit exceeded" in str(e):
                self._handle_congestion_overload()
            raise

    def _start_background_traffic(self):
        self.background_traffic.update({
            'active': True,
            'rate_multiplier': 2 + 3 * self.congestion_intensity,
            'last_generated': time.time()
        })
        
    def _handle_congestion_overload(self):
        self.max_probes_per_second = int(self.base_probe_rate * 0.8)
        time.sleep(0.1 * self.congestion_intensity)

    def get_true_params(self):
        # getter method for evaluation
        try:
            edge_data = self.network.graph[self.network.SOURCE][self.network.DESTINATION]
            
            if isinstance(edge_data, dict) and len(edge_data) > 0:
                edge_key = list(edge_data.keys())[0]
                edge_attrs = edge_data[edge_key]
                
                return edge_attrs.get('mean', 0), edge_attrs.get('std', 0)
        except (KeyError, IndexError):
            print(f"Warning: Could not find edge data between {self.network.SOURCE} and {self.network.DESTINATION}")
        
        # Default values if we can't get the parameters
        return 0, 0
    
    def get_distribution_parameters(self):
        
        #Get the true distribution parameters for the current network state.
        #This accounts for congestion effects on the delay distribution.
        
        #placejholders
        base_mu = 0.1  
        base_sigma = 0.01  
        
        try:
            # Calculate end-to-end parameters by combining edge parameters
            total_mu = 0
            total_var = 0
            
            if hasattr(self, 'network') and hasattr(self.network, 'graph'):
                for i in range(len(self.path) - 1):
                    u, v = self.path[i], self.path[i+1]
                    edge_params = self.network.get_edge_parameters(u, v)
                    
                    if 'mean' in edge_params:
                        total_mu += edge_params['mean']
                        total_var += edge_params['std']**2
                    elif 'mu' in edge_params:
                        total_mu += edge_params['mu']
                        total_var += edge_params['sigma']**2
                
                if total_mu > 0:
                    base_mu = total_mu
                    base_sigma = np.sqrt(total_var)
        except Exception as e:
            print(f"Error getting base parameters: {str(e)}")
        
        # Apply congestion effects
        if self.congestion_intensity > 0:
            # Scale parameters based on congestion intensity
            congestion_factor = self.congestion_delay_factor
            mu = base_mu * congestion_factor
            sigma = base_sigma * congestion_factor
            
            print(f"True parameters with congestion {self.congestion_intensity:.1f}: "
                  f"μ={mu:.4f}, σ={sigma:.4f} (base: μ={base_mu:.4f}, σ={base_sigma:.4f})")
        else:
            mu = base_mu
            sigma = base_sigma
            print(f"True parameters (normal): μ={mu:.4f}, σ={sigma:.4f}")
        
        return {'mu': mu, 'sigma': sigma}

    def compare_distribution_parameters(self, pred_mean: float, pred_std: float) -> float:
        true_params = self.get_distribution_parameters()
        return super().compare_distribution_parameters(
            pred_mean * (1 + self.congestion_intensity/2),
            pred_std * (1 + self.congestion_intensity)
        )

    def _backup_original_params(self):
        """Store original distribution parameters for all edges."""
        params = {}
        for u, v in self.graph.edges():
            if self.distribution_type == "lognormal":
                params[(u, v)] = {
                    'mu': self.graph[u][v]['mu'],
                    'sigma': self.graph[u][v]['sigma']
                }
            elif self.distribution_type == "gamma":
                params[(u, v)] = {
                    'shape': self.graph[u][v]['shape'],
                    'scale': self.graph[u][v]['scale']
                }
            elif self.distribution_type == "normal":
                params[(u, v)] = {
                    'mean': self.graph[u][v]['mean'],
                    'std': self.graph[u][v]['std']
                }
        return params
    
    def set_congestion(self, enabled, level=None):
        if enabled and level is not None:
            self.congestion_intensity = max(0.0, min(1.0, level))
            self._apply_congestion_effects()
        self.congestion_enabled = enabled
        
        # If congestion is disabled, reset to original parameters
        if not enabled:
            self._restore_original_params()
    
    def _apply_congestion_effects(self):
        for (u, v) in self.graph.edges():
            params = self.original_params[(u, v)]
            congestion_factor = 1 + 2 * self.congestion_intensity
            
            if self.distribution_type == "lognormal":
                self.graph[u][v]['mu'] = params['mu'] + np.log(congestion_factor)
                self.graph[u][v]['sigma'] = params['sigma'] * (1 + 0.3*self.congestion_intensity)
            elif self.distribution_type == "normal":
                self.graph[u][v]['mean'] = params['mean'] * congestion_factor
                self.graph[u][v]['std'] = params['std'] * (1 + 0.2*self.congestion_intensity)
            elif self.distribution_type == "gamma":
                self.graph[u][v]['shape'] = params['shape'] * (1 + 0.5*self.congestion_intensity)
                self.graph[u][v]['scale'] = params['scale'] * congestion_factor
    
    def _restore_original_params(self):
        """Restore original network parameters."""
        for (u, v), params in self.original_params.items():
            for param_name, param_value in params.items():
                self.graph[u][v][param_name] = param_value
    
    def _get_drop_probability(self):
        # Use exponential curve for realistic behavior:
        # - Very low drops until ~30% congestion
        # - Rapid increase after ~70% congestion
        if not self.congestion_enabled:
            return 0.0
            
        # almost no drops below 30% congestion,
        # rapid increase after 70% congestion
        if self.congestion_intensity < 0.3:
            return self.congestion_intensity * 0.05  # Max 1.5% drop at 30% congestion
        elif self.congestion_intensity < 0.7:
            return 0.015 + (self.congestion_intensity - 0.3) * 0.2  # 1.5% to 9.5% between 30-70% congestion
        else:
            # Exponential increase for severe congestion
            base = 0.095  # 9.5% at 70% congestion
            max_drop = 0.5  # 50% max drop rate at 100% congestion
            return base + (self.congestion_intensity - 0.7) * (max_drop - base) / 0.3
    
    def sample_edge_delay(self, u, v):
        
        #Override to add packet dropping based on congestion level.
    
        self._record_packet()
        
        # Check for packet drop based on congestion
        if self.congestion_enabled and random.random() < self._get_drop_probability():
            self.traffic_stats['dropped_packets'] += 1
            return None  # Indicate packet drop
            
        # Use parent class to sample the delay (already adjusted for congestion)
        return super().sample_edge_delay(u, v)
    
    def _record_packet(self):
        self.traffic_stats['total_packets'] += 1
        
        # Record timestamp for rolling window calculation
        now = time.time()
        self.packet_timestamps.append(now)
        
        # Remove timestamps older than 60 seconds
        cutoff = now - 60
        while self.packet_timestamps and self.packet_timestamps[0] < cutoff:
            self.packet_timestamps.pop(0)
            
        # Update last minute packet count
        self.traffic_stats['last_minute_packets'] = len(self.packet_timestamps)
        self.traffic_stats['timestamp'] = now
    
    def simulate_traffic(self, num_flows, switches, intensity=1.0):

        # Adjust num_flows based on intensity
        adjusted_flows = int(num_flows * intensity)
        if adjusted_flows <= 0:
            return 0, 0  # No traffic to generate
            
        # Record start state
        initial_total = self.traffic_stats['total_packets']
        initial_dropped = self.traffic_stats['dropped_packets']
        
        # Generate traffic using parent method
        successful_flows = 0
        for i in range(adjusted_flows):
            try:
                path = nx.shortest_path(self.graph, source=self.SOURCE, target=self.DESTINATION)
            except nx.NetworkXNoPath:
                print(f"No path found for flow {i+1}.")
                continue

            packet = {
                'src_ip': f"10.0.0.{self.SOURCE}",
                'dst_ip': f"10.0.0.{self.DESTINATION}",
                'src_port': random.randint(1024, 65535),
                'dst_port': random.choice([80, 443]),
                'protocol': 'TCP',
                'flow_id': f"flow_{i}_{time.time()}"
            }
            
            flow_dropped = False
            print(f"\nFlow {i+1}: Path from node {self.SOURCE} to node {self.DESTINATION} via {path}")

            for idx, node in enumerate(path):
                if idx == 0:
                    role = "ingress"
                elif idx == len(path) - 1:
                    role = "egress"
                else:
                    role = "forward"

                # Process packet in switch if available
                if node in switches:
                    switches[node].process_packet(packet, role)

                if idx < len(path) - 1:
                    next_node = path[idx + 1]
                    delay = self.sample_edge_delay(node, next_node)
                    
                    # Check if packet was dropped
                    if delay is None:
                        print(f"  [DROP] Packet dropped between {node} and {next_node} due to congestion")
                        flow_dropped = True
                        break
                        
                    print(f"  Link from {node} to {next_node} delay: {delay:.2f} ms")
                    time.sleep(delay / 1000.0)  # Convert ms to seconds
            
            if not flow_dropped:
                successful_flows += 1
        
        # Calculate statistics
        total_generated = self.traffic_stats['total_packets'] - initial_total
        total_dropped = self.traffic_stats['dropped_packets'] - initial_dropped
        
        return successful_flows, total_dropped
    
    def get_congestion_stats(self):
        return {
            'congestion_level': self.congestion_intensity,
            'drop_probability': self._get_drop_probability(),
            'traffic_stats': self.traffic_stats.copy()
        }
    
    def start_background_traffic(self, switches, flows_per_interval=10, interval=5.0):

        def traffic_generator():
            running = True
            while running:
                try:
                    # Traffic intensity varies with congestion
                    if self.congestion_enabled:
                        intensity = 1.0 + self.congestion_intensity
                    self.simulate_traffic(flows_per_interval, switches, intensity)
                    time.sleep(interval)
                except Exception as e:
                    print(f"Error in background traffic: {e}")
                    running = False
        
        thread = threading.Thread(target=traffic_generator)
        thread.daemon = True
        thread.start()
        return thread
    
    def get_edge_parameters(self, u, v):

        if v not in self.graph[u]:
            raise ValueError(f"No direct edge exists between nodes {u} and {v}")
        
        if self.distribution_type == "lognormal":
            return {
                'mu': self.graph[u][v]['mu'],
                'sigma': self.graph[u][v]['sigma']
            }
        elif self.distribution_type == "gamma":
            return {
                'shape': self.graph[u][v]['shape'],
                'scale': self.graph[u][v]['scale']
            }
        elif self.distribution_type == "normal":
            return {
                'mean': self.graph[u][v]['mean'],
                'std': self.graph[u][v]['std']
            }
        return {}
    
    def get_path(self, source: int, target: int) -> List[int]:
        try:
            return nx.shortest_path(self.graph, source=source, target=target)
        except nx.NetworkXNoPath:
            raise ValueError(f"No path between {source} and {target}") from None

    def simulate_traffic(self, *args, **kwargs):
        try:
            path = self.get_path(self.SOURCE, self.DESTINATION)
        except ValueError as e:
            print(f"Traffic simulation failed: {str(e)}")
            return 0, 0
            
        return super().simulate_traffic(*args, **kwargs)

    def update_delays(self):
        self.current_delays = [
            d * (1 + 2.5*self.congestion_intensity**2) 
            for d in self.base_delays
        ]
        
    def send_probe(self, time: float) -> float:
        if self._is_congested(time):
            delay = self._get_congested_delay()
            drop_prob = 0.4 * self.congestion_intensity
        else:
            delay = self._get_normal_delay()
            drop_prob = 0.1
            
        return delay if random.random() > drop_prob else None

    def compare_estimate(self, mu_pred: float, sigma_pred: float) -> float:
        true_mu, true_sigma = self.get_true_params()
        return 0.5 * (
            (sigma_pred/true_sigma)**2 + 
            (true_mu - mu_pred)**2/true_sigma**2 - 
            1 + np.log(true_sigma**2/sigma_pred**2)
        )

    def _is_congested(self, time_point):
        return any(start <= time_point <= end for start, end in self.congestion_intervals)

    def _get_congested_delay(self):
        base_delay = self.measure_end_to_end_delay()
        return base_delay * self.congestion_delay_factor

    def _get_normal_delay(self):
        return self.measure_end_to_end_delay()