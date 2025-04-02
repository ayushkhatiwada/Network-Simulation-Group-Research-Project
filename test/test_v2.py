import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../active_probing_evolution_2')))
from active_simulator_v2 import ActiveSimulator_v2


class TestActiveSimulatorV2(unittest.TestCase):
    def setUp(self):
        # Use a fixed random seed for reproducibility
        self.simulator = ActiveSimulator_v2(random_seed=42)
        
    def test_initialization(self):
        """Test that the simulator initializes correctly with appropriate defaults"""
        self.assertEqual(self.simulator.normal_drop_probability, 0.1)
        self.assertEqual(self.simulator.congested_drop_probability, 0.4)
        self.assertEqual(self.simulator.congestion_delay_factor, 1.5)
        self.assertEqual(self.simulator.paths, "1")
        self.assertEqual(self.simulator.max_probes_per_second, 10)
        self.assertEqual(self.simulator.max_departure_time, 100.0)
        self.assertIsNotNone(self.simulator.congestion_intervals)
        self.assertGreater(len(self.simulator.congestion_intervals), 0)
    
    def test_congestion_intervals(self):
        """Test that congestion intervals are generated correctly"""
        # Check that congestion intervals are within valid time range
        for start, end in self.simulator.congestion_intervals:
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(end, self.simulator.max_departure_time)
            self.assertLess(start, end)  # Start time should be less than end time
        
        # Check that congestion intervals don't overlap
        sorted_intervals = sorted(self.simulator.congestion_intervals)
        for i in range(len(sorted_intervals) - 1):
            _, end1 = sorted_intervals[i]
            start2, _ = sorted_intervals[i + 1]
            self.assertLess(end1, start2)
    
    def test_reproducibility(self):
        """Test that with the same seed, congestion intervals and behavior are reproducible"""
        # Create two simulators with the same seed
        sim1 = ActiveSimulator_v2(random_seed=99)
        sim2 = ActiveSimulator_v2(random_seed=99)
        
        # Congestion intervals should be identical
        self.assertEqual(sim1.congestion_intervals, sim2.congestion_intervals)
        
        # Send probes at various times and check for reproducible behavior
        departure_times = [0.5, 5.5, 10.5, 15.5, 20.5, 25.5, 30.5, 35.5, 40.5, 45.5]
        
        # Check if each time is in a congestion interval for both simulators
        in_congestion1 = []
        in_congestion2 = []
        
        for t in departure_times:
            in_congestion1.append(any(start <= t <= end for start, end in sim1.congestion_intervals))
            in_congestion2.append(any(start <= t <= end for start, end in sim2.congestion_intervals))
        
        self.assertEqual(in_congestion1, in_congestion2)
        
        # Now test that drops are reproducible
        results1 = []
        results2 = []
        
        for t in departure_times:
            results1.append(sim1.send_probe_at(t) is None)  # True if dropped
            results2.append(sim2.send_probe_at(t) is None)  # True if dropped
            
        self.assertEqual(results1, results2)
    
    def test_congestion_effect(self):
        """Test that congestion affects delay and drop probability"""
        simulator = ActiveSimulator_v2(random_seed=42)
        
        # Set extreme values to make the test more predictable
        simulator.normal_drop_probability = 0.0  # No drops in normal state
        simulator.congested_drop_probability = 1.0  # Always drop in congested state
        simulator.congestion_delay_factor = 2.0  # Double delay in congestion
        
        # Find times in and out of congestion
        congested_time = None
        normal_time = None
        
        for t in range(100):
            t_float = float(t)
            is_congested = any(start <= t_float <= end for start, end in simulator.congestion_intervals)
            if is_congested and congested_time is None:
                congested_time = t_float
            elif not is_congested and normal_time is None:
                normal_time = t_float
                
            if congested_time is not None and normal_time is not None:
                break
        
        # Skip test if we can't find both time types (very unlikely)
        if congested_time is None or normal_time is None:
            self.skipTest("Couldn't find both congested and normal times")
            
        # Test that packets in congestion are dropped
        result_congested = simulator.send_probe_at(congested_time)
        self.assertIsNone(result_congested)  # Should be dropped
        
        # Reset simulator to test delay factor
        simulator = ActiveSimulator_v2(random_seed=42)
        simulator.normal_drop_probability = 0.0
        simulator.congested_drop_probability = 0.0  # No drops
        simulator.congestion_delay_factor = 2.0
        
        # Send probes at same time points and compare delays
        delay_normal = simulator.send_probe_at(normal_time)
        delay_congested = simulator.send_probe_at(congested_time)
        
        # Test the behavior with the same cached delay
        cached_delay_normal = simulator.time_cache[normal_time]
        cached_delay_congested = simulator.time_cache[congested_time]
        
        self.assertAlmostEqual(delay_normal, cached_delay_normal)
        self.assertAlmostEqual(delay_congested, cached_delay_congested * simulator.congestion_delay_factor)
    
    def test_rate_limiting(self):
        """Test that rate limiting is enforced"""
        simulator = ActiveSimulator_v2(random_seed=42)
        simulator.normal_drop_probability = 0.0  # Ensure no drops for this test
        simulator.congested_drop_probability = 0.0
        
        # Send max_probes_per_second probes in the same second
        for i in range(simulator.max_probes_per_second):
            simulator.send_probe_at(1.0 + 0.01 * i)
            
        # The next probe in the same second should raise an exception
        with self.assertRaises(Exception) as context:
            simulator.send_probe_at(1.99)
        
        self.assertTrue("Rate limit exceeded" in str(context.exception))
    
    def test_time_constraints(self):
        """Test that time constraints are enforced"""
        simulator = ActiveSimulator_v2(random_seed=42)
        
        # Test lower bound
        with self.assertRaises(ValueError) as context:
            simulator.send_probe_at(-1.0)
        self.assertTrue("Time must be in [0" in str(context.exception))
        
        # Test upper bound
        with self.assertRaises(ValueError) as context:
            simulator.send_probe_at(101.0)
        self.assertTrue("Time must be in [0" in str(context.exception))
    
    def test_event_log(self):
        """Test that event log correctly captures probe events"""
        simulator = ActiveSimulator_v2(random_seed=42)
        
        # Send a mix of probes, some in congestion, some not
        times = [10.0, 20.0, 30.0, 40.0, 50.0]
        results = []
        
        for t in times:
            results.append(simulator.send_probe_at(t))
        
        # Check event log entries
        event_log = simulator.get_event_log()
        self.assertEqual(len(event_log), len(times))
        
        for i, t in enumerate(times):
            entry = event_log[i]
            self.assertEqual(entry[0], t)  # Departure time
            
            if results[i] is None:
                # Dropped packet
                self.assertIsNone(entry[1])  # Arrival time
                self.assertIsNone(entry[2])  # Delay
            else:
                # Successful packet
                self.assertIsNotNone(entry[1])  # Arrival time
                self.assertIsNotNone(entry[2])  # Delay
                self.assertEqual(entry[1], t + entry[2])  # Arrival = Departure + Delay

if __name__ == "__main__":
    unittest.main()
