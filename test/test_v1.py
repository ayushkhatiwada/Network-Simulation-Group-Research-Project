import unittest
import random
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../active_monitoring_evolution_2')))
from active_simulator_v1 import ActiveSimulator_v1


class TestActiveSimulatorV1(unittest.TestCase):
    def setUp(self):
        # Use a fixed random seed for reproducibility
        self.simulator = ActiveSimulator_v1(random_seed=42)
        
    def test_initialization(self):
        """Test that the simulator initializes correctly with appropriate defaults"""
        self.assertEqual(self.simulator.drop_probability, 0.1)
        self.assertEqual(self.simulator.paths, "1")
        self.assertEqual(self.simulator.max_probes_per_second, 10)
        self.assertEqual(self.simulator.max_departure_time, 100.0)
        
    def test_packet_drops(self):
        """Test that packets are dropped according to the drop probability"""
        # Set a deterministic seed for this test
        random.seed(42)
        
        # Create a new simulator with a 100% drop rate to guarantee drops
        drop_sim = ActiveSimulator_v1(random_seed=42)
        drop_sim.drop_probability = 1.0
        
        # Send a probe - it should be dropped (return None)
        result = drop_sim.send_probe_at(5.0)
        self.assertIsNone(result)
        
        # Check the event log correctly records the dropped packet
        self.assertEqual(len(drop_sim.event_log), 1)
        departure_time, arrival_time, delay = drop_sim.event_log[0]
        self.assertEqual(departure_time, 5.0)
        self.assertIsNone(arrival_time)
        self.assertIsNone(delay)
        
        # Create another simulator with 0% drop rate
        no_drop_sim = ActiveSimulator_v1(random_seed=42)
        no_drop_sim.drop_probability = 0.0
        
        # Send a probe - it should not be dropped
        result = no_drop_sim.send_probe_at(5.0)
        self.assertIsNotNone(result)
        
        # Check the event log correctly records the successful packet
        self.assertEqual(len(no_drop_sim.event_log), 1)
        departure_time, arrival_time, delay = no_drop_sim.event_log[0]
        self.assertEqual(departure_time, 5.0)
        self.assertIsNotNone(arrival_time)
        self.assertIsNotNone(delay)
    
    def test_rate_limiting(self):
        """Test that rate limiting is enforced"""
        simulator = ActiveSimulator_v1(random_seed=42)
        simulator.drop_probability = 0.0  # Ensure no drops for this test
        
        # Send max_probes_per_second probes in the same second
        for i in range(simulator.max_probes_per_second):
            simulator.send_probe_at(1.0 + 0.01 * i)
            
        # The next probe in the same second should raise an exception
        with self.assertRaises(Exception) as context:
            simulator.send_probe_at(1.99)
        
        self.assertTrue("Probe rate limit exceeded" in str(context.exception))
        
    def test_time_constraints(self):
        """Test that time constraints are enforced"""
        simulator = ActiveSimulator_v1(random_seed=42)
        
        # Test lower bound
        with self.assertRaises(ValueError) as context:
            simulator.send_probe_at(-1.0)
        self.assertTrue("between 0 and 100" in str(context.exception))
        
        # Test upper bound
        with self.assertRaises(ValueError) as context:
            simulator.send_probe_at(101.0)
        self.assertTrue("between 0 and 100" in str(context.exception))
        
    def test_reproducibility(self):
        """Test that with the same seed, packet drops are reproducible"""
        # Create two simulators with the same seed
        sim1 = ActiveSimulator_v1(random_seed=99)
        sim2 = ActiveSimulator_v1(random_seed=99)
        
        # Set a drop probability where we expect some drops and some successes
        sim1.drop_probability = 0.5
        sim2.drop_probability = 0.5
        
        # Send multiple probes and compare results
        departure_times = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        
        results1 = []
        results2 = []
        
        for dt in departure_times:
            results1.append(sim1.send_probe_at(dt))
        
        for dt in departure_times:
            results2.append(sim2.send_probe_at(dt))
        
        # Instead of comparing the exact values, compare which ones are None (dropped)
        # and which ones are not None (successful)
        is_dropped1 = [result is None for result in results1]
        is_dropped2 = [result is None for result in results2]
        
        # The drop pattern should be identical
        self.assertEqual(is_dropped1, is_dropped2)
        
    def test_send_multiple_probes(self):
        """Test that the send_multiple_probes method works with drops"""
        simulator = ActiveSimulator_v1(random_seed=42)
        simulator.drop_probability = 0.5  # 50% drop rate for a mix of drops and successes
        
        # Send multiple probes
        departure_times = [10.0, 11.0, 12.0, 13.0, 14.0]
        delays = simulator.send_multiple_probes(departure_times)
        
        # Check that we get a list of delays, with possible None values
        self.assertEqual(len(delays), len(departure_times))
        
        # Check the event log contains all the probes
        self.assertEqual(len(simulator.event_log), len(departure_times))
        
        # Verify some are dropped (None) and some succeed (not None)
        # With a 50% drop rate and random seed 42, we should have a mix
        dropped = [d for d in delays if d is None]
        succeeded = [d for d in delays if d is not None]
        
        # We should have both drops and successes with 50% drop rate
        # Note: This could theoretically fail if extremely unlucky, but very unlikely
        self.assertTrue(len(dropped) > 0, "Expected some packets to be dropped")
        self.assertTrue(len(succeeded) > 0, "Expected some packets to succeed")

if __name__ == "__main__":
    unittest.main()
