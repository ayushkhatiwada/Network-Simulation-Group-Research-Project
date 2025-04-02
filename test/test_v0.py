import unittest
import random
from io import StringIO
import sys
import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../active_probing_evolution_2')))
from active_simulator_v0 import ActiveSimulator_v0


class TestActiveSimulator(unittest.TestCase):
    """
    Test cases for the ActiveSimulator_v0 class, covering all features.
    """
    
    def setUp(self):
        """Set up test fixtures"""
        # Redirect stdout to capture print outputs
        self.held_output = StringIO()
        self.original_stdout = sys.stdout
        sys.stdout = self.held_output
        
    def tearDown(self):
        """Tear down test fixtures"""
        # Restore stdout
        sys.stdout = self.original_stdout
    
    # Distribution Parameters Tests for Single-Edge Network
    
    def test_compare_distribution_parameters_correct_values(self):
        """Test compare_distribution_parameters with correct expected values"""
        # Create simulator with path 1 and fixed seed
        simulator = ActiveSimulator_v0(paths="1", random_seed=42)
        
        # The ground truth values for path 1 are mean=0.8, std=0.15
        # Test with exact values
        kl_div = simulator.compare_distribution_parameters(0.8, 0.15)
        
        # KL divergence should be 0 when predictions match exactly
        self.assertAlmostEqual(kl_div, 0.0, places=5)
        self.assertIn("KL divergence: 0.0000 ✅", self.held_output.getvalue())
    
    def test_compare_distribution_parameters_close_values(self):
        """Test compare_distribution_parameters with close values (should pass)"""
        simulator = ActiveSimulator_v0(paths="1", random_seed=42)
        
        # Test with slightly off values that should still pass the 0.05 threshold
        kl_div = simulator.compare_distribution_parameters(0.82, 0.155)
        
        # Calculate expected KL divergence
        actual_mean, actual_std = 0.8, 0.15
        pred_mean, pred_std = 0.82, 0.155
        expected_kl = math.log(pred_std / actual_std) + ((actual_std**2 + (actual_mean - pred_mean)**2) / (2 * pred_std**2)) - 0.5
        
        # Verify KL divergence calculation
        self.assertAlmostEqual(kl_div, expected_kl, places=5)
        self.assertLess(kl_div, 0.05)  # Should be below threshold
        self.assertIn("✅", self.held_output.getvalue())
    
    def test_compare_distribution_parameters_bad_values(self):
        """Test compare_distribution_parameters with values that exceed threshold"""
        simulator = ActiveSimulator_v0(paths="1", random_seed=42)
        
        # Test with values that should fail the 0.05 threshold
        kl_div = simulator.compare_distribution_parameters(1.0, 0.2)
        
        # This should be above the threshold
        self.assertGreater(kl_div, 0.05)
        self.assertIn("❌", self.held_output.getvalue())
    
    def test_compare_distribution_parameters_wrong_path(self):
        """Test compare_distribution_parameters with wrong path should raise error"""
        simulator = ActiveSimulator_v0(paths="2", random_seed=42)
        
        # Should raise ValueError because we're using path 2 with the single-path comparison
        with self.assertRaises(ValueError) as context:
            simulator.compare_distribution_parameters(0.8, 0.15)
        
        self.assertIn("can only be used with a single-edge network", str(context.exception))
    
    # Distribution Parameters Tests for Dual-Edge Network
    
    def test_compare_distribution_parameters_2_correct_values(self):
        """Test compare_distribution_parameters_2 with correct expected values"""
        simulator = ActiveSimulator_v0(paths="2", random_seed=42)
        
        # The ground truth values for path 2 are:
        # edge 1: mean=0.8, std=0.15
        # edge 2: mean=0.5, std=0.1
        kl_divs = simulator.compare_distribution_parameters_2(0.8, 0.5, 0.15, 0.1)
        
        # Both KL divergences should be 0 when predictions match exactly
        self.assertEqual(len(kl_divs), 2)
        self.assertAlmostEqual(kl_divs[0], 0.0, places=5)
        self.assertAlmostEqual(kl_divs[1], 0.0, places=5)
        self.assertIn("KL divergence for edge 1: 0.0000 ✅", self.held_output.getvalue())
        self.assertIn("KL divergence for edge 2: 0.0000 ✅", self.held_output.getvalue())
    
    def test_compare_distribution_parameters_2_close_values(self):
        """Test compare_distribution_parameters_2 with close values"""
        simulator = ActiveSimulator_v0(paths="2", random_seed=42)
        
        # Test with slightly off values that should still pass the 0.05 threshold
        kl_divs = simulator.compare_distribution_parameters_2(0.82, 0.48, 0.155, 0.105)
        
        # Both should be below the threshold
        self.assertEqual(len(kl_divs), 2)
        self.assertLess(kl_divs[0], 0.05)
        self.assertLess(kl_divs[1], 0.05)
        self.assertIn("✅", self.held_output.getvalue())
    
    def test_compare_distribution_parameters_2_mixed_values(self):
        """Test compare_distribution_parameters_2 with mixed results (one pass, one fail)"""
        simulator = ActiveSimulator_v0(paths="2", random_seed=42)
        
        # First edge should pass, second should fail
        kl_divs = simulator.compare_distribution_parameters_2(0.81, 0.65, 0.15, 0.12)
        
        self.assertEqual(len(kl_divs), 2)
        self.assertLess(kl_divs[0], 0.05)  # First should pass
        self.assertGreater(kl_divs[1], 0.05)  # Second should fail
        self.assertIn("edge 1: ", self.held_output.getvalue())
        self.assertIn("✅", self.held_output.getvalue())
        self.assertIn("❌", self.held_output.getvalue())
    
    def test_compare_distribution_parameters_2_wrong_path(self):
        """Test compare_distribution_parameters_2 with wrong path should raise error"""
        simulator = ActiveSimulator_v0(paths="1", random_seed=42)
        
        # Should raise ValueError because we're using path 1 with the dual-path comparison
        with self.assertRaises(ValueError) as context:
            simulator.compare_distribution_parameters_2(0.8, 0.5, 0.15, 0.1)
        
        self.assertIn("can only be used with a dual-edge network", str(context.exception))
    
    # Random Seed Tests
    
    def test_random_seed_reproducibility(self):
        """Test that using the same random seed produces the same delays"""
        # Reset the global random seed before each test
        random.seed(42)
        
        # Create first simulator and send probes
        sim1 = ActiveSimulator_v0(paths="1", random_seed=42)
        times = [0.5, 1.5, 2.5, 3.5, 4.5]
        delays1 = sim1.send_multiple_probes(times)
        
        # Reset random state again to ensure complete reproducibility
        random.seed(42)
        
        # Create second simulator and send probes
        sim2 = ActiveSimulator_v0(paths="1", random_seed=42)
        delays2 = sim2.send_multiple_probes(times)
        
        # Delays should be identical
        for i, (d1, d2) in enumerate(zip(delays1, delays2)):
            self.assertEqual(d1, d2, f"Delay mismatch at index {i}: {d1} != {d2}")
    
    def test_different_random_seeds(self):
        """Test that different random seeds produce different delays"""
        # Create two simulators with different seeds
        sim1 = ActiveSimulator_v0(paths="1", random_seed=42)
        sim2 = ActiveSimulator_v0(paths="1", random_seed=43)
        
        # Send probes at the same time
        time_point = 1.0
        delay1 = sim1.send_probe_at(time_point)
        delay2 = sim2.send_probe_at(time_point)
        
        # Delays should be different with high probability
        # (There's a tiny chance they could be the same by coincidence)
        self.assertNotEqual(delay1, delay2)
    
    # Probe Sending Tests
    
    def test_send_probe_at(self):
        """Test sending a single probe at a specific time"""
        simulator = ActiveSimulator_v0(paths="1", random_seed=42)
        
        # Send a probe and verify results
        delay = simulator.send_probe_at(5.0)
        
        # Verify delay is a positive float
        self.assertIsInstance(delay, float)
        self.assertGreaterEqual(delay, 0.0)
        
        # Verify event log contains the probe
        event_log = simulator.get_event_log()
        self.assertEqual(len(event_log), 1)
        
        departure_time, arrival_time, logged_delay = event_log[0]
        self.assertEqual(departure_time, 5.0)
        self.assertEqual(arrival_time, 5.0 + delay)
        self.assertEqual(logged_delay, delay)
    
    def test_send_multiple_probes(self):
        """Test sending multiple probes"""
        simulator = ActiveSimulator_v0(paths="1", random_seed=42)
        
        # Send multiple probes and verify results
        times = [1.0, 2.0, 3.0]
        delays = simulator.send_multiple_probes(times)
        
        # Verify we got the right number of delays
        self.assertEqual(len(delays), len(times))
        
        # Verify event log contains all probes in order
        event_log = simulator.get_event_log()
        self.assertEqual(len(event_log), len(times))
        
        for i, (departure_time, arrival_time, logged_delay) in enumerate(event_log):
            self.assertEqual(departure_time, times[i])
            self.assertEqual(arrival_time, times[i] + delays[i])
            self.assertEqual(logged_delay, delays[i])
    
    def test_send_multiple_probes_unordered(self):
        """Test sending multiple probes with unordered times"""
        simulator = ActiveSimulator_v0(paths="1", random_seed=42)
        
        # Send probes in unordered times
        unordered_times = [3.0, 1.0, 2.0]
        expected_order = sorted(unordered_times)  # [1.0, 2.0, 3.0]
        
        delays = simulator.send_multiple_probes(unordered_times)
        
        # Verify event log has probes in ordered times
        event_log = simulator.get_event_log()
        self.assertEqual(len(event_log), len(unordered_times))
        
        for i, (departure_time, _, _) in enumerate(event_log):
            self.assertEqual(departure_time, expected_order[i])
    
    # Rate Limiting Tests
    
    def test_rate_limiting(self):
        """Test that rate limiting works properly"""
        simulator = ActiveSimulator_v0(paths="1", random_seed=42)
        
        # Send maximum allowed probes in one second
        max_probes = simulator.max_probes_per_second
        times = [1.0 + 0.01 * i for i in range(max_probes)]  # 1.00, 1.01, 1.02, ... all in second 1
        
        # This should succeed
        delays = simulator.send_multiple_probes(times)
        self.assertEqual(len(delays), max_probes)
        
        # Trying to send one more in the same second should fail
        with self.assertRaises(Exception) as context:
            simulator.send_probe_at(1.99)  # Still in second 1
        
        self.assertIn("Probe rate limit exceeded", str(context.exception))
        
        # But sending in the next second should work
        delay = simulator.send_probe_at(2.0)  # Second 2
        self.assertIsInstance(delay, float)
    
    # Invalid Input Tests
    
    def test_invalid_departure_time_negative(self):
        """Test that negative departure times are rejected"""
        simulator = ActiveSimulator_v0(paths="1", random_seed=42)
        
        with self.assertRaises(ValueError) as context:
            simulator.send_probe_at(-1.0)
        
        self.assertIn("Departure time must be between 0 and", str(context.exception))
    
    def test_invalid_departure_time_too_large(self):
        """Test that departure times beyond the maximum are rejected"""
        simulator = ActiveSimulator_v0(paths="1", random_seed=42)
        max_time = simulator.max_departure_time
        
        with self.assertRaises(ValueError) as context:
            simulator.send_probe_at(max_time + 1)
        
        self.assertIn("Departure time must be between 0 and", str(context.exception))
    
    # Time Caching Tests
    
    def test_time_caching(self):
        """Test that sending probes at the same time returns the same delay"""
        simulator = ActiveSimulator_v0(paths="1", random_seed=42)
        
        # Send probe at specific time
        time_point = 5.0
        first_delay = simulator.send_probe_at(time_point)
        
        # Clear event log to send another probe at the same time
        simulator.event_log = []
        
        # Send another probe at the same time
        second_delay = simulator.send_probe_at(time_point)
        
        # Delays should be identical due to caching
        self.assertEqual(first_delay, second_delay)
        
        # But different times should have different delays
        different_time = 6.0
        different_delay = simulator.send_probe_at(different_time)
        
        # With high probability, the delay should be different
        # (There's a tiny chance they could be the same by coincidence)
        self.assertNotEqual(first_delay, different_delay)

if __name__ == "__main__":
    unittest.main()
