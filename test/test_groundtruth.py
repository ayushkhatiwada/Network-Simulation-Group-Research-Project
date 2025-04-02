import unittest
import random
import statistics
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../active_probing_algos')))
from ground_truth import GroundTruthNetwork


class TestGroundTruthNetwork(unittest.TestCase):
    """Test cases for the GroundTruthNetwork class"""

    def setUp(self):
        """Set up test fixtures, if any"""
        # Set random seed for reproducible tests
        random.seed(42)

    def test_init_default(self):
        """Test initialization with default parameters"""
        # Default should be single edge (path="1")
        network = GroundTruthNetwork()
        
        # Check that nodes exist
        self.assertTrue(network.graph.has_node(1))
        self.assertTrue(network.graph.has_node(2))
        
        # Check SOURCE and DESTINATION are set correctly
        self.assertEqual(network.SOURCE, 1)
        self.assertEqual(network.DESTINATION, 2)
        
        # Check that there's exactly one edge between nodes 1 and 2
        edges = network.graph[1][2]
        self.assertEqual(len(edges), 1)
        
        # Check that the edge parameters are correct
        edge_data = next(iter(edges.values()))
        self.assertEqual(edge_data['mean'], 0.8)
        self.assertEqual(edge_data['std'], 0.15)

    def test_init_path1_string(self):
        """Test initialization with path="1" (string)"""
        network = GroundTruthNetwork(paths="1")
        
        # Check that there's exactly one edge between nodes 1 and 2
        edges = network.graph[1][2]
        self.assertEqual(len(edges), 1)
        
        # Check that the edge parameters are correct
        edge_data = next(iter(edges.values()))
        self.assertEqual(edge_data['mean'], 0.8)
        self.assertEqual(edge_data['std'], 0.15)

    def test_init_path1_int(self):
        """Test initialization with path=1 (integer)"""
        network = GroundTruthNetwork(paths=1)
        
        # Check that there's exactly one edge between nodes 1 and 2
        edges = network.graph[1][2]
        self.assertEqual(len(edges), 1)
        
        # Check that the edge parameters are correct
        edge_data = next(iter(edges.values()))
        self.assertEqual(edge_data['mean'], 0.8)
        self.assertEqual(edge_data['std'], 0.15)

    def test_init_path2_string(self):
        """Test initialization with path="2" (string)"""
        network = GroundTruthNetwork(paths="2")
        
        # Check that there are exactly two edges between nodes 1 and 2
        edges = network.graph[1][2]
        self.assertEqual(len(edges), 2)
        
        # Check that we can find both edges with their parameters
        edge_ids = set()
        mean_values = set()
        std_values = set()
        
        for edge_data in edges.values():
            edge_ids.add(edge_data.get('id', None))
            mean_values.add(edge_data['mean'])
            std_values.add(edge_data['std'])
        
        self.assertIn('edge_1', edge_ids)
        self.assertIn('edge_2', edge_ids)
        self.assertIn(0.8, mean_values)
        self.assertIn(0.5, mean_values)
        self.assertIn(0.15, std_values)
        self.assertIn(0.1, std_values)

    def test_init_path2_int(self):
        """Test initialization with path=2 (integer)"""
        network = GroundTruthNetwork(paths=2)
        
        # Check that there are exactly two edges between nodes 1 and 2
        edges = network.graph[1][2]
        self.assertEqual(len(edges), 2)
        
        # Check that we can find both edges with their parameters
        edge_ids = set()
        mean_values = set()
        std_values = set()
        
        for edge_data in edges.values():
            edge_ids.add(edge_data.get('id', None))
            mean_values.add(edge_data['mean'])
            std_values.add(edge_data['std'])
        
        self.assertIn('edge_1', edge_ids)
        self.assertIn('edge_2', edge_ids)
        self.assertIn(0.8, mean_values)
        self.assertIn(0.5, mean_values)
        self.assertIn(0.15, std_values)
        self.assertIn(0.1, std_values)

    def test_init_invalid_path(self):
        """Test initialization with invalid path parameter"""
        # Should fallback to default (single edge)
        network = GroundTruthNetwork(paths="invalid")
        
        # Check that there's exactly one edge between nodes 1 and 2
        edges = network.graph[1][2]
        self.assertEqual(len(edges), 1)

    def test_sample_edge_delay_single_edge(self):
        """Test sampling delay from a single edge"""
        random.seed(42)  # Ensure reproducibility
        network = GroundTruthNetwork(paths="1")
        
        # Sample multiple delays
        n_samples = 1000
        delays = [network.sample_edge_delay(1, 2) for _ in range(n_samples)]
        
        # Check that all delays are non-negative
        self.assertTrue(all(delay >= 0 for delay in delays))
        
        # Check that the mean and std are close to the expected values
        # (allowing for some sampling error)
        mean_delay = statistics.mean(delays)
        std_delay = statistics.stdev(delays)
        
        self.assertAlmostEqual(mean_delay, 0.8, delta=0.05)
        self.assertAlmostEqual(std_delay, 0.15, delta=0.05)

    def test_sample_edge_delay_dual_edge(self):
        """Test sampling delay from dual edges"""
        random.seed(42)  # Ensure reproducibility
        network = GroundTruthNetwork(paths="2")
        
        # Sample many delays to ensure we hit both edges
        n_samples = 2000
        delays = [network.sample_edge_delay(1, 2) for _ in range(n_samples)]
        
        # Check that all delays are non-negative
        self.assertTrue(all(delay >= 0 for delay in delays))
        
        # Histogram should roughly show two peaks around 0.5 and 0.8
        # Count samples in ranges around expected means
        count_near_05 = sum(1 for d in delays if 0.3 <= d <= 0.7)
        count_near_08 = sum(1 for d in delays if 0.6 <= d <= 1.0)
        
        # Both ranges should have a significant number of samples
        # (roughly n_samples/2 each, but allowing for randomness)
        self.assertGreater(count_near_05, n_samples * 0.3)
        self.assertGreater(count_near_08, n_samples * 0.3)

    def test_get_single_edge_distribution_parameters(self):
        """Test getting distribution parameters for a single edge"""
        network = GroundTruthNetwork(paths="1")
        
        params = network.get_single_edge_distribution_parameters(1, 2)
        
        self.assertEqual(params["mean"], 0.8)
        self.assertEqual(params["std"], 0.15)

    def test_get_single_edge_distribution_parameters_error(self):
        """Test error when getting single edge parameters for a multi-edge network"""
        network = GroundTruthNetwork(paths="2")
        
        # Should raise an assertion error as there are multiple edges
        with self.assertRaises(AssertionError):
            network.get_single_edge_distribution_parameters(1, 2)

    def test_get_multi_edge_distribution_parameters(self):
        """Test getting distribution parameters for multiple edges"""
        network = GroundTruthNetwork(paths="2")
        
        params = network.get_multi_edge_distribution_parameters(1, 2)
        
        # Should return a list of two parameter dictionaries
        self.assertEqual(len(params), 2)
        
        # Check that both expected parameter sets are present
        means = [p["mean"] for p in params]
        stds = [p["std"] for p in params]
        
        self.assertIn(0.8, means)
        self.assertIn(0.5, means)
        self.assertIn(0.15, stds)
        self.assertIn(0.1, stds)

    def test_get_multi_edge_distribution_parameters_single_edge(self):
        """Test getting multi-edge distribution parameters for a single-edge network"""
        network = GroundTruthNetwork(paths="1")
        
        params = network.get_multi_edge_distribution_parameters(1, 2)
        
        # Should return a list with one parameter dictionary
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0]["mean"], 0.8)
        self.assertEqual(params[0]["std"], 0.15)

if __name__ == "__main__":
    unittest.main()
