import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from tabulate import tabulate

class ProberEvaluator:
    """
    Evaluates and compares different probing strategies.
    """
    def __init__(self, simulator):
        """
        Initialize the evaluator.
        
        Parameters:
        -----------
        simulator: ActiveSimulator
            The simulator to use for evaluation
        """
        self.simulator = simulator
        self.results = {}
        self.ground_truth = {}
        
        # Get ground truth parameters
        self.update_ground_truth()
    
    def update_ground_truth(self):
        """Update ground truth parameters for different time points"""
        # Get the max simulation time from the simulator if available
        max_time = getattr(self.simulator, 'max_simulation_time', 100)
        
        # Sample ground truth at regular intervals
        for t in np.linspace(0, max_time, 100):
            params = self.simulator.get_current_distribution_parameters(
                self.simulator.network.SOURCE, 
                self.simulator.network.DESTINATION, 
                at_time=t
            )
            self.ground_truth[t] = params
    
    def evaluate_prober(self, prober, duration=100.0, reset_simulator=True):
        """
        Evaluate a probing system's performance.
        
        Parameters:
        -----------
        prober: BaseProber
            The probing system to evaluate
        duration: float
            Duration of the evaluation in seconds
        reset_simulator: bool
            Whether to reset the simulator before evaluation
            
        Returns:
        --------
        dict: Evaluation results
        """
        print(f"\nEvaluating {prober.name}...")
        
        # Reset simulator if requested
        if reset_simulator and hasattr(self.simulator, 'reset'):
            self.simulator.reset()
        
        # Run the prober
        result = prober.run(duration)
        
        # Calculate accuracy metrics if distribution_estimates are available
        if "distribution_estimates" in result and result["distribution_estimates"]:
            # Calculate accuracy metrics
            mean_errors = []
            std_errors = []
            
            for t, pred_mean, pred_std in result["distribution_estimates"]:
                # Find closest ground truth time point
                closest_t = min(self.ground_truth.keys(), key=lambda x: abs(x - t))
                
                # Get ground truth parameters
                gt_params = self.ground_truth[closest_t]
                gt_mean = gt_params["mean"]
                gt_std = gt_params["std"]
                
                # Calculate errors - avoid division by zero
                if gt_mean != 0:
                    mean_error = abs(pred_mean - gt_mean) / gt_mean
                else:
                    mean_error = abs(pred_mean)
                    
                if gt_std != 0:
                    std_error = abs(pred_std - gt_std) / gt_std
                else:
                    std_error = abs(pred_std)
                
                # Handle potential inf/nan values
                mean_error = min(mean_error, 100.0)  # Cap at 100 (10,000%) to avoid inf
                std_error = min(std_error, 100.0)
                
                mean_errors.append(mean_error)
                std_errors.append(std_error)
            
            # Only calculate if we have valid errors
            if mean_errors:
                result["accuracy"] = {
                    "mean_error": np.mean(mean_errors),
                    "std_error": np.mean(std_errors)
                }
            else:
                result["accuracy"] = {
                    "mean_error": 0.0,
                    "std_error": 0.0
                }
        else:
            # If no distribution estimates, set default accuracy metrics
            result["accuracy"] = {
                "mean_error": 0.0,
                "std_error": 0.0
            }
        
        # Store results
        self.results[prober.name] = result
        
        return result
    
    def _calculate_accuracy(self, results):
        """
        Calculate accuracy metrics by comparing with ground truth.
        
        Parameters:
        -----------
        results: list
            List of (timestamp, mean, std, kl_div) tuples
            
        Returns:
        --------
        dict: Accuracy metrics
        """
        if not results:
            return {"mean_error": float('inf'), "std_error": float('inf')}
        
        mean_errors = []
        std_errors = []
        
        for t, mean, std, _ in results:
            # Find closest ground truth time point
            closest_t = min(self.ground_truth.keys(), key=lambda x: abs(x - t))
            
            # Get ground truth parameters
            gt_params = self.ground_truth[closest_t]
            gt_mean = gt_params["mean"]
            gt_std = gt_params["std"]
            
            # Calculate errors
            mean_error = abs(mean - gt_mean) / gt_mean if gt_mean != 0 else abs(mean)
            std_error = abs(std - gt_std) / gt_std if gt_std != 0 else abs(std)
            
            mean_errors.append(mean_error)
            std_errors.append(std_error)
        
        return {
            "mean_error": np.mean(mean_errors),
            "std_error": np.mean(std_errors),
            "mean_errors": mean_errors,
            "std_errors": std_errors
        }
    
    def _print_summary(self, prober_name):
        """Print summary of evaluation results"""
        result = self.results[prober_name]
        
        print(f"\n{prober_name} Summary:")
        print(f"Probes sent: {result['probes_sent']}")
        print(f"Probes received: {result['probes_received']}")
        print(f"Loss rate: {result['loss_rate']:.2%}")
        print(f"Runtime: {result['runtime']:.2f} seconds")
        print(f"Average CPU usage: {result['avg_cpu']:.2f}%")
        print(f"Average memory usage: {result['avg_memory']:.2f} MB")
        print(f"Maximum memory usage: {result['max_memory']:.2f} MB")
        print(f"Mean error: {result['accuracy']['mean_error']:.4f}")
        print(f"Std error: {result['accuracy']['std_error']:.4f}")
    
    def compare_probers(self, metrics=None):
        """
        Compare all evaluated probers.
        
        Parameters:
        -----------
        metrics: list
            List of metrics to compare (defaults to standard set)
            
        Returns:
        --------
        pd.DataFrame: Comparison table
        """
        if not self.results:
            print("No probers have been evaluated yet.")
            return None
        
        if metrics is None:
            metrics = [
                "probes_sent", "probes_received", "loss_rate", 
                "runtime", "avg_cpu", "avg_memory", "max_memory",
                "accuracy.mean_error", "accuracy.std_error"
            ]
        
        # Prepare data for comparison
        comparison_data = {}
        
        for name, result in self.results.items():
            comparison_data[name] = {}
            
            for metric in metrics:
                if "." in metric:
                    # Handle nested metrics
                    parts = metric.split(".")
                    value = result
                    for part in parts:
                        value = value[part]
                else:
                    # Handle top-level metrics
                    value = result[metric]
                
                comparison_data[name][metric] = value
        
        # Convert to DataFrame
        df = pd.DataFrame(comparison_data).T
        
        # Print table
        print("\nProber Comparison:")
        print(tabulate(df, headers="keys", tablefmt="pipe", floatfmt=".4f"))
        
        return df