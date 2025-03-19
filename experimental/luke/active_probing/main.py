import logging
import time
import matplotlib.pyplot as plt
import os
import copy
from simulators.active_simulator_v2 import ActiveSimulator_v2
from probing_systems.memory_efficient_prober import MemoryEfficientProber
from probing_systems.high_rate_prober import HighRateProber
from probing_systems.adaptive_prober import AdaptiveProber
from evaluation.evaluator import ProberEvaluator
from evaluation.visualizer import ProberVisualizer

class RateLimitFilter(logging.Filter):
    def __init__(self, name='', limit=10):
        super().__init__(name)
        self.limit = limit
        self.counter = 0

    def filter(self, record):
        if "Probe rate limit exceeded" in record.getMessage():
            self.counter += 1
            if self.counter > self.limit:
                return False
        return True

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    rate_limit_filter = RateLimitFilter(limit=10)  # Adjust the limit as needed
    logger.addFilter(rate_limit_filter)

    # Set a higher logging level for specific modules if needed
    logging.getLogger('simulators.active_simulator_v2').setLevel(logging.WARNING)
    logging.getLogger('probing_systems.high_rate_prober').setLevel(logging.WARNING)
    
    # Enable debug logging for the AdaptiveProber to diagnose issues
    logging.getLogger('probing_systems.adaptive_prober').setLevel(logging.DEBUG)

def main():
    setup_logging()
    logging.info("Initializing simulation environment...")
    
    # Set the simulation duration explicitly
    sim_duration = 60.0  # seconds
    
    # Create the simulator with the correct duration
    simulator_config = {
        "paths": "1",
        "congestion_duration": 10.0,
        "congestion_intensity": 0.7,
        "num_congestion_windows": 2,
        "max_simulation_time": sim_duration  # This is key
    }
    
    # Create a list to hold all the probing systems we want to evaluate
    all_probers = []
    
    # Create simulators with explicit max_simulation_time
    simulator1 = ActiveSimulator_v2(**simulator_config)
    high_rate_prober = HighRateProber(simulator1, probes_per_second=10)
    all_probers.append((high_rate_prober, simulator1))
    
    simulator2 = ActiveSimulator_v2(**simulator_config)
    adaptive_prober = AdaptiveProber(simulator2, base_probes_per_second=1, max_probes_per_second=10, max_allowed_rate=10)
    all_probers.append((adaptive_prober, simulator2))
    
    simulator3 = ActiveSimulator_v2(**simulator_config)
    memory_efficient_prober = MemoryEfficientProber(simulator3, probes_per_second=5)
    all_probers.append((memory_efficient_prober, simulator3))
    
    # Create the main evaluator using the first simulator
    main_evaluator = ProberEvaluator(simulator1)
    
    # Evaluate each prober with its own simulator
    logging.info("Evaluating probers...")
    for prober, simulator in all_probers:
        # Create a separate evaluator for each prober's simulator
        prober_evaluator = ProberEvaluator(simulator)
        result = prober_evaluator.evaluate_prober(prober, duration=sim_duration, reset_simulator=True)
        
        # Copy the results to the main evaluator
        main_evaluator.results[prober.name] = result
    
    # Compare probers
    main_evaluator.compare_probers()
    
    # Create visualizer with the correct simulation duration
    visualizer = ProberVisualizer(main_evaluator, sim_duration=sim_duration)
    
    # Directory to save plots
    plot_dir = "/Users/lukemciver/Network-Simulation-Group-Research-Project/experimental/luke/active_probing"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Generate plots
    logging.info("Generating plots...")
    
    # Delay estimates
    for prober_name in main_evaluator.results.keys():
        fig = visualizer.plot_delay_estimates(prober_name)
        if fig:
            fig_path = os.path.join(plot_dir, f"{prober_name}_delay_estimates.png")
            fig.savefig(fig_path)
            logging.info(f"Saved {fig_path}")
    
    # Error over time
    for prober_name in main_evaluator.results.keys():
        fig = visualizer.plot_error_over_time(prober_name)
        if fig:
            fig_path = os.path.join(plot_dir, f"{prober_name}_error_over_time.png")
            fig.savefig(fig_path)
            logging.info(f"Saved {fig_path}")
    
    # Resource usage
    for prober_name in main_evaluator.results.keys():
        fig = visualizer.plot_resource_usage(prober_name)
        if fig:
            fig_path = os.path.join(plot_dir, f"{prober_name}_resource_usage.png")
            fig.savefig(fig_path)
            logging.info(f"Saved {fig_path}")
    
    # Comparison plots
    fig = visualizer.plot_comparison(metric="accuracy.mean_error")
    if fig:
        fig_path = os.path.join(plot_dir, "comparison_mean_error.png")
        fig.savefig(fig_path)
        logging.info(f"Saved {fig_path}")
    
    fig = visualizer.plot_comparison(metric="probes_sent")
    if fig:
        fig_path = os.path.join(plot_dir, "comparison_probes_sent.png")
        fig.savefig(fig_path)
        logging.info(f"Saved {fig_path}")
    
    fig = visualizer.plot_tradeoff(x_metric="probes_sent", y_metric="accuracy.mean_error")
    if fig:
        fig_path = os.path.join(plot_dir, "tradeoff_probes_vs_error.png")
        fig.savefig(fig_path)
        logging.info(f"Saved {fig_path}")
    
    fig = visualizer.plot_tradeoff(x_metric="avg_memory", y_metric="accuracy.mean_error")
    if fig:
        fig_path = os.path.join(plot_dir, "tradeoff_memory_vs_error.png")
        fig.savefig(fig_path)
        logging.info(f"Saved {fig_path}")
    
    logging.info("Evaluation complete!")

if __name__ == "__main__":
    main()