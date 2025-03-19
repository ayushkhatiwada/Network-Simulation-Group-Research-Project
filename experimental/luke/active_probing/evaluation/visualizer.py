import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import logging

class ProberVisualizer:
    """
    Visualizes probing results and comparisons.
    """
    def __init__(self, evaluator, sim_duration=60.0):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        evaluator: ProberEvaluator
            The evaluator containing results
        sim_duration: float
            The actual simulation duration (for proper axis limits)
        """
        self.evaluator = evaluator
        self.results = evaluator.results
        self.ground_truth = evaluator.ground_truth
        self.sim_duration = sim_duration
        
        # Set up plotting style
        plt.style.use('ggplot')
        sns.set_palette("colorblind")
    
    def plot_delay_estimates(self, prober_name):
        """
        Plot delay estimates against ground truth.
        """
        if prober_name not in self.results:
            print(f"No results found for {prober_name}")
            return None
        
        result = self.results[prober_name]
        
        # Check if distribution_estimates are available
        if "distribution_estimates" not in result or not result["distribution_estimates"]:
            print(f"No distribution estimates found for {prober_name}")
            return None
        
        # Get distribution estimates
        estimates = [(t, m, s) for t, m, s in result["distribution_estimates"] if t <= self.sim_duration]
        
        if not estimates:
            print(f"No estimates within simulation duration for {prober_name}")
            return None
        
        # Create figure
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot ground truth delay over time (only within simulation duration)
        gt_times = sorted([t for t in self.ground_truth.keys() if t <= self.sim_duration])
        gt_delays = []
        gt_means = []
        gt_stds = []
        
        for t in gt_times:
            params = self.ground_truth[t]
            gt_means.append(params["mean"])
            gt_stds.append(params["std"])
            # Generate a sample delay with jitter for visualization
            gt_delays.append(params["mean"] + np.random.normal(0, params["std"]/3))
        
        # Plot ground truth mean delay
        ax1.plot(gt_times, gt_means, 'k--', linewidth=1.5, label="Ground Truth Mean")
        
        # Plot ground truth delay with jitter
        ax1.plot(gt_times, gt_delays, 'gray', alpha=0.6, linewidth=0.5, label="Ground Truth Delay")
        
        # Plot ground truth standard deviation as shaded area
        upper = [m + s for m, s in zip(gt_means, gt_stds)]
        lower = [m - s for m, s in zip(gt_means, gt_stds)]
        ax1.fill_between(gt_times, lower, upper, color='gray', alpha=0.3, label="Ground Truth ±σ")
        
        # Extract prober's estimates
        timestamps = [e[0] for e in estimates]
        means = [e[1] for e in estimates]
        stds = [e[2] for e in estimates]
        
        # Plot prober's estimates with error bars
        ax1.errorbar(timestamps, means, yerr=stds, fmt='bo-', linewidth=1.5, 
                     elinewidth=1, capsize=3, label=f"{prober_name} Estimate")
        
        # Plot congestion windows if available
        if hasattr(self.evaluator.simulator, 'get_congestion_windows'):
            congestion_windows = self.evaluator.simulator.get_congestion_windows()
            
            if congestion_windows:
                ax2 = ax1.twinx()
                
                # Plot congestion intensity - only within simulation duration
                for start, end, intensity in congestion_windows:
                    # Skip windows entirely outside our simulation duration
                    if start >= self.sim_duration:
                        continue
                        
                    # Clip end time to simulation duration
                    end = min(end, self.sim_duration)
                    
                    # Draw congestion window
                    color_intensity = min(0.8, intensity + 0.2)  # Ensure visibility
                    ax1.axvspan(start, end, alpha=color_intensity, color='red')
                    ax2.plot([start, end], [intensity, intensity], 'r-', alpha=0.7)
                
                ax2.set_ylabel('Congestion Intensity')
                ax2.set_ylim(0, 1.1)
                ax2.grid(False)
        
        # Set labels and title
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Delay (ms)')
        ax1.set_title(f'Delay Estimates: {prober_name}')
        
        # Set x-axis limits to match simulation duration
        ax1.set_xlim(0, self.sim_duration)
        
        ax1.grid(True)
        ax1.legend(loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def plot_error_over_time(self, prober_name):
        """
        Plot estimation error over time.
        """
        if prober_name not in self.results:
            print(f"No results found for {prober_name}")
            return None
        
        result = self.results[prober_name]
        
        # Check if distribution_estimates are available (use these instead of raw results)
        if "distribution_estimates" not in result or not result["distribution_estimates"]:
            print(f"No distribution estimates found for {prober_name}")
            return None
        
        # Create figure
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Extract distribution estimates within simulation duration
        estimates = [(t, m, s) for t, m, s in result["distribution_estimates"] if t <= self.sim_duration]
        
        if not estimates:
            print(f"No estimates within simulation duration for {prober_name}")
            return None
        
        timestamps = [e[0] for e in estimates]
        means = [e[1] for e in estimates]
        stds = [e[2] for e in estimates]
        
        # Calculate errors
        mean_errors = []
        std_errors = []
        
        for i, t in enumerate(timestamps):
            # Find closest ground truth time point
            closest_t = min(self.ground_truth.keys(), key=lambda x: abs(x - t))
            
            # Get ground truth parameters
            gt_params = self.ground_truth[closest_t]
            gt_mean = gt_params["mean"]
            gt_std = gt_params["std"]
            
            # Calculate relative errors
            mean_error = abs(means[i] - gt_mean) / gt_mean if gt_mean != 0 else abs(means[i])
            std_error = abs(stds[i] - gt_std) / gt_std if gt_std != 0 else abs(stds[i])
            
            mean_errors.append(mean_error)
            std_errors.append(std_error)
        
        # Plot errors
        ax1.plot(timestamps, mean_errors, 'b-', linewidth=2, label='Mean Error')
        ax1.plot(timestamps, std_errors, 'g-', linewidth=2, label='Std Error')
        
        # Plot congestion windows if available
        if hasattr(self.evaluator.simulator, 'get_congestion_windows'):
            congestion_windows = self.evaluator.simulator.get_congestion_windows()
            
            if congestion_windows:
                ax2 = ax1.twinx()
                
                # Plot congestion intensity - only within simulation duration
                for start, end, intensity in congestion_windows:
                    # Skip windows entirely outside our simulation duration
                    if start >= self.sim_duration:
                        continue
                        
                    # Clip end time to simulation duration
                    end = min(end, self.sim_duration)
                    
                    # Draw congestion window with intensity-based color
                    color_intensity = min(0.8, intensity + 0.2)  # Ensure visibility
                    ax1.axvspan(start, end, alpha=color_intensity, color='red')
                    ax2.plot([start, end], [intensity, intensity], 'r-', alpha=0.7)
                
                ax2.set_ylabel('Congestion Intensity')
                ax2.set_ylim(0, 1.1)
                ax2.grid(False)
        
        # Set labels and title
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Relative Error')
        ax1.set_title(f'Estimation Error Over Time: {prober_name}')
        
        # Set x-axis limits to match simulation duration
        ax1.set_xlim(0, self.sim_duration)
        
        ax1.grid(True)
        ax1.legend(loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def plot_resource_usage(self, prober_name):
        """
        Plot resource usage over time.
        
        Parameters:
        -----------
        prober_name: str
            Name of the prober to visualize
            
        Returns:
        --------
        matplotlib.figure.Figure: The figure object
        """
        if prober_name not in self.results:
            print(f"No results found for {prober_name}")
            return None
        
        result = self.results[prober_name]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Time points (assuming measurements taken every 0.5 seconds)
        time_points = np.arange(0, len(result["cpu_usage"]) * 0.5, 0.5)
        
        # Plot CPU usage
        ax1.plot(time_points, result["cpu_usage"], 'b-')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_title(f'Resource Usage: {prober_name}')
        ax1.grid(True)
        
        # Plot memory usage
        ax2.plot(time_points, result["memory_usage"], 'g-')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_comparison(self, metric='accuracy.mean_error'):
        """
        Compare different probing systems on a specific metric.
        
        Parameters:
        -----------
        metric: str
            Metric to compare (dot notation for nested metrics)
            
        Returns:
        --------
        matplotlib.figure.Figure: The figure object
        """
        if not self.results:
            print("No probers have been evaluated yet.")
            return None
        
        # Extract metric values
        names = []
        values = []
        
        for name, result in self.results.items():
            names.append(name)
            
            # Handle nested metrics
            if "." in metric:
                parts = metric.split(".")
                value = result
                for part in parts:
                    value = value[part]
            else:
                value = result[metric]
            
            values.append(value)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot bar chart
        bars = ax.bar(names, values)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.4f}', ha='center', va='bottom')
        
        # Set labels and title
        ax.set_xlabel('Probing Strategy')
        ax.set_ylabel(metric.replace('.', ' '))
        ax.set_title(f'Comparison of {metric.replace(".", " ")}')
        ax.grid(True, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_tradeoff(self, x_metric='probes_sent', y_metric='accuracy.mean_error'):
        """
        Plot accuracy vs. resource usage tradeoff.
        
        Parameters:
        -----------
        x_metric: str
            Resource metric (dot notation)
        y_metric: str
            Accuracy metric (dot notation)
            
        Returns:
        --------
        matplotlib.figure.Figure: The figure object
        """
        if not self.results:
            print("No probers have been evaluated yet.")
            return None
        
        # Extract metric values
        names = []
        x_values = []
        y_values = []
        
        for name, result in self.results.items():
            names.append(name)
            
            # Extract x metric
            if "." in x_metric:
                parts = x_metric.split(".")
                x_value = result
                for part in parts:
                    x_value = x_value[part]
            else:
                x_value = result[x_metric]
            
            # Extract y metric
            if "." in y_metric:
                parts = y_metric.split(".")
                y_value = result
                for part in parts:
                    y_value = y_value[part]
            else:
                y_value = result[y_metric]
            
            names.append(name)
            x_values.append(x_value)
            y_values.append(y_value)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot scatter plot
        ax.scatter(x_values, y_values)
        
        # Add regression line
        z = np.polyfit(x_values, y_values, 1)
        p = np.poly1d(z)
        ax.plot(x_values, p(x_values), "r--")
        
        # Set labels and title
        ax.set_xlabel(x_metric.replace('.', ' '))
        ax.set_ylabel(y_metric.replace('.', ' '))
        ax.set_title(f'{y_metric.replace(".", " ")} vs. {x_metric.replace(".", " ")}')
        
        plt.tight_layout()
        return fig