import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import os

def ensure_plot_dir():
    """Ensure the plots directory exists"""
    os.makedirs('plots', exist_ok=True)

def plot_parameter_errors(results: Dict, strategy_name: str):
    ensure_plot_dir()
    
    # congestion levels and errors
    congestion_levels = sorted(set([meta['congestion_level'] for meta in results.get('metadata', [])]))
    
    if not congestion_levels:
        print("No congestion level data available for plotting")
        return
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Create mapping from metadata index to parameter index
    # This ensures we're using the correct indices for parameters
    metadata_indices = {i: i for i in range(len(results.get('metadata', [])))}
    
    # Plot mu errors
    mu_errors_by_level = {}
    for level in congestion_levels:
        mu_errors_by_level[level] = []
        for i, meta in enumerate(results.get('metadata', [])):
            if meta.get('congestion_level') == level and i < len(results.get('mu', [])):
                mu_errors_by_level[level].append(results['mu'][i])
    
    # Plot sigma errors
    sigma_errors_by_level = {}
    for level in congestion_levels:
        sigma_errors_by_level[level] = []
        for i, meta in enumerate(results.get('metadata', [])):
            if meta.get('congestion_level') == level and i < len(results.get('sigma', [])):
                sigma_errors_by_level[level].append(results['sigma'][i])
    
    # Box plots for mu errors
    axs[0].boxplot([mu_errors_by_level[level] for level in congestion_levels])
    axs[0].set_title(f'μ Error by Congestion Level ({strategy_name})')
    axs[0].set_xlabel('Congestion Level')
    axs[0].set_ylabel('Absolute Error')
    axs[0].set_xticklabels([f"{level:.1f}" for level in congestion_levels])
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # Box plots for sigma errors
    axs[1].boxplot([sigma_errors_by_level[level] for level in congestion_levels])
    axs[1].set_title(f'σ Error by Congestion Level ({strategy_name})')
    axs[1].set_xlabel('Congestion Level')
    axs[1].set_ylabel('Absolute Error')
    axs[1].set_xticklabels([f"{level:.1f}" for level in congestion_levels])
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'plots/{strategy_name}_parameter_errors.png')
    plt.close()

def plot_kl_divergence(results: Dict, strategy_name: str):
    ensure_plot_dir()
    
    # Extract congestion levels and KL values
    congestion_levels = sorted(set([meta['congestion_level'] for meta in results.get('metadata', [])]))
    
    if not congestion_levels:
        print("No congestion level data available for plotting")
        return
    
    # Group KL values by congestion level
    kl_by_level = {}
    for level in congestion_levels:
        kl_by_level[level] = []
        for i, meta in enumerate(results.get('metadata', [])):
            if meta.get('congestion_level') == level and i < len(results.get('kl_divergence', [])):
                kl_by_level[level].append(results['kl_divergence'][i])
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Box plot for KL divergence
    plt.boxplot([kl_by_level[level] for level in congestion_levels])
    plt.title(f'KL Divergence by Congestion Level ({strategy_name})')
    plt.xlabel('Congestion Level')
    plt.ylabel('KL Divergence')
    plt.xticks(range(1, len(congestion_levels) + 1), [f"{level:.1f}" for level in congestion_levels])
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'plots/{strategy_name}_kl_divergence.png')
    plt.close()

def plot_resource_usage(results: Dict, strategy_name: str):
    """
    Plot resource usage (probes and time) across different congestion levels
    
    Args:
        results: Results dictionary from evaluator
        strategy_name: Name of the strategy for the plot title
    """
    ensure_plot_dir()
    
    # Extract congestion levels
    congestion_levels = sorted(set([meta['congestion_level'] for meta in results.get('metadata', [])]))
    
    if not congestion_levels:
        print("No congestion level data available for plotting")
        return
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Group probes by congestion level
    probes_by_level = {}
    for level in congestion_levels:
        probes_by_level[level] = []
        for i, meta in enumerate(results.get('metadata', [])):
            if meta.get('congestion_level') == level and i < len(results.get('probes_used', [])):
                probes_by_level[level].append(results['probes_used'][i])
    
    # Group time by congestion level
    time_by_level = {}
    for level in congestion_levels:
        time_by_level[level] = []
        for i, meta in enumerate(results.get('metadata', [])):
            if meta.get('congestion_level') == level and i < len(results.get('time', [])):
                time_by_level[level].append(results['time'][i])
    
    # Box plots for probes used
    axs[0].boxplot([probes_by_level[level] for level in congestion_levels])
    axs[0].set_title(f'Probes Used by Congestion Level ({strategy_name})')
    axs[0].set_xlabel('Congestion Level')
    axs[0].set_ylabel('Number of Probes')
    axs[0].set_xticklabels([f"{level:.1f}" for level in congestion_levels])
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # Box plots for time
    axs[1].boxplot([time_by_level[level] for level in congestion_levels])
    axs[1].set_title(f'Time Used by Congestion Level ({strategy_name})')
    axs[1].set_xlabel('Congestion Level')
    axs[1].set_ylabel('Time (s)')
    axs[1].set_xticklabels([f"{level:.1f}" for level in congestion_levels])
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'plots/{strategy_name}_resource_usage.png')
    plt.close()

def plot_strategy_comparison(brute_results: Dict, adaptive_results: Dict, dual_results: Dict = None):
    """
    Plot comparison between strategies
    
    Args:
        brute_results: Results dictionary for brute force strategy
        adaptive_results: Results dictionary for adaptive strategy
        dual_results: Results dictionary for dual distribution strategy (optional)
    """
    ensure_plot_dir()
    
    # Determine strategies to compare
    strategies = ['Brute Force', 'Adaptive']
    if dual_results is not None:
        strategies.append('Dual Distribution')
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Average KL divergence
    kl_values = [np.mean(brute_results.get('kl_divergence', [0])), 
                np.mean(adaptive_results.get('kl_divergence', [0]))]
    if dual_results is not None:
        kl_values.append(np.mean(dual_results.get('kl_divergence', [0])))
    
    axs[0, 0].bar(strategies, kl_values)
    axs[0, 0].set_title('Average KL Divergence')
    axs[0, 0].set_ylabel('KL Divergence')
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Average parameter errors
    mu_errors = [np.mean(brute_results.get('mu', [0])), 
                np.mean(adaptive_results.get('mu', [0]))]
    sigma_errors = [np.mean(brute_results.get('sigma', [0])), 
                   np.mean(adaptive_results.get('sigma', [0]))]
    
    if dual_results is not None:
        mu_errors.append(np.mean(dual_results.get('mu', [0])))
        sigma_errors.append(np.mean(dual_results.get('sigma', [0])))
    
    axs[0, 1].bar(strategies, mu_errors)
    axs[0, 1].set_title('Average μ Error')
    axs[0, 1].set_ylabel('Absolute Error')
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    axs[1, 0].bar(strategies, sigma_errors)
    axs[1, 0].set_title('Average σ Error')
    axs[1, 0].set_ylabel('Absolute Error')
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Average probes used
    probes_used = [np.mean(brute_results.get('probes_used', [0])), 
                  np.mean(adaptive_results.get('probes_used', [0]))]
    
    if dual_results is not None:
        probes_used.append(np.mean(dual_results.get('probes_used', [0])))
    
    axs[1, 1].bar(strategies, probes_used)
    axs[1, 1].set_title('Average Probes Used')
    axs[1, 1].set_ylabel('Number of Probes')
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('plots/strategy_comparison.png')
    plt.close()
