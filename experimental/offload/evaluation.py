import time
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable
import os
import warnings

def timing_wrapper(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

def kl_divergence_normal(mu1, sigma1, mu2, sigma2):
    """
    Calculate KL divergence between two normal distributions.
    KL(P||Q) measures how different Q is from P.
    """
    return (np.log(sigma2/sigma1) + 
            (sigma1**2 + (mu1-mu2)**2)/(2*sigma2**2) - 0.5)

def kl_divergence_lognormal(mu1, sigma1, mu2, sigma2):
    """
    Calculate KL divergence between two lognormal distributions with numerical stability.
    """
    # Add small epsilon to prevent division by zero
    sigma1 = max(sigma1, 1e-6)
    sigma2 = max(sigma2, 1e-6)
    
    return (np.log(sigma2/sigma1) + 
            (sigma1**2 + (mu1-mu2)**2)/(2*sigma2**2) - 0.5)

def kl_divergence_gamma(shape1, scale1, shape2, scale2):
    """
    Calculate KL divergence between two gamma distributions.
    """
    part1 = (shape1 - shape2) * np.psi(shape1)
    part2 = -np.log(scale1) + np.log(scale2)
    part3 = shape1 * (scale1 - scale2) / scale2
    part4 = shape2 * np.log(scale2/scale1)
    part5 = np.log(stats.gamma.pdf(shape1)/stats.gamma.pdf(shape2))
    
    return part1 + part2 + part3 + part4 + part5

def evaluate_distribution(true_params, estimated_params, dist_type="lognormal"):
    """
    Evaluate the difference between true and estimated distribution parameters.
    
    Args:
        true_params: Dict with true distribution parameters
        estimated_params: Dict with estimated distribution parameters
        dist_type: Type of distribution
        
    Returns:
        Dict with evaluation metrics
    """
    results = {
        'distribution_type': dist_type,
        'parameter_error': {},
        'kl_divergence': None
    }
    
    if dist_type == "lognormal":
        # Parameter errors
        results['parameter_error']['mu'] = abs(true_params['mu'] - estimated_params['mu'])
        results['parameter_error']['sigma'] = abs(true_params['sigma'] - estimated_params['sigma'])
        
        # KL divergence
        results['kl_divergence'] = kl_divergence_lognormal(
            true_params['mu'], true_params['sigma'],
            estimated_params['mu'], estimated_params['sigma']
        )
        
    elif dist_type == "normal":
        # Parameter errors
        results['parameter_error']['mean'] = abs(true_params['mean'] - estimated_params['mean'])
        results['parameter_error']['std'] = abs(true_params['std'] - estimated_params['std'])
        
        # KL divergence
        results['kl_divergence'] = kl_divergence_normal(
            true_params['mean'], true_params['std'],
            estimated_params['mean'], estimated_params['std']
        )
        
    elif dist_type == "gamma":
        # Parameter errors
        results['parameter_error']['shape'] = abs(true_params['shape'] - estimated_params['shape'])
        results['parameter_error']['scale'] = abs(true_params['scale'] - estimated_params['scale'])
        
        # KL divergence
        results['kl_divergence'] = kl_divergence_gamma(
            true_params['shape'], true_params['scale'],
            estimated_params['shape'], estimated_params['scale']
        )
    
    # Threshold for "good" estimation (KL < 0.05)
    results['is_good_estimate'] = results['kl_divergence'] < 0.05
    
    return results

def plot_distribution_comparison(true_params, estimated_params, samples=None, dist_type="lognormal"):
    """
    Plot the true and estimated distributions for comparison.
    
    Args:
        true_params: Dict with true distribution parameters
        estimated_params: Dict with estimated distribution parameters
        samples: Optional array of actual samples for histogram
        dist_type: Type of distribution
    """
    plt.figure(figsize=(10, 6))
    
    # Generate x values based on both true and estimated parameters
    if dist_type == "lognormal":
        # Get valid range from both distributions
        x_min = min(np.exp(true_params['mu'] - 3*true_params['sigma']),
                   np.exp(estimated_params.get('mu', 0) - 3*estimated_params.get('sigma', 1)))
        x_max = max(np.exp(true_params['mu'] + 3*true_params['sigma']),
                   np.exp(estimated_params.get('mu', 0) + 3*estimated_params.get('sigma', 1)))
        
        # Fallback to sample data if parameters are invalid
        if not np.isfinite(x_min) or not np.isfinite(x_max):
            valid_samples = [s for s in samples if s > 0]
            x_min = np.min(valid_samples) * 0.8 if valid_samples else 0.1
            x_max = np.max(valid_samples) * 1.2 if valid_samples else 10.0
            
        x = np.linspace(x_min, x_max, 1000)
        
        # Handle invalid estimates
        if not np.isfinite(estimated_params.get('mu', 0)) or not np.isfinite(estimated_params.get('sigma', 1)):
            est_pdf = np.zeros_like(x)
        else:
            est_pdf = stats.lognorm.pdf(x, s=estimated_params['sigma'], 
                                      scale=np.exp(estimated_params['mu']))
            
        true_pdf = stats.lognorm.pdf(x, s=true_params['sigma'], 
                                   scale=np.exp(true_params['mu']))
        
        plt.plot(x, true_pdf, 'r-', label=f'True LogNormal(μ={true_params["mu"]:.2f}, σ={true_params["sigma"]:.2f})')
        plt.plot(x, est_pdf, 'b--', label=f'Estimated LogNormal(μ={estimated_params["mu"]:.2f}, σ={estimated_params["sigma"]:.2f})')
        
    elif dist_type == "normal":
        x = np.linspace(true_params['mean'] - 4*true_params['std'], 
                       true_params['mean'] + 4*true_params['std'], 1000)
        true_pdf = stats.norm.pdf(x, true_params['mean'], true_params['std'])
        est_pdf = stats.norm.pdf(x, estimated_params['mean'], estimated_params['std'])
        
        plt.plot(x, true_pdf, 'r-', label=f'True Normal(μ={true_params["mean"]:.2f}, σ={true_params["std"]:.2f})')
        plt.plot(x, est_pdf, 'b--', label=f'Estimated Normal(μ={estimated_params["mean"]:.2f}, σ={estimated_params["std"]:.2f})')
        
    elif dist_type == "gamma":
        x = np.linspace(0, true_params['shape'] * true_params['scale'] * 3, 1000)
        true_pdf = stats.gamma.pdf(x, true_params['shape'], scale=true_params['scale'])
        est_pdf = stats.gamma.pdf(x, estimated_params['shape'], scale=estimated_params['scale'])
        
        plt.plot(x, true_pdf, 'r-', 
                label=f'True Gamma(k={true_params["shape"]:.2f}, θ={true_params["scale"]:.2f})')
        plt.plot(x, est_pdf, 'b--', 
                label=f'Estimated Gamma(k={estimated_params["shape"]:.2f}, θ={estimated_params["scale"]:.2f})')
    
    # Plot histogram of samples if provided
    if samples is not None and len(samples) > 0:
        plt.hist(samples, bins=30, density=True, alpha=0.3, color='green', label='Samples')
    
    # Add KL divergence to the plot
    kl = evaluate_distribution(true_params, estimated_params, dist_type)['kl_divergence']
    plt.text(0.05, 0.95, f'KL Divergence: {kl:.4f}', transform=plt.gca().transAxes, 
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title(f'Comparison of True vs. Estimated {dist_type.capitalize()} Distribution')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join('results', f'{dist_type}_distribution_comparison.png'), dpi=300)
    
    return plt.gcf()  # Return the figure for saving or further customization

def congestion_aware_probing_experiment(prober, estimator, source, target, 
                                      congestion_levels, probes_per_level=500, 
                                      output_dir="results"):
    results = {}
    
    for level in congestion_levels:
        prober.set_congestion_level(level)
        samples = []
        stats = {
            'total_probes': 0, 'dropped_probes': 0,
            'path_lengths': []
        }
        
        metadata = {
            'congestion_factor': 1 + 2 * level,
            'path_length': None
        }
        
        while stats['total_probes'] < probes_per_level:
            batch_size = min(20, probes_per_level - stats['total_probes'])
            batch_samples, batch_stats = prober.probe_path(source, target, batch_size)
            
            # Handle completely failed batches
            if len(batch_samples) == 0:
                print(f"Warning: All probes dropped at level {level}")
                stats['total_probes'] += batch_size
                stats['dropped_probes'] += batch_size
                continue
                
            # Safe mean calculation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                metadata['path_length'] = np.nanmean(batch_stats.get('path_lengths', []))
            
            samples.extend(batch_samples)
            stats['total_probes'] += batch_stats['total_probes']
            stats['dropped_probes'] += batch_stats['dropped_probes']
            stats['path_lengths'].append(metadata['path_length'])
        
        results[level] = {
            'samples': samples,
            'stats': stats,
            'estimated_params': estimator.estimate_parameters(
                samples, 
                {
                    **metadata,
                    'distribution_type': prober.network.distribution_type
                }
            ),
            'true_params': prober.network.get_distribution_parameters(source, target)
        }
        
    return results

def plot_delay_distributions(delays_by_congestion, true_params, estimated_params, 
                            distribution_type="lognormal", output_dir="results"):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(delays_by_congestion)))
    
    for i, (level, delays) in enumerate(sorted(delays_by_congestion.items())):
        valid_delays = [d for d in delays if np.isfinite(d)]
        if len(valid_delays) < 10:
            continue
            
        plt.hist(valid_delays, bins=30, alpha=0.3, color=colors[i],
                density=True, label=f'Congestion {level:.1f}')
        
        # Filter NaNs in parameters
        if distribution_type == "lognormal":
            est_mu = estimated_params[level].get('mu', true_params[level]['mu'])
            est_sigma = estimated_params[level].get('sigma', true_params[level]['sigma'])
            
            if not np.isfinite(est_mu) or not np.isfinite(est_sigma):
                continue
                
            x = np.linspace(min(valid_delays)*0.8, max(valid_delays)*1.2, 1000)
            true_pdf = stats.lognorm.pdf(x, s=true_params[level]['sigma'], 
                                       scale=np.exp(true_params[level]['mu']))
            est_pdf = stats.lognorm.pdf(x, s=est_sigma, 
                                      scale=np.exp(est_mu))
            
            if est_pdf is not None:
                plt.plot(x, est_pdf, '-', color=colors[i], linewidth=2)
        elif distribution_type == "gamma":
            true_pdf = stats.gamma.pdf(x, true_params[level]['shape'], 
                                     scale=true_params[level]['scale'])
            est_pdf = stats.gamma.pdf(x, estimated_params[level]['shape'], 
                                    scale=estimated_params[level]['scale'])
        
        plt.plot(x, true_pdf, '--', color=colors[i], linewidth=2)
    
    plt.title('Delay Distributions by Congestion Level')
    plt.xlabel('Delay (ms)')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'congestion_distributions.png'), dpi=300)
    plt.close()

def plot_metrics_vs_congestion(results, output_dir="results"):
    """
    Plot various metrics against congestion level.
    
    Args:
        results: Dict with experiment results
        output_dir: Directory to save plots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Extract data
    levels = sorted(results.keys())
    kl_divergences = [results[level]['evaluation']['kl_divergence'] for level in levels]
    drop_rates = [results[level].get('congestion_stats', {}).get('overall_drop_rate', 0) * 100 for level in levels]
    num_probes = [results[level]['total_probes_sent'] for level in levels]
    successful_samples = [results[level]['num_samples'] for level in levels]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot KL divergence
    ax1 = axes[0, 0]
    ax1.plot(levels, kl_divergences, 'o-', linewidth=2, color='blue')
    ax1.set_xlabel('Congestion Level')
    ax1.set_ylabel('KL Divergence')
    ax1.set_title('KL Divergence vs. Congestion Level')
    ax1.grid(True, alpha=0.3)
    
    # Plot drop rate
    ax2 = axes[0, 1]
    ax2.plot(levels, drop_rates, 'o-', linewidth=2, color='red')
    ax2.set_xlabel('Congestion Level')
    ax2.set_ylabel('Packet Drop Rate (%)')
    ax2.set_title('Packet Drop Rate vs. Congestion Level')
    ax2.grid(True, alpha=0.3)
    
    # Plot number of probes
    ax3 = axes[1, 0]
    ax3.plot(levels, num_probes, 'o-', linewidth=2, color='green')
    ax3.set_xlabel('Congestion Level')
    ax3.set_ylabel('Total Probes Sent')
    ax3.set_title('Probes Required vs. Congestion Level')
    ax3.grid(True, alpha=0.3)
    
    # Plot successful samples
    ax4 = axes[1, 1]
    ax4.plot(levels, successful_samples, 'o-', linewidth=2, color='purple')
    ax4.set_xlabel('Congestion Level')
    ax4.set_ylabel('Successful Samples')
    ax4.set_title('Successful Samples vs. Congestion Level')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_vs_congestion.png'), dpi=300)
    plt.close()
    
    # Plot parameter estimation accuracy
    plt.figure(figsize=(12, 6))
    
    # Extract data based on distribution type
    first_result = next(iter(results.values()))
    if 'mu' in first_result['true_params']:
        # Lognormal distribution
        true_mu = [results[level]['true_params']['mu'] for level in levels]
        est_mu = [results[level]['estimated_params']['mu'] for level in levels]
        true_sigma = [results[level]['true_params']['sigma'] for level in levels]
        est_sigma = [results[level]['estimated_params']['sigma'] for level in levels]
        
        plt.subplot(1, 2, 1)
        plt.plot(levels, true_mu, 'o-', color='blue', label='True μ')
        plt.plot(levels, est_mu, 's--', color='red', label='Estimated μ')
        plt.xlabel('Congestion Level')
        plt.ylabel('μ Parameter')
        plt.title('μ Parameter Estimation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(levels, true_sigma, 'o-', color='blue', label='True σ')
        plt.plot(levels, est_sigma, 's--', color='red', label='Estimated σ')
        plt.xlabel('Congestion Level')
        plt.ylabel('σ Parameter')
        plt.title('σ Parameter Estimation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    elif 'mean' in first_result['true_params']:
        # Normal distribution
        true_mean = [results[level]['true_params']['mean'] for level in levels]
        est_mean = [results[level]['estimated_params']['mean'] for level in levels]
        true_std = [results[level]['true_params']['std'] for level in levels]
        est_std = [results[level]['estimated_params']['std'] for level in levels]
        
        plt.subplot(1, 2, 1)
        plt.plot(levels, true_mean, 'o-', color='blue', label='True Mean')
        plt.plot(levels, est_mean, 's--', color='red', label='Estimated Mean')
        plt.xlabel('Congestion Level')
        plt.ylabel('Mean Parameter')
        plt.title('Mean Parameter Estimation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(levels, true_std, 'o-', color='blue', label='True Std')
        plt.plot(levels, est_std, 's--', color='red', label='Estimated Std')
        plt.xlabel('Congestion Level')
        plt.ylabel('Std Parameter')
        plt.title('Std Parameter Estimation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    elif 'shape' in first_result['true_params']:
        # Gamma distribution
        true_shape = [results[level]['true_params']['shape'] for level in levels]
        est_shape = [results[level]['estimated_params']['shape'] for level in levels]
        true_scale = [results[level]['true_params']['scale'] for level in levels]
        est_scale = [results[level]['estimated_params']['scale'] for level in levels]
        
        plt.subplot(1, 2, 1)
        plt.plot(levels, true_shape, 'o-', color='blue', label='True Shape')
        plt.plot(levels, est_shape, 's--', color='red', label='Estimated Shape')
        plt.xlabel('Congestion Level')
        plt.ylabel('Shape Parameter')
        plt.title('Shape Parameter Estimation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(levels, true_scale, 'o-', color='blue', label='True Scale')
        plt.plot(levels, est_scale, 's--', color='red', label='Estimated Scale')
        plt.xlabel('Congestion Level')
        plt.ylabel('Scale Parameter')
        plt.title('Scale Parameter Estimation')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_estimation.png'), dpi=300)
    plt.close()

def calculate_kl_divergence(samples, estimated_params, true_params, distribution_type="lognormal"):
    """
    Calculate KL divergence between estimated and true distributions.
    
    Args:
        samples: List of delay measurements
        estimated_params: Dict with estimated distribution parameters
        true_params: Dict with true distribution parameters
        distribution_type: Type of distribution
        
    Returns:
        KL divergence value (float)
    """
    # Check parameter validity first
    valid_params = True
    if distribution_type == "lognormal":
        valid_params = all(key in true_params for key in ('mu', 'sigma')) and all(key in estimated_params for key in ('mu', 'sigma'))
    elif distribution_type == "normal":
        valid_params = all(key in true_params for key in ('mean', 'std')) and all(key in estimated_params for key in ('mean', 'std'))
    elif distribution_type == "gamma":
        valid_params = all(key in true_params for key in ('shape', 'scale')) and all(key in estimated_params for key in ('shape', 'scale'))
    
    if not valid_params:
        return float('nan')
    
    try:
        if distribution_type == "lognormal":
            if 'mu' in true_params and 'sigma' in true_params and 'mu' in estimated_params and 'sigma' in estimated_params:
                if estimated_params['mu'] < 0 or true_params['mu'] < 0:
                    return float('nan')
                return kl_divergence_lognormal(
                    true_params['mu'], true_params['sigma'],
                    estimated_params['mu'], estimated_params['sigma']
                )
        elif distribution_type == "normal":
            if 'mean' in true_params and 'std' in true_params and 'mean' in estimated_params and 'std' in estimated_params:
                return kl_divergence_normal(
                    true_params['mean'], true_params['std'],
                    estimated_params['mean'], estimated_params['std']
                )
        elif distribution_type == "gamma":
            if 'shape' in true_params and 'scale' in true_params and 'shape' in estimated_params and 'scale' in estimated_params:
                return kl_divergence_gamma(
                    true_params['shape'], true_params['scale'],
                    estimated_params['shape'], estimated_params['scale']
                )
                
        # If we get here, something went wrong - use the evaluate_distribution function as a fallback
        result = evaluate_distribution(true_params, estimated_params, distribution_type)
        return result.get('kl_divergence', float('nan'))
        
    except Exception as e:
        print(f"Error calculating KL divergence: {str(e)}")
        return float('nan')