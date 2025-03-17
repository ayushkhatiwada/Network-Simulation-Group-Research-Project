from typing import Dict, Callable
import time
import numpy as np
from .strategies import basic_probe_strategy, adaptive_probe_strategy
from experimental.luke.active_probing.congestion_network import CongestionNetwork

class StrategyEvaluator:
    def __init__(self):
        self.network = CongestionNetwork()
        self.results_template = {
            'kl_divergence': [],
            'probes_used': [],
            'time': [],
            'param_errors': {},
            'metadata': []
        }

    def evaluate(self, strategy: Callable, param_targets: list, 
                congestion_levels: list = [0.0, 0.3, 0.6, 0.9], trials: int = 5) -> Dict:
        results = {param: [] for param in param_targets}
        results.update(self.results_template.copy())
        
        for level in congestion_levels:
            self.network.update_congestion(level)
            true_params = self.network.get_distribution_parameters()
            print(f"\nEvaluating at congestion level {level}")
            
            for trial in range(trials):
                start_time = time.time()
                estimated_params, meta = strategy(self.network.send_probe)
                elapsed = time.time() - start_time
                
                if not estimated_params or all(v is None for v in estimated_params.values()):
                    print(f"Warning: Strategy returned no valid parameters at congestion level {level}, trial {trial}")
                    continue
                
                # Ensure we have enough samples before evaluating-- high variance will break KL divergence
                min_samples = 50  
                if meta.get('probes_used', 0) < min_samples:
                    print(f"Warning: Not enough samples ({meta.get('probes_used', 0)}) for reliable evaluation")
                    continue
                
                # Print comparison of true vs estimated parameters
                print(f"\nTrial {trial} - Samples: {meta.get('probes_used', 0)}")
                print(f"  True params: μ={true_params.get('mu', 0):.4f}, σ={true_params.get('sigma', 0):.4f}")
                
                if 'normal_mu' in estimated_params and 'congested_mu' in estimated_params:
                    print(f"  Est normal: μ={estimated_params['normal_mu']:.4f}, σ={estimated_params['normal_sigma']:.4f}")
                    print(f"  Est congested: μ={estimated_params['congested_mu']:.4f}, σ={estimated_params['congested_sigma']:.4f}")
                else:
                    print(f"  Est params: μ={estimated_params.get('mu', 0):.4f}, σ={estimated_params.get('sigma', 0):.4f}")
                
                try:
                    if 'normal_mu' in estimated_params and 'congested_mu' in estimated_params:
                        normal_kl = self._calculate_kl_divergence(
                            {'mu': estimated_params['normal_mu'], 'sigma': estimated_params['normal_sigma']},
                            true_params
                        )
                        
                        congested_kl = self._calculate_kl_divergence(
                            {'mu': estimated_params['congested_mu'], 'sigma': estimated_params['congested_sigma']},
                            true_params
                        )
                        
                        kl = min(normal_kl, congested_kl)
                        print(f"  KL divergence: {kl:.4f} (min of normal: {normal_kl:.4f}, congested: {congested_kl:.4f})")
                    else:
                        kl = self._calculate_kl_divergence(
                            {'mu': estimated_params.get('mu'), 'sigma': estimated_params.get('sigma')},
                            true_params
                        )
                        print(f"  KL divergence: {kl:.4f}")
                    
                    kl = min(kl, 3.0)
                    
                    results['kl_divergence'].append(kl)
                    results['probes_used'].append(meta.get('probes_used', 0))
                    results['time'].append(elapsed)
                    
                    results['metadata'].append({
                        'congestion_level': level,
                        'trial': trial,
                        'probes_used': meta.get('probes_used', 0)
                    })
                    
                    # Calculate parameter errors - use the better matching parameters
                    if 'normal_mu' in estimated_params and 'congested_mu' in estimated_params:
                        normal_errors = {
                            param: abs(true_params.get(param, 0) - estimated_params.get(f'normal_{param}', 0))
                            for param in param_targets
                        }
                        
                        congested_errors = {
                            param: abs(true_params.get(param, 0) - estimated_params.get(f'congested_{param}', 0))
                            for param in param_targets
                        }
                        
                        # Use errors from the better matching distribution
                        if normal_kl <= congested_kl:
                            for param in param_targets:
                                results[param].append(normal_errors[param])
                        else:
                            for param in param_targets:
                                results[param].append(congested_errors[param])
                    else:
                        # Use standard parameter errors
                        for param in param_targets:
                            if param in true_params and param in estimated_params and estimated_params[param] is not None:
                                error = abs(true_params[param] - estimated_params[param])
                                results[param].append(error)
                except Exception as e:
                    print(f"Error calculating metrics at congestion level {level}: {str(e)}")
        
        return results

    def _calculate_kl_divergence(self, estimated_params, true_params):
        """Calculate KL divergence between estimated and true parameters"""
        try:
            est_mu = estimated_params.get('mu')
            est_sigma = estimated_params.get('sigma')
            true_mu = true_params.get('mu', 0.1)
            true_sigma = true_params.get('sigma', 0.01)
            
            # Ensure parameters are valid
            if est_mu is None or est_sigma is None:
                return 3.0  # Return max KL if parameters are invalid
            
            est_sigma = max(0.001, est_sigma)  # Prevent division by zero
            true_sigma = max(0.001, true_sigma)  # Prevent division by zero
            
            # Calculate KL divergence terms separately for better numerical stability
            log_term = np.log(est_sigma/true_sigma)
            variance_term = (true_sigma**2) / (est_sigma**2)
            mean_term = ((true_mu - est_mu)**2) / (est_sigma**2)
            
            # Calculate KL with bounds to prevent numerical issues
            kl = 0.5 * (log_term + variance_term + mean_term - 1.0)
            return max(0, kl)
        except Exception as e:
            print(f"Error in KL calculation: {str(e)}")
            return 3.0  # Return max KL on error

    def generate_report(self, results: Dict) -> str:
        report = [
            "Strategy Evaluation Report",
            "=========================",
            f"Trials Completed: {len(results['kl_divergence'])}",
            "\nAccuracy Metrics:",
            f"  Average KL Divergence: {np.nanmean(results['kl_divergence']):.4f}"
        ]
        
        if 'mu' in results:
            report.append(f"  Parameter Error Rates:\n    mu: {np.nanmean(results['mu']):.4f}")
        if 'sigma' in results:
            report.append(f"    sigma: {np.nanmean(results['sigma']):.4f}")
            
        report.extend([
            "\nResource Usage:",
            f"  Avg Time (s): {np.nanmean(results['time']):.2f}",
            f"  Avg Probes Used: {int(np.nanmean(results['probes_used']))}"
        ])
        
        return "\n".join(report)

    def compare_distribution_parameters(self, pred_params):
        kl_total = 0
        for edge in self.network.path[:-1]:
            true_params = self.network.get_edge_parameters(edge[0], edge[1])
            # Calculate KL for each edge
            kl_total += self._calculate_edge_kl(pred_params[edge], true_params)
        return kl_total / len(self.network.path)  # Average KL across path