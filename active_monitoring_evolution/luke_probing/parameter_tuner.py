import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import json
import copy
from typing import Dict, List, Tuple, Any

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from active_simulator_v3 import ActiveSimulator_v3
from active_simulator_v2 import ActiveSimulator_v2
from active_simulator_v1 import ActiveSimulator_v1

from base_prober import BaseProber
from adaptive_prober import AdaptiveProber
from probing_evaluator import ProbingEvaluator  

        
# command line args
# python parameter_tuner.py --simulator=2 --seeds=10
# python parameter_tuner.py --simulator=2 --load

class ParameterTuner:
    def __init__(self, max_departure_time=100):
        self.max_departure_time = max_departure_time
        self.evaluator = ProbingEvaluator(max_departure_time)
        
    def calculate_kl_divergence(self, pred_mean, pred_std, true_mean, true_std):
        return self.evaluator.calculate_kl_divergence(pred_mean, pred_std, true_mean, true_std)
    
    def auto_tune_parameters(self, probes_per_second=10, simulator_version=2, num_seeds=5, verbose=True):
        # params
        # probes, simversion,
        # num_seeds (sample size for how much we test)
        # verbose (print progress)
        # returns a dict with best params and all results
    
        congestion_z_values = [1.5, 2.0, 2.5, 3.0, 3.5]
        outlier_z_values = [2.0, 2.5, 3.0, 3.5, 4.0]
        
        seeds = list(range(42, 42 + num_seeds))
        best_params = None
        best_score = float('inf') 
        
        results = {}
        
        for congestion_z in congestion_z_values:
            for outlier_z in outlier_z_values:
                if verbose:
                    print(f"\nTesting parameters: congestion_z={congestion_z}, outlier_z={outlier_z}")
                
                param_results = []
                
                for seed in seeds:
                    if verbose:
                        print(f"  Running with seed {seed}...")
                    
                    if simulator_version == 2:
                        base_simulator = ActiveSimulator_v2(self.max_departure_time, random_seed=seed)
                    else:
                        base_simulator = ActiveSimulator_v1(self.max_departure_time, random_seed=seed)
                    
                    adaptive_simulator = copy.deepcopy(base_simulator)
                    
                    # simulators
                    base_simulator.max_probes_per_second = probes_per_second
                    base_simulator.probe_count_per_second = {}
                    
                    adaptive_simulator.max_probes_per_second = probes_per_second
                    adaptive_simulator.probe_count_per_second = {}
                    
                    # probers
                    base_prober = BaseProber(base_simulator, probes_per_second)
                    adaptive_prober = AdaptiveProber(
                        adaptive_simulator, 
                        max_probes_per_second=probes_per_second,
                        congestion_z=congestion_z,
                        outlier_z=outlier_z,
                        debug=False 
                    )
                    
                    base_prober.probe()
                    adaptive_prober.probe()
                    
                    # Get ground truth and metrics
                    ground_truth_params = base_simulator.network.get_single_edge_distribution_parameters(
                        base_simulator.network.SOURCE, 
                        base_simulator.network.DESTINATION
                    )
                    
                    base_metrics = base_prober.get_metrics()
                    adaptive_metrics = adaptive_prober.get_metrics()
                    
                    # Use final estimates
                    final_adaptive_mean = adaptive_metrics['metrics_per_timeslot'][-1][1]
                    final_adaptive_std = adaptive_metrics['metrics_per_timeslot'][-1][2]
                    
                    # Calculate KL divergence
                    adaptive_kl = self.calculate_kl_divergence(
                        final_adaptive_mean, 
                        final_adaptive_std, 
                        ground_truth_params['mean'], 
                        ground_truth_params['std']
                    )
                    
                    param_results.append({
                        'seed': seed,
                        'kl_divergence': adaptive_kl,
                        'final_mean': final_adaptive_mean,
                        'final_std': final_adaptive_std,
                        'true_mean': ground_truth_params['mean'],
                        'true_std': ground_truth_params['std']
                    })
                
                # Average KL divergence across all seeds
                avg_kl = np.mean([r['kl_divergence'] for r in param_results])
                
                if verbose:
                    print(f"  Average KL divergence: {avg_kl:.6f}")
                
                # Store results for this parameter combination
                param_key = (congestion_z, outlier_z)
                results[param_key] = {
                    'avg_kl_divergence': avg_kl,
                    'individual_results': param_results
                }
                
                # Update best parameters if this is better
                if avg_kl < best_score:
                    best_score = avg_kl
                    best_params = {
                        'congestion_z': congestion_z,
                        'outlier_z': outlier_z,
                        'avg_kl_divergence': avg_kl
                    }
        
        if verbose:
            print("\nParameter tuning complete!")
            print(f"Best parameters: congestion_z={best_params['congestion_z']}, "
                  f"outlier_z={best_params['outlier_z']}")
            print(f"Average KL divergence with best parameters: {best_params['avg_kl_divergence']:.6f}")
        
        return {
            'best_params': best_params,
            'all_results': results
        }

    def visualize_parameter_tuning(self, tuning_results, plot_filename=None):

        # Extract parameter combinations and their scores
        param_combinations = list(tuning_results['all_results'].keys())
        congestion_z_values = sorted(list(set([p[0] for p in param_combinations])))
        outlier_z_values = sorted(list(set([p[1] for p in param_combinations])))
        
        # Create a 2D grid of scores
        scores = np.zeros((len(congestion_z_values), len(outlier_z_values)))
        for i, cz in enumerate(congestion_z_values):
            for j, oz in enumerate(outlier_z_values):
                if (cz, oz) in tuning_results['all_results']:
                    scores[i, j] = tuning_results['all_results'][(cz, oz)]['avg_kl_divergence']
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(scores, cmap='viridis_r')  # Reversed colormap so darker is better
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Avg KL Divergence (lower is better)', rotation=-90, va="bottom")
        
        # Show all ticks and label them
        ax.set_xticks(np.arange(len(outlier_z_values)))
        ax.set_yticks(np.arange(len(congestion_z_values)))
        ax.set_xticklabels(outlier_z_values)
        ax.set_yticklabels(congestion_z_values)
        
        # Label axes
        ax.set_xlabel('Outlier Z')
        ax.set_ylabel('Congestion Z')
        ax.set_title('Parameter Tuning Results')
        
        # Loop over data dimensions and create text annotations
        for i in range(len(congestion_z_values)):
            for j in range(len(outlier_z_values)):
                text = ax.text(j, i, f"{scores[i, j]:.4f}",
                               ha="center", va="center", color="w")
        
        # Highlight the best parameter combination
        best_cz = tuning_results['best_params']['congestion_z']
        best_oz = tuning_results['best_params']['outlier_z']
        best_i = congestion_z_values.index(best_cz)
        best_j = outlier_z_values.index(best_oz)
        
        # Add a red rectangle around the best parameters
        rect = plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1, fill=False, 
                             edgecolor='red', lw=3)
        ax.add_patch(rect)
        
        plt.tight_layout()
        
        if plot_filename:
            plt.savefig(plot_filename)
            print(f"Plot saved to {plot_filename}")
        
        plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Tune parameters for AdaptiveProber')
    parser.add_argument('--simulator', type=int, default=2,
                        help='Simulator version to use (1, 2, or 3)')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of seeds to use for tuning')
    parser.add_argument('--probes', type=int, default=10,
                        help='Probes per second to use during tuning')
    parser.add_argument('--load', action='store_true',
                        help='Load and display previously tuned parameters without running tuning')
    
    args = parser.parse_args()
    
    tuner = ParameterTuner()
    tuned_params_dir = 'tuned_params'
    os.makedirs(tuned_params_dir, exist_ok=True)
    
    if args.load:
        tuned_params_file = f'{tuned_params_dir}/best_params_v{args.simulator}.json'
        if os.path.exists(tuned_params_file):
            with open(tuned_params_file, 'r') as f:
                best_params = json.load(f)
                print("\n=== OPTIMAL PARAMETERS ===")
                print(f"congestion_z={best_params['congestion_z']}, outlier_z={best_params['outlier_z']}")
                print("\nTo use these parameters with AdaptiveProber:")
                print(f"adaptive_prober = AdaptiveProber(simulator, max_probes_per_second, " 
                      f"congestion_z={best_params['congestion_z']}, outlier_z={best_params['outlier_z']})")
                print("\nAverage KL divergence with these parameters:", best_params['avg_kl_divergence'])
                
                viz_file = f'{tuned_params_dir}/parameter_tuning_v{args.simulator}.png'
                if os.path.exists(viz_file):
                    print(f"\nVisualization available at: {viz_file}")
        else:
            print(f"No tuned parameters found for simulator version {args.simulator}")
            print(f"Run without --load to tune parameters first")
    else:
        # Run the parameter tuning
        print(f"Auto-tuning parameters with {args.seeds} seeds...")
        tuning_results = tuner.auto_tune_parameters(
            probes_per_second=args.probes,
            simulator_version=args.simulator,
            num_seeds=args.seeds
        )
        
        best_params = tuning_results['best_params']
        with open(f'{tuned_params_dir}/best_params_v{args.simulator}.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        
        # Visualize tuning results
        tuner.visualize_parameter_tuning(
            tuning_results, 
            plot_filename=f'{tuned_params_dir}/parameter_tuning_v{args.simulator}.png'
        )
        
        print(f"\nBest parameters saved to {tuned_params_dir}/best_params_v{args.simulator}.json")
        
        # Print in a copy-paste friendly format
        print("\n=== OPTIMAL PARAMETERS ===")
        print(f"congestion_z={best_params['congestion_z']}, outlier_z={best_params['outlier_z']}")
        print("\nTo use these parameters with AdaptiveProber:")
        print(f"adaptive_prober = AdaptiveProber(simulator, max_probes_per_second, " 
              f"congestion_z={best_params['congestion_z']}, outlier_z={best_params['outlier_z']})") 
