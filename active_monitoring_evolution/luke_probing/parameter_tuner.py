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
from active_simulator_v0 import ActiveSimulator_v0

from base_prober import BaseProber
from adaptive_prober import AdaptiveProber
from probing_evaluator import ProbingEvaluator  

        
# command line args
# python parameter_tuner.py --simulator=2 --seeds=10
# python parameter_tuner.py --simulator=2 --load

class ParameterTuner:
    def __init__(self, max_departure_time=5):
        self.max_departure_time = max_departure_time
        self.evaluator = ProbingEvaluator(max_departure_time)
        
    def calculate_kl_divergence(self, pred_mean, pred_std, true_mean, true_std):
        return self.evaluator.calculate_kl_divergence(pred_mean, pred_std, true_mean, true_std)
    
    def auto_tune_parameters(self, probes_per_second=100, simulator_version=2, simulation_duration=100, num_seeds=5, verbose=True):
        """Tune parameters for a specific simulation duration and probe rate."""
        # params
        # probes, simversion, simulation_duration
        # num_seeds (sample size for how much we test)
        # verbose (print progress)
        # returns a dict with best params and all results
    
        congestion_z_values = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        outlier_z_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        
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
                    
                    if simulator_version == 3:
                        base_simulator = ActiveSimulator_v3(simulation_duration=simulation_duration, random_seed=seed)
                    elif simulator_version == 2:
                        base_simulator = ActiveSimulator_v2(simulation_duration=simulation_duration, random_seed=seed)
                    elif simulator_version == 1:
                        base_simulator = ActiveSimulator_v1(simulation_duration=simulation_duration, random_seed=seed)
                    else: # Default to v0
                        base_simulator = ActiveSimulator_v0(simulation_duration=simulation_duration, random_seed=seed)
                    
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
                    ground_truth_params_list = base_simulator.network.get_distribution_parameters()
                    if ground_truth_params_list:
                        ground_truth_dict = ground_truth_params_list[0][2] # Extract the dict
                        gt_mean = ground_truth_dict.get('mean', 0) # Use .get for safety
                        gt_std = ground_truth_dict.get('std', 1)   # Use .get for safety
                        if verbose:
                             print(f"    Ground truth parameters extracted: mean={gt_mean:.3f}, std={gt_std:.3f}")
                    else:
                        # Handle case where the network might not have the edge somehow
                        print(f"    Warning: Could not retrieve ground truth params for seed {seed}. Using defaults.")
                        gt_mean = 0
                        gt_std = 1 # Fallback defaults
                    
                    adaptive_metrics = adaptive_prober.get_metrics()
                    
                    # Use final estimates (handle potential empty metrics)
                    if adaptive_metrics['metrics_per_timeslot']:
                        final_adaptive_mean = adaptive_metrics['metrics_per_timeslot'][-1][1]
                        final_adaptive_std = adaptive_metrics['metrics_per_timeslot'][-1][2]
                    else:
                        # Handle case where adaptive prober produced no metrics
                        print(f"    Warning: Adaptive prober produced no metrics for seed {seed}. Using default estimates.")
                        final_adaptive_mean = 0.0
                        final_adaptive_std = 1.0 # Avoid zero std
                    
                    # Calculate KL divergence using extracted gt_mean and gt_std
                    adaptive_kl = self.calculate_kl_divergence(
                        final_adaptive_mean, 
                        final_adaptive_std, 
                        gt_mean, 
                        gt_std
                    )
                    
                    param_results.append({
                        'seed': seed,
                        'kl_divergence': adaptive_kl,
                        'final_mean': final_adaptive_mean,
                        'final_std': final_adaptive_std,
                        'true_mean': gt_mean, # Store extracted ground truth
                        'true_std': gt_std    # Store extracted ground truth
                    })
                    if verbose:
                        print(f"    Seed {seed} KL: {adaptive_kl:.6f}")
                
                # Average KL divergence across all seeds
                avg_kl = np.mean([r['kl_divergence'] for r in param_results])
                
                if verbose:
                    print(f"  Average KL for ({congestion_z}, {outlier_z}): {avg_kl:.6f}")
                
                # Store results for this parameter combination
                param_key = (congestion_z, outlier_z)
                results[param_key] = {
                    'avg_kl_divergence': avg_kl,
                    'details': param_results
                }
                
                # Update best parameters if this is better
                if avg_kl < best_score:
                    best_score = avg_kl
                    best_params = {
                        'congestion_z': congestion_z,
                        'outlier_z': outlier_z,
                        'avg_kl_divergence': avg_kl,
                        'simulator_version': simulator_version,
                        'simulation_duration': simulation_duration,
                        'probes_per_second': probes_per_second,
                        'num_seeds': num_seeds
                    }
        
        if verbose:
            print("\nParameter tuning complete!")
            print(f"Best parameters for duration={simulation_duration}s, pps={probes_per_second}: congestion_z={best_params['congestion_z']}, "
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
    parser.add_argument('--seeds', type=int, default=10,
                        help='Number of seeds to use for tuning')
    parser.add_argument('--probes', type=int, default=100,
                        help='Probes per second to use during tuning')
    parser.add_argument('--duration', type=int, default=100,
                        help='Simulation duration for tuning')
    parser.add_argument('--all-durations', action='store_true',
                        help='Tune parameters for all durations (10,20,...,100)')
    parser.add_argument('--load', action='store_true',
                        help='Load and display previously tuned parameters without running tuning')
    
    args = parser.parse_args()
    
    tuner = ParameterTuner()
    tuned_params_dir = 'tuned_params'
    os.makedirs(tuned_params_dir, exist_ok=True)
    
    if args.load:
        if args.all_durations:
            durations = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            for duration in durations:
                tuned_params_file = f'{tuned_params_dir}/best_params_v{args.simulator}_d{duration}.json'
                if os.path.exists(tuned_params_file):
                    with open(tuned_params_file, 'r') as f:
                        best_params = json.load(f)
                        print(f"\n=== OPTIMAL PARAMETERS FOR DURATION={duration}s ===")
                        print(f"congestion_z={best_params['congestion_z']}, outlier_z={best_params['outlier_z']}")
                        print(f"Average KL divergence: {best_params['avg_kl_divergence']}")
                else:
                    print(f"No tuned parameters found for simulator version {args.simulator}, duration {duration}s")
        else:
            tuned_params_file = f'{tuned_params_dir}/best_params_v{args.simulator}_d{args.duration}.json'
            if os.path.exists(tuned_params_file):
                with open(tuned_params_file, 'r') as f:
                    best_params = json.load(f)
                    print("\n=== OPTIMAL PARAMETERS ===")
                    print(f"For duration={best_params['simulation_duration']}s, probes_per_second={best_params['probes_per_second']}")
                    print(f"congestion_z={best_params['congestion_z']}, outlier_z={best_params['outlier_z']}")
                    print("\nTo use these parameters with AdaptiveProber:")
                    print(f"adaptive_prober = AdaptiveProber(simulator, max_probes_per_second, " 
                          f"congestion_z={best_params['congestion_z']}, outlier_z={best_params['outlier_z']})")
                    print("\nAverage KL divergence with these parameters:", best_params['avg_kl_divergence'])
                    
                    viz_file = f'{tuned_params_dir}/parameter_tuning_v{args.simulator}_d{args.duration}.png'
                    if os.path.exists(viz_file):
                        print(f"\nVisualization available at: {viz_file}")
            else:
                print(f"No tuned parameters found for simulator version {args.simulator}, duration {args.duration}s")
                print(f"Run without --load to tune parameters first")
    else:
        # Run the parameter tuning
        if args.all_durations:
            durations = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            all_best_params = {}
            
            for duration in durations:
                print(f"\n\n==== TUNING FOR DURATION {duration}s ====\n")
                tuning_results = tuner.auto_tune_parameters(
                    probes_per_second=args.probes,
                    simulator_version=args.simulator,
                    simulation_duration=duration,
                    num_seeds=args.seeds
                )
                
                best_params = tuning_results['best_params']
                all_best_params[duration] = best_params
                
                # Save individual parameters
                with open(f'{tuned_params_dir}/best_params_v{args.simulator}_d{duration}.json', 'w') as f:
                    json.dump(best_params, f, indent=2)
                
                # Visualize tuning results
                tuner.visualize_parameter_tuning(
                    tuning_results, 
                    plot_filename=f'{tuned_params_dir}/parameter_tuning_v{args.simulator}_d{duration}.png'
                )
            
            # Save a summary file with all parameters
            with open(f'{tuned_params_dir}/all_best_params_v{args.simulator}.json', 'w') as f:
                json.dump(all_best_params, f, indent=2)
                
            print("\n\n==== ALL TUNING COMPLETE ====")
            print(f"All parameters saved to {tuned_params_dir}/all_best_params_v{args.simulator}.json")
            print("\nOptimal parameters summary:")
            for duration, params in all_best_params.items():
                print(f"Duration {duration}s: congestion_z={params['congestion_z']}, outlier_z={params['outlier_z']}, KL={params['avg_kl_divergence']:.6f}")
        else:
            # Run for single duration
            print(f"Auto-tuning parameters with {args.seeds} seeds for duration={args.duration}s...")
            tuning_results = tuner.auto_tune_parameters(
                probes_per_second=args.probes,
                simulator_version=args.simulator,
                simulation_duration=args.duration,
                num_seeds=args.seeds
            )
            
            best_params = tuning_results['best_params']
            with open(f'{tuned_params_dir}/best_params_v{args.simulator}_d{args.duration}.json', 'w') as f:
                json.dump(best_params, f, indent=2)
            
            # Visualize tuning results
            tuner.visualize_parameter_tuning(
                tuning_results, 
                plot_filename=f'{tuned_params_dir}/parameter_tuning_v{args.simulator}_d{args.duration}.png'
            )
            
            print(f"\nBest parameters saved to {tuned_params_dir}/best_params_v{args.simulator}_d{args.duration}.json")
            
            # Print in a copy-paste friendly format
            print("\n=== OPTIMAL PARAMETERS ===")
            print(f"congestion_z={best_params['congestion_z']}, outlier_z={best_params['outlier_z']}")
            print("\nTo use these parameters with AdaptiveProber:")
            print(f"adaptive_prober = AdaptiveProber(simulator, max_probes_per_second, " 
                  f"congestion_z={best_params['congestion_z']}, outlier_z={best_params['outlier_z']})")
