from experimental.luke.active_probing.testing_framework import StrategyEvaluator
from experimental.luke.active_probing.strategies import (
    brute_force_strategy,
    adaptive_congestion_strategy,
    dual_distribution_strategy
)
from experimental.luke.active_probing.plotting import (
    plot_parameter_errors,
    plot_kl_divergence,
    plot_resource_usage,
    plot_strategy_comparison
)

def run_experiments():
    evaluator = StrategyEvaluator()
    
    print("=== Running Brute Force Strategy Evaluation ===")
    brute_results = evaluator.evaluate(
        brute_force_strategy,
        param_targets=['mu', 'sigma'],
        congestion_levels=[0.0, 0.3, 0.6, 0.9],
        trials=3
    )
    print(evaluator.generate_report(brute_results))
    
    # Generate plots for brute force strategy
    plot_parameter_errors(brute_results, "Brute_Force")
    plot_kl_divergence(brute_results, "Brute_Force")
    plot_resource_usage(brute_results, "Brute_Force")
    
    print("\n=== Running Adaptive Congestion Strategy Evaluation ===")
    adaptive_results = evaluator.evaluate(
        adaptive_congestion_strategy,
        param_targets=['mu', 'sigma'],
        congestion_levels=[0.0, 0.6, 0.9],  # Test different congestion levels
        trials=2
    )
    print(evaluator.generate_report(adaptive_results))
    
    # Generate plots for adaptive strategy
    plot_parameter_errors(adaptive_results, "Adaptive")
    plot_kl_divergence(adaptive_results, "Adaptive")
    plot_resource_usage(adaptive_results, "Adaptive")
    
    print("\n=== Running Dual Distribution Strategy Evaluation ===")
    dual_results = evaluator.evaluate(
        dual_distribution_strategy,
        param_targets=['mu', 'sigma'],
        congestion_levels=[0.0, 0.6, 0.9],
        trials=2
    )
    print(evaluator.generate_report(dual_results))
    
    # Generate plots for dual distribution strategy
    plot_parameter_errors(dual_results, "Dual_Distribution")
    plot_kl_divergence(dual_results, "Dual_Distribution")
    plot_resource_usage(dual_results, "Dual_Distribution")
    
    # Compare all strategies
    plot_strategy_comparison(brute_results, adaptive_results, dual_results)
    
    print("\nPlots have been saved to the 'plots' directory")

if __name__ == "__main__":
    run_experiments()
    print("\nExperiment sequence completed")