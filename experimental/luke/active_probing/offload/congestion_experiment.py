import os
import argparse
import numpy as np
from experimental.luke.active_probing import (
    PathProber, DistributionEstimator,
    congestion_aware_probing_experiment,
    plot_delay_distributions
)

def main():
    parser = argparse.ArgumentParser(description="Network Congestion Probing Experiment")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    prober = PathProber(use_congestion=True)
    estimator = DistributionEstimator()
    
    results = congestion_aware_probing_experiment(
        prober=prober,
        estimator=estimator,
        source=1,
        target=5,
        congestion_levels=[0.0, 0.2, 0.4, 0.6, 0.8],
        probes_per_level=500,
        output_dir=args.output_dir
    )
    
    delays_by_congestion = {level: res['samples'] for level, res in results.items()}
    true_params = {level: res['true_params'] for level, res in results.items()}
    estimated_params = {level: res['estimated_params'] for level, res in results.items()}
    
    plot_delay_distributions(
        delays_by_congestion,
        true_params,
        estimated_params,
        distribution_type=prober.network.distribution_type,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()