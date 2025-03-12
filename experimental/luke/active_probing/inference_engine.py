import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from network.ground_truth import GroundTruthNetwork
from experimental.luke.active_probing.path_prober import PathProber

class InferenceEngine:
    def __init__(self):
        self.prober = PathProber()
        
    def _calculate_distribution_confidence(self, delays):
        if len(delays) < 40:
            return 0.0
            
        # bootstrap sampling for confidence intervals
        n_bootstrap = 100
        bootstrap_estimates = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(delays, size=len(delays), replace=True)
            log_sample = np.log(sample)
            mu = np.mean(log_sample)
            sigma = np.std(log_sample)
            bootstrap_estimates.append((mu, sigma))
        
        mu_estimates = [e[0] for e in bootstrap_estimates]
        sigma_estimates = [e[1] for e in bootstrap_estimates]
        
        # confidence interval widths relative to estimates
        mu_width = np.percentile(mu_estimates, 95) - np.percentile(mu_estimates, 5)
        sigma_width = np.percentile(sigma_estimates, 95) - np.percentile(sigma_estimates, 5)
        
        mu_confidence = np.exp(-mu_width)
        sigma_confidence = np.exp(-sigma_width)
        
        # distribution stability using KS test
        half = len(delays) // 2
        ks_stat, _ = stats.ks_2samp(delays[:half], delays[half:])
        ks_confidence = np.exp(-ks_stat * 2)
        
        return min(mu_confidence, sigma_confidence, ks_confidence)

    # store confidence as probes increase
    def collect_confidence_data(self, mu, sigma, max_probes=1000):
        source = self.prober.network.SOURCE
        destination = self.prober.network.DESTINATION
        
        self.prober.network = GroundTruthNetwork(distribution_type="lognormal")
        
        all_delays = []
        confidences = []
        num_probes = []
        ci_widths = []  # 
        batch_size = 20
        
        while len(all_delays) < max_probes:
            new_delays = self.prober.probe_path(source, destination, batch_size)
            all_delays.extend(new_delays)
            
            confidence = self._calculate_distribution_confidence(all_delays)
            confidences.append(confidence)
            num_probes.append(len(all_delays))
            
            # current confidence intervals
            if len(all_delays) >= 40:
                log_delays = np.log(all_delays)
                mu_ci = stats.t.interval(0.95, len(log_delays)-1, 
                                       loc=np.mean(log_delays), 
                                       scale=stats.sem(log_delays))
                ci_widths.append(mu_ci[1] - mu_ci[0])
            else:
                ci_widths.append(np.nan)
        
        return num_probes, confidences, ci_widths

    def test_and_plot_convergence(self, 
                                mu_values=[1.0, 2.0, 3.0], 
                                sigma_values=[0.3, 0.5, 0.7],
                                probe_rates=[10, 20, 50, 100],
                                target_confidence=0.9,
                                max_probes=1000):
        
        #hardcoded output dir
        output_dir = "/Users/lukemciver/Network-Simulation-Group-Research-Project/experimental/luke/active_probing/plots"
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        confidence_data = {}
        

        for mu in mu_values:
            for sigma in sigma_values:
                probes, confidences, ci_widths = self.collect_confidence_data(mu, sigma, max_probes)
                confidence_data[(mu, sigma)] = (probes, confidences, ci_widths)
                results[(mu, sigma)] = {}
                
                # convergence times for each rate
                for rate in probe_rates:
                    try:
                        probes_needed = next(p for p, c in zip(probes, confidences) 
                                           if c >= target_confidence)
                    except StopIteration:
                        probes_needed = max_probes
                    time_to_converge = probes_needed / rate
                    results[(mu, sigma)][rate] = time_to_converge
        
        # convergence time vs rate; varying μ, fixed σ
        fixed_sigma = sigma_values[1]
        plt.figure(figsize=(12, 6))
        for mu in mu_values:
            times = [results[(mu, fixed_sigma)][rate] for rate in probe_rates]
            plt.plot(probe_rates, times, 'o-', label=f'μ={mu}, σ={fixed_sigma}')
        
        plt.xlabel('Probe Rate (probes/second)')
        plt.ylabel('Time to Convergence (seconds)')
        plt.title('Convergence Time vs Probe Rate (Fixed σ)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'convergence_time_mu_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # cnvergence time vs rate; varying σ, fixed μ
        fixed_mu = mu_values[1]
        plt.figure(figsize=(12, 6))
        for sigma in sigma_values:
            times = [results[(fixed_mu, sigma)][rate] for rate in probe_rates]
            plt.plot(probe_rates, times, 'o-', label=f'μ={fixed_mu}, σ={sigma}')
        
        plt.xlabel('Probe Rate (probes/second)')
        plt.ylabel('Time to Convergence (seconds)')
        plt.title('Convergence Time vs Probe Rate (Fixed μ)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'convergence_time_sigma_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # varying μ with fixed σ
        fixed_sigma = sigma_values[1]
        plt.figure(figsize=(12, 6))
        for mu in mu_values:
            probes, confidences, _ = confidence_data[(mu, fixed_sigma)]
            plt.plot(probes, confidences, '-', label=f'μ={mu}, σ={fixed_sigma}')
        
        plt.axhline(y=target_confidence, color='r', linestyle='--', label='Target Confidence')
        plt.xlabel('Number of Probes')
        plt.ylabel('Confidence Score')
        plt.title('Distribution Confidence vs Number of Probes (Fixed σ)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'confidence_mu_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # CI width plot for varying μ
        plt.figure(figsize=(12, 6))
        for mu in mu_values:
            probes, _, ci_widths = confidence_data[(mu, fixed_sigma)]
            plt.plot(probes, ci_widths, '-', label=f'μ={mu}, σ={fixed_sigma}')
        
        plt.xlabel('Number of Probes')
        plt.ylabel('95% CI Width')
        plt.title('Confidence Interval Width vs Number of Probes (Fixed σ)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'ci_width_mu_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        fixed_mu = mu_values[1]
        
        # CI width plot for varying σ
        plt.figure(figsize=(12, 6))
        for sigma in sigma_values:
            probes, _, ci_widths = confidence_data[(fixed_mu, sigma)]
            plt.plot(probes, ci_widths, '-', label=f'μ={fixed_mu}, σ={sigma}')
        
        plt.xlabel('Number of Probes')
        plt.ylabel('95% CI Width')
        plt.title('Confidence Interval Width vs Number of Probes (Fixed μ)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'ci_width_sigma_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # confidence progression plot for varying σ
        plt.figure(figsize=(12, 6))
        for sigma in sigma_values:
            probes, confidences, _ = confidence_data[(fixed_mu, sigma)]
            plt.plot(probes, confidences, '-', label=f'μ={fixed_mu}, σ={sigma}')
        
        plt.axhline(y=target_confidence, color='r', linestyle='--', label='Target Confidence')
        plt.xlabel('Number of Probes')
        plt.ylabel('Confidence Score')
        plt.title('Distribution Confidence vs Number of Probes (Fixed μ)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'confidence_sigma_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return results, confidence_data

if __name__ == "__main__":
    engine = InferenceEngine()
    results, confidence_data = engine.test_and_plot_convergence()
    
    print("\n===== CONVERGENCE STUDY SUMMARY =====")
    for (mu, sigma), data in results.items():
        print(f"\nμ={mu}, σ={sigma}:")
        for rate, time in data.items():
            print(f"  Rate {rate} probes/sec: {time:.2f} seconds")