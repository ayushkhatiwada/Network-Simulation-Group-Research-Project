import numpy as np
import scipy.stats as stats
from active_monitoring.active_interface import Simulator

class StatisticalMonitoringSystem:
    def __init__(self, candidate_intervals: list[float] = None, probes_per_interval: int = 50):
        if candidate_intervals is None:
            candidate_intervals = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        self.candidate_intervals = candidate_intervals
        self.probes_per_interval = probes_per_interval
        self.optimal_interval = None
        self.results = {} 
        self.simulator = Simulator()

    def run_candidate_test(self, interval: float) -> list[float]:
        self.simulator.reset() 
        departure_times = [i * interval for i in range(self.probes_per_interval)]
        events = self.simulator.send_multiple_probes(departure_times)
        delays = [delay for (_, _, delay) in events]
        return delays

    def evaluate_interval(self, delays: list[float]) -> tuple[float, float]:
        delays = np.array(delays)
        gamma_params = stats.gamma.fit(delays, floc=0)
        gamma_stat, gamma_pvalue = stats.kstest(delays, 'gamma', args=gamma_params)
        
        normal_params = stats.norm.fit(delays)
        normal_stat, normal_pvalue = stats.kstest(delays, 'norm', args=normal_params)
        
        return gamma_pvalue, normal_pvalue

    def find_optimal_interval(self) -> float:
        best_interval = None
        best_gamma_pvalue = -1
        
        for interval in self.candidate_intervals:
            delays = self.run_candidate_test(interval)
            gamma_pvalue, normal_pvalue = self.evaluate_interval(delays)
            self.results[interval] = {
                'delays': delays,
                'gamma_pvalue': gamma_pvalue,
                'normal_pvalue': normal_pvalue,
                'mean_delay': np.mean(delays),
                'std_delay': np.std(delays)
            }
            print(f"Interval: {interval}s -> Gamma p-value: {gamma_pvalue:.4f}, Normal p-value: {normal_pvalue:.4f}")
            # For our purpose, select the interval with the highest gamma p-value.
            if gamma_pvalue > best_gamma_pvalue:
                best_gamma_pvalue = gamma_pvalue
                best_interval = interval
        
        self.optimal_interval = best_interval
        print(f"Optimal interval selected: {self.optimal_interval} seconds with gamma p-value: {best_gamma_pvalue:.4f}")
        return best_interval

    def run_monitoring(self) -> list[tuple[float, float, float]]:
        if self.optimal_interval is None:
            self.find_optimal_interval()
        self.simulator.reset()
        departure_times = [i * self.optimal_interval for i in range(self.probes_per_interval)]
        events = self.simulator.send_multiple_probes(departure_times)
        return events

    def get_results(self) -> dict:
        return self.results


if __name__ == "__main__":
    auto_monitor = StatisticalMonitoringSystem(probes_per_interval=50)
    
    optimal_interval = auto_monitor.find_optimal_interval()
    
    events = auto_monitor.run_monitoring()
    
    results = auto_monitor.get_results()
    for interval, data in results.items():
        print(f"\nInterval: {interval}s")
        print(f"  Mean delay: {data['mean_delay']:.2f} ms, Std Dev: {data['std_delay']:.2f} ms")
        print(f"  Gamma KS p-value: {data['gamma_pvalue']:.4f}")
        print(f"  Normal KS p-value: {data['normal_pvalue']:.4f}")
