import math
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm 

class HLLSketch:
    def __init__(self, b=7):
        self.b = b
        self.m = 1 << b 
        self.registers = [0] * self.m
        self.samples = []

    def _alpha(self):
        if self.m == 16:
            return 0.673
        elif self.m == 32:
            return 0.697
        elif self.m == 64:
            return 0.709
        else:
            return 0.7213 / (1 + 1.079 / self.m)

    def _rho(self, w, max_bits):
        if w == 0:
            return max_bits + 1
        return max_bits - w.bit_length() + 1

    def update(self, item):
        x = hash(item) & 0xffffffffffffffff

        idx = x >> (64 - self.b)

        w = x & ((1 << (64 - self.b)) - 1)
        rank = self._rho(w, 64 - self.b)
        self.registers[idx] = max(self.registers[idx], rank)

    def estimate(self):
        alpha_m = self._alpha()
        indicator = sum(2 ** (-r) for r in self.registers)
        E = alpha_m * self.m * self.m / indicator

        if E <= 2.5 * self.m:
            V = self.registers.count(0)
            if V != 0:
                E = self.m * math.log(self.m / V)
        return E

    def record_sample(self, current_time_ms):
        est = self.estimate()
        self.samples.append((current_time_ms, est))

    def get_samples(self):
        return self.samples

    def passive_delay_estimation(ingress_samples, egress_samples):
        if not ingress_samples or not egress_samples:
            return None
        ingress_peak = max(ingress_samples, key=lambda x: x[1])
        egress_peak = max(egress_samples, key=lambda x: x[1])

        delay_ms = egress_peak[0] - ingress_peak[0]
        return delay_ms

# =====================
# SIMULATION PARAMETERS
# =====================

N = 5000             
packet_interval_ms = 2 
true_delay_mean = 10.0
true_delay_std  = 2.0  
sampling_period_ms = 50  

ingress_hll = HLLSketch(b=7)
egress_hll  = HLLSketch(b=7)

ingress_times = []
egress_times = []

# =====================
# RUN THE SIMULATION
# =====================

current_time_ms = 0.0  

next_ingress_sample_time = 0.0
next_egress_sample_time  = 0.0

actual_delays = []

for i in range(N):
    arrival_time_ingress = current_time_ms

    d = np.random.normal(true_delay_mean, true_delay_std)

    arrival_time_egress = arrival_time_ingress + d

    ingress_hll.update(i)
    ingress_times.append(arrival_time_ingress)

    egress_hll.update((i, arrival_time_egress))
    egress_times.append(arrival_time_egress)
    
    actual_delays.append(d)

    if arrival_time_ingress >= next_ingress_sample_time:
        ingress_hll.record_sample(arrival_time_ingress)
        next_ingress_sample_time += sampling_period_ms

    if arrival_time_egress >= next_egress_sample_time:
        egress_hll.record_sample(arrival_time_egress)
        next_egress_sample_time += sampling_period_ms
    current_time_ms += packet_interval_ms

# ==============
# POST-SIMULATION
# ==============

ingress_samples = ingress_hll.get_samples()
egress_samples  = egress_hll.get_samples()

hll_estimated_delay_ms = HLLSketch.passive_delay_estimation(ingress_samples, egress_samples) / 1000

actual_avg_delay_ms = np.mean(actual_delays)

print(f"HLL-based Delay Estimate: {hll_estimated_delay_ms:.2f} ms")
print(f"Actual Average Delay:     {actual_avg_delay_ms:.2f} ms")
diff = abs(hll_estimated_delay_ms - actual_avg_delay_ms)
print(f"Difference (absolute):    {diff:.2f} ms")

# ================
# PLOT THE RESULTS
# ================
plt.figure()
times_ingress = [s[0] for s in ingress_samples]
vals_ingress = [s[1] for s in ingress_samples]
times_egress = [s[0] for s in egress_samples]
vals_egress = [s[1] for s in egress_samples]

plt.scatter(times_ingress, vals_ingress, marker='o', label='Ingress HLL Samples')
plt.scatter(times_egress, vals_egress, marker='x', label='Egress HLL Samples')
plt.title("HLL Signature Over Time")
plt.xlabel("Time (ms)")
plt.ylabel("Estimated Distinct Count")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
counts, bins, _ = plt.hist(actual_delays, bins=50, density=True, alpha=0.6, label="Actual Delay Histogram")
plt.title("Histogram of Actual Delays with Ground Truth")
plt.xlabel("Delay (ms)")
plt.ylabel("Density")

x = np.linspace(min(actual_delays), max(actual_delays), 1000)
true_pdf = norm.pdf(x, true_delay_mean, true_delay_std)
plt.plot(x, true_pdf, 'r-', linewidth=2, label="True Delay Distribution")
plt.legend()
plt.grid(True)
plt.show()
