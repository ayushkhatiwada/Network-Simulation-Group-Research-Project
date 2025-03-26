import time
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class BeaucoupSketch:
    """
    BeaucoupSketch for delay profiling based on a coupon collector approach.
    
    The sketch maintains m buckets (or coupons) and uses reservoir sampling to
    store a uniformly random sample of delay measurements in each bucket. The set
    of samples (coupons) can be used to estimate delay metrics like the average delay.
    """
    def __init__(self, b=7):
        self.b = b
        self.m = 1 << b 
        self.coupons = [None] * self.m   
        self.counts = [0] * self.m       
        self.samples = []               

    def update(self, delay):
        """
        Incorporate a new delay measurement into the sketch.
        Uses a hash on the delay to determine which coupon bucket to update,
        and employs reservoir sampling to maintain a uniform sample in that bucket.
        """
        h = hash(delay) & 0xffffffffffffffff
        idx = h % self.m
        self.counts[idx] += 1

        if self.coupons[idx] is None:
            self.coupons[idx] = delay
        else:
            if random.random() < 1.0 / self.counts[idx]:
                self.coupons[idx] = delay

    def estimate(self):
        """
        Compute an estimate of the delay metric from the collected coupons.
        Here, we simply compute the average of the stored delays.
        """
        valid_samples = [x for x in self.coupons if x is not None]
        if not valid_samples:
            return 0.0
        return sum(valid_samples) / len(valid_samples)

    def record_sample(self):
        """
        Record the current delay estimate along with a timestamp (in ms).
        This builds a time series of the delay signature.
        """
        timestamp = time.time() * 1000  
        est = self.estimate()
        self.samples.append((timestamp, est))

    def get_samples(self):
        return self.samples

    @staticmethod
    def passive_delay_estimation(ingress_samples, egress_samples):
        """
        Use the recorded samples at ingress and egress to compute a passive delay estimate.
        This method finds the peak delay estimates in each series and returns the time
        difference between these peaks.
        """
        if not ingress_samples or not egress_samples:
            return None
        ingress_peak = max(ingress_samples, key=lambda x: x[1])
        egress_peak = max(egress_samples, key=lambda x: x[1])
        delay = egress_peak[0] - ingress_peak[0]
        return delay

    def merge(self, other):
        """
        Merge another BeaucoupSketch into this one.
        Both sketches must have the same parameters.
        For each bucket, we combine the samples (here, we choose the lower delay as an example).
        """
        if self.m != other.m or self.b != other.b:
            raise ValueError("Cannot merge BeaucoupSketch with different parameters")
        for i in range(self.m):
            if other.coupons[i] is not None:
                if self.coupons[i] is None:
                    self.coupons[i] = other.coupons[i]
                else:
                    self.coupons[i] = min(self.coupons[i], other.coupons[i])


# =====================
# SIMULATION PARAMETERS
# =====================

N = 5000            
packet_interval_ms = 2  
true_delay_mean = 10.0 
true_delay_std  = 2.0 
sampling_period_ms = 50  

base_delay_ingress = 1.0  
base_delay_std = 0.1      


ingress_bk = BeaucoupSketch(b=7)
egress_bk  = BeaucoupSketch(b=7)

actual_delays = []

current_time_ms = 0.0

next_ingress_sample_time = 0.0
next_egress_sample_time = 0.0

# =====================
# RUN THE SIMULATION
# =====================
for i in range(N):
    arrival_time_ingress = current_time_ms

    ingress_delay = np.random.normal(base_delay_ingress, base_delay_std)
    true_delay = np.random.normal(true_delay_mean, true_delay_std)
    
    arrival_time_egress = arrival_time_ingress + true_delay
    egress_delay = ingress_delay + true_delay
    
    ingress_bk.update(ingress_delay)
    egress_bk.update(egress_delay)
    
    actual_delays.append(true_delay)
    
    if arrival_time_ingress >= next_ingress_sample_time:
        ingress_bk.samples.append((arrival_time_ingress, ingress_bk.estimate()))
        next_ingress_sample_time += sampling_period_ms
    
    if arrival_time_egress >= next_egress_sample_time:
        egress_bk.samples.append((arrival_time_egress, egress_bk.estimate()))
        next_egress_sample_time += sampling_period_ms
    
    current_time_ms += packet_interval_ms

# ======================
# POST-SIMULATION ANALYSIS
# ======================
ingress_samples = ingress_bk.get_samples()
egress_samples  = egress_bk.get_samples()

bk_estimated_delay_ms = BeaucoupSketch.passive_delay_estimation(ingress_samples, egress_samples)

actual_avg_delay_ms = np.mean(actual_delays)

print(f"BeaucoupSketch-based Delay Estimate: {bk_estimated_delay_ms:.2f} ms")
print(f"Actual Average Network Delay:        {actual_avg_delay_ms:.2f} ms")
print(f"Absolute Difference:                 {abs(bk_estimated_delay_ms - actual_avg_delay_ms):.2f} ms")

# ======================
# PLOTTING THE RESULTS
# ======================
plt.figure()
times_ingress = [s[0] for s in ingress_samples]
vals_ingress = [s[1] for s in ingress_samples]
times_egress = [s[0] for s in egress_samples]
vals_egress = [s[1] for s in egress_samples]

plt.scatter(times_ingress, vals_ingress, marker='o', label='Ingress Beaucoup Samples')
plt.scatter(times_egress, vals_egress, marker='x', label='Egress Beaucoup Samples')
plt.title("BeaucoupSketch Delay Signature Over Time")
plt.xlabel("Simulated Time (ms)")
plt.ylabel("Estimated Delay (ms)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
counts, bins, _ = plt.hist(actual_delays, bins=50, density=True, alpha=0.6, label="Actual Delay Histogram")
plt.title("Histogram of Actual Network Delays with Ground Truth")
plt.xlabel("Delay (ms)")
plt.ylabel("Density")

x = np.linspace(min(actual_delays), max(actual_delays), 1000)
true_pdf = norm.pdf(x, true_delay_mean, true_delay_std)
plt.plot(x, true_pdf, 'r-', linewidth=2, label="True Delay Distribution")
plt.legend()
plt.grid(True)
plt.show()
