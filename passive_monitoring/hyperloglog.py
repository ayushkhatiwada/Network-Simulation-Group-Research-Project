import math
import time

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
        x = hash(item) & 0xffffffffffffffff  # force 64-bit positive value

        idx = x >> (64 - self.b)
        
        w = x & ((1 << (64 - self.b)) - 1)
        rank = self._rho(w, 64 - self.b)
        
        self.registers[idx] = max(self.registers[idx], rank)

    def estimate(self):
        """
        Estimate the number of distinct items (cardinality) added to the sketch.
        This serves as the “signature” of the traffic in a given time window.
        """
        alpha_m = self._alpha()
        indicator = sum(2 ** (-r) for r in self.registers)
        E = alpha_m * self.m * self.m / indicator

        if E <= 2.5 * self.m:
            V = self.registers.count(0)
            if V != 0:
                E = self.m * math.log(self.m / V)
        return E

    def record_sample(self):
        """
        Record the current estimate along with a timestamp (in ms).
        This method allows the sketch to build a time series signature.
        """
        timestamp = time.time() * 1000  
        est = self.estimate()
        self.samples.append((timestamp, est))

    def get_samples(self):
        return self.samples

    def passive_delay_estimation(ingress_samples, egress_samples):
        if not ingress_samples or not egress_samples:
            return None
        
        ingress_peak = max(ingress_samples, key=lambda x: x[1])
        egress_peak = max(egress_samples, key=lambda x: x[1])
        delay = egress_peak[0] - ingress_peak[0]
        return delay

    def merge(self, other):
        """
        Merge another HLL sketch into this one.
        Both sketches must have the same precision (b) and number of registers (m).
        """
        if self.m != other.m or self.b != other.b:
            raise ValueError("Cannot merge HLL sketches with different parameters")
        self.registers = [max(r1, r2) for r1, r2 in zip(self.registers, other.registers)]
