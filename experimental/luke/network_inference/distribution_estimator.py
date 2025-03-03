import numpy as np
from scipy import stats
from typing import Dict, Tuple, List, Optional


class DistributionEstimator:    
    @staticmethod
    def fit_gamma(data: np.ndarray) -> Tuple[float, float]:
        # Fit gamma distribution
        shape, loc, scale = stats.gamma.fit(data, floc=0)
        return shape, scale
    
    @staticmethod
    def fit_normal(data: np.ndarray) -> Tuple[float, float]:
        # Fit normal distribution
        mean, std = stats.norm.fit(data)
        return mean, std
    
    @staticmethod
    def test_distribution_fit(data: np.ndarray):
        # Fit both distributions
        shape, loc, scale = stats.gamma.fit(data, floc=0)
        mean, std = stats.norm.fit(data)
        
        # Calculate goodness of fit using KS test
        _, p_gamma = stats.kstest(data, 'gamma', args=(shape, loc, scale))
        _, p_normal = stats.kstest(data, 'norm', args=(mean, std))
        
        # Return the distribution with better fit (higher p-value)
        return 'gamma' if p_gamma > p_normal else 'normal'