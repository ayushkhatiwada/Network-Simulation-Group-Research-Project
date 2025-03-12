import numpy as np
from scipy import stats
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt


class DistributionEstimator:    
    def __init__(self):
        pass
    
    @staticmethod
    def fit_gamma(data: np.ndarray) -> Tuple[float, float]:
        shape, loc, scale = stats.gamma.fit(data, floc=0)
        return shape, scale
    
    @staticmethod
    def fit_normal(data: np.ndarray) -> Tuple[float, float]:
        mean, std = stats.norm.fit(data)
        return mean, std
    
    def test_distribution_fit(self, data):
        # Remove extreme outliers for more stable fitting
        data = np.array(data)
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
        
        # min samples to fit
        if len(filtered_data) < 5:  
            filtered_data = data
        
        try:
            _, p_normal = stats.normaltest(filtered_data)
        except:
            p_normal = 0
            
        # data must be positive
        positive_data = filtered_data[filtered_data > 0]
        if len(positive_data) > 5:
            log_data = np.log(positive_data)
            try:
                _, p_lognormal = stats.normaltest(log_data)
            except:
                p_lognormal = 0
        else:
            p_lognormal = 0
            
        try:
            gamma_params = stats.gamma.fit(filtered_data, floc=0)
            _, p_gamma = stats.kstest(filtered_data, 'gamma', gamma_params)
        except:
            p_gamma = 0
            
        # p-values (higher is better)
        p_values = {
            'normal': p_normal,
            'gamma': p_gamma, 
            'lognormal': p_lognormal
        }
        
        best_fit = max(p_values, key=p_values.get)
        return best_fit
    
    def fit_gamma(self, data):
        data = np.array(data)
        data = data[data > 0]
        
        if len(data) < 2:
            return 1.0, np.mean(data) if len(data) > 0 else 1.0
        
        mean = np.mean(data)
        var = np.var(data)
        
        if var == 0:
            return 1.0, mean
            
        shape = mean**2 / var
        scale = var / mean
        
        return shape, scale
    
    def fit_lognormal(self, data):
        data = np.array(data)
        data = data[data > 0]
        
        if len(data) < 2:
            return 0.0, 1.0
            
        log_data = np.log(data)
        mu = np.mean(log_data)
        sigma = np.std(log_data)
        
        return mu, sigma