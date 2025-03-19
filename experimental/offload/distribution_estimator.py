import numpy as np
from scipy import stats
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt


class DistributionEstimator:    
    def __init__(self):
        pass
    
    @staticmethod
    def fit_gamma(data: np.ndarray) -> Tuple[float, float]:
        data = data[data > 0]
        if len(data) < 2:
            return 1.0, np.mean(data) if len(data) > 0 else 1.0
            
        mean = np.mean(data)
        var = np.var(data)
        shape = mean**2 / var if var > 0 else 1.0
        scale = var / mean if mean > 0 else 1.0
        return max(shape, 0.1), max(scale, 0.1)
    
    @staticmethod
    def fit_normal(data: np.ndarray) -> Tuple[float, float]:
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        return max(mean, 0.001), max(std, 0.001)
    
    @staticmethod
    def fit_lognormal(data: np.ndarray, metadata: Dict = None) -> Tuple[float, float]:
        data = data[data > 0]
        if len(data) < 2:
            return 0.0, 1.0
            
        log_data = np.log(data)
        mu = np.mean(log_data)
        sigma = np.std(log_data, ddof=1)
        
        if metadata and 'congestion_factor' in metadata:
            mu += np.log(metadata['congestion_factor'])
            
        return max(mu, 0.001), max(sigma, 0.001)

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
    
    def fit_lognormal(self, data, metadata=None):
        data = np.array(data)
        data = data[data > 0]
        
        if len(data) < 2:
            return 0.0, 1.0
            
        log_data = np.log(data)
        mu = np.mean(log_data)
        sigma = np.std(log_data, ddof=1)
        
        # Compensate for congestion scaling if metadata provided
        if metadata and 'congestion_factor' in metadata:
            mu += np.log(metadata['congestion_factor'])
        
        sigma = max(sigma, 1e-6)
        return mu, sigma

    def estimate_parameters(self, data, metadata=None):
        # Convert to numpy array first
        data_array = np.array(data, dtype=np.float64)
        
        # Now use numpy operations
        finite_mask = np.isfinite(data_array)
        if np.sum(finite_mask) < 10:
            return self._empty_params(metadata.get('distribution_type', 'lognormal'))
            
        clean_data = data_array[finite_mask]
        
        if len(clean_data) < 10:
            return self._empty_params(metadata.get('distribution_type', 'lognormal'))
            
        # Force consistent parameter keys with network type
        if metadata and 'distribution_type' in metadata:
            dist_type = metadata['distribution_type']
        else:
            dist_type = self.test_distribution_fit(data)
        
        fit_methods = {
            'lognormal': lambda: self.fit_lognormal(data, metadata),
            'gamma': lambda: self.fit_gamma(data),
            'normal': lambda: self.fit_normal(data)
        }
        
        if dist_type not in fit_methods:
            dist_type = 'lognormal'  # default fallback
            
        params = fit_methods[dist_type]()
        
        return {
            'distribution_type': dist_type,
            **dict(zip(
                ['mu', 'sigma'] if dist_type == 'lognormal' else 
                ['shape', 'scale'] if dist_type == 'gamma' else 
                ['mean', 'std'],
                params
            ))
        }

    def _empty_params(self, dist_type):
        """Return safe default parameters for empty/invalid data"""
        return {
            'distribution_type': dist_type,
            'mu': 0.0 if dist_type == 'lognormal' else np.nan,
            'sigma': 1.0 if dist_type == 'lognormal' else np.nan,
            'shape': 1.0 if dist_type == 'gamma' else np.nan,
            'scale': 1.0 if dist_type == 'gamma' else np.nan,
            'mean': 0.0 if dist_type == 'normal' else np.nan,
            'std': 1.0 if dist_type == 'normal' else np.nan
        }