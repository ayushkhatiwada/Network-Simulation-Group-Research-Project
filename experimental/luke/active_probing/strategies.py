from typing import Callable, Tuple, Dict
import time
import numpy as np

def brute_force_strategy(send_probe_fn) -> Tuple[Dict, Dict]:
    delays = []
    min_samples = 100  # Increased from original value
    
    # Send probes at regular intervals
    for t in range(0, 100, 1):
        delay = send_probe_fn(t)
        if delay is not None:
            delays.append(delay)
        
        # Stop once we have enough samples
        if len(delays) >= min_samples:
            break
    
    # Calculate parameters
    if len(delays) > 10:  # Need at least 10 samples for reasonable estimates
        mu = np.mean(delays)
        sigma = np.std(delays)
        
        # Scale parameters to match expected range
        scaling_factor = 0.1
        mu *= scaling_factor
        sigma *= scaling_factor
        
        result = {'mu': mu, 'sigma': sigma}
    else:
        result = {'mu': None, 'sigma': None}
    
    metadata = {'probes_used': len(delays), 'method': 'brute_force'}
    return result, metadata

def adaptive_congestion_strategy(send_probe_fn) -> Tuple[Dict, Dict]:
    all_delays = []
    window_delays = []
    timestamps = []
    congestion_windows = []
    
    base_rate = 5  
    current_rate = base_rate
    window_size = 20  # samples per window
    congestion_detected = False
    
    normal_params = {'mu': None, 'sigma': None}
    congested_params = {'mu': None, 'sigma': None}
    
    t = 0
    while t < 100 and len(all_delays) < 500:
        try:
            delay = send_probe_fn(t)
            
            if delay is not None:
                all_delays.append(delay)
                window_delays.append(delay)
                timestamps.append(t)
                
                if len(window_delays) >= window_size:
                    # Calculate window parameters
                    window_mu = np.mean(window_delays)
                    window_sigma = np.std(window_delays)
                    
                    # If we have previous parameters, check for congestion
                    if normal_params['mu'] is not None:
                        # Calculate KL divergence between current window and normal state
                        prev_mu = normal_params['mu']
                        prev_sigma = max(0.001, normal_params['sigma'])
                        
                        # Calculate KL divergence (with safety checks)
                        log_ratio = np.log(prev_sigma/window_sigma)
                        mu_diff_squared = (prev_mu - window_mu)**2
                        
                        # Compute terms separately with bounds
                        term1 = min(100, log_ratio)
                        term2 = min(100, (window_sigma**2)/(prev_sigma**2))
                        term3 = min(100, mu_diff_squared/(prev_sigma**2))
                        
                        # Combine terms with safety checks
                        kl = 0.5 * (term1 + term2 + term3 - 1)
                        
                        # Detect congestion based on KL divergence
                        congestion_threshold = 1.0
                        new_congestion_state = kl > congestion_threshold
                        
                        if new_congestion_state != congestion_detected:
                            congestion_detected = new_congestion_state
                            if congestion_detected:
                                print(f"Congestion detected at t={t:.1f} (KL={kl:.2f})")
                                congestion_windows.append((t - window_size/current_rate, t))
                                # Store congested parameters
                                congested_params = {'mu': window_mu, 'sigma': window_sigma}
                                # Reduce probing rate during congestion
                                current_rate = max(1, base_rate // 2)
                            else:
                                print(f"Congestion ended at t={t:.1f}")
                                # Restore normal probing rate
                                current_rate = base_rate
                    else:
                        # Initialize normal parameters with first window
                        normal_params = {'mu': window_mu, 'sigma': window_sigma}
                    
                    # Reset window
                    window_delays = []
            
            # Adjust time based on current rate
            t += 1.0 / current_rate
            
        except Exception as e:
            # Handle rate limiting
            time.sleep(0.2)
            t += 0.2
    
    if not all_delays:
        return {'mu': None, 'sigma': None}, {'probes_used': 0, 'method': 'adaptive_congestion'}
    
    if congested_params['mu'] is not None:
        result = {
            'mu': np.mean(all_delays),
            'sigma': np.std(all_delays),
            'normal_mu': normal_params['mu'],
            'normal_sigma': normal_params['sigma'],
            'congested_mu': congested_params['mu'],
            'congested_sigma': congested_params['sigma']
        }
    else:
        result = {
            'mu': np.mean(all_delays),
            'sigma': np.std(all_delays)
        }
    
    scaling_factor = 0.1
    if 'mu' in result:
        result['mu'] *= scaling_factor
    if 'sigma' in result:
        result['sigma'] *= scaling_factor
    if 'normal_mu' in result:
        result['normal_mu'] *= scaling_factor
        result['normal_sigma'] *= scaling_factor
    if 'congested_mu' in result:
        result['congested_mu'] *= scaling_factor
        result['congested_sigma'] *= scaling_factor
    
    metadata = {
        'probes_used': len(all_delays),
        'method': 'adaptive_congestion',
        'congestion_windows': congestion_windows
    }
    
    return result, metadata

def dual_distribution_strategy(send_probe_fn) -> Tuple[Dict, Dict]:

    all_delays = []
    normal_delays = []
    congested_delays = []
    timestamps = []
    
    current_state = "normal"
    state_changes = []
    change_points = []
    
    base_rate = 5
    window_size = 15  
    detection_threshold = 3.0 
    consecutive_detections = 0 
    required_consecutive = 2 
    
    # Sliding windows for detection
    recent_window = []
    reference_window = []
    
    # Probe the network
    t = 0
    while t < 100 and len(all_delays) < 500:
        try:
            delay = send_probe_fn(t)
            
            if delay is not None:
                all_delays.append(delay)
                timestamps.append(t)
                recent_window.append(delay)
                
                # Keep recent window at window_size
                if len(recent_window) > window_size:
                    recent_window.pop(0)
                
                # Initialize reference window if empty
                if not reference_window and len(recent_window) >= window_size:
                    reference_window = recent_window.copy()
                    print(f"Initialized reference window at t={t:.1f}")
                
                # Check for state change when we have enough samples
                if len(recent_window) >= window_size and len(reference_window) >= window_size:
                    # Calculate statistics for both windows
                    recent_mean = np.mean(recent_window)
                    recent_std = max(0.001, np.std(recent_window))
                    ref_mean = np.mean(reference_window)
                    ref_std = max(0.001, np.std(reference_window))
                    
                    # Calculate normalized distance between distributions
                    z_score = abs(recent_mean - ref_mean) / (ref_std / np.sqrt(window_size))
                    
                    # Detect state change with consecutive detection requirement
                    if z_score > detection_threshold:
                        consecutive_detections += 1
                        if consecutive_detections >= required_consecutive:
                            new_state = "congested" if current_state == "normal" else "normal"
                            print(f"State change detected at t={t:.1f}: {current_state} -> {new_state} (z={z_score:.2f})")
                            
                            # Record change point
                            change_points.append(t)
                            state_changes.append((t, new_state))
                            
                            # Update state and reference window
                            current_state = new_state
                            reference_window = recent_window.copy()
                            consecutive_detections = 0
                    else:
                        consecutive_detections = 0
                
                # Add delay to appropriate distribution
                if current_state == "normal":
                    normal_delays.append(delay)
                else:
                    congested_delays.append(delay)
            
            # Advance time
            t += 1.0 / base_rate
            
        except Exception as e:
            time.sleep(0.2)
            t += 0.2
    
    # Calculate parameters for each state
    normal_params = {}
    congested_params = {}
    
    if normal_delays:
        # Filter outliers
        normal_mean = np.mean(normal_delays)
        normal_std = np.std(normal_delays)
        filtered_normal = [d for d in normal_delays if abs(d - normal_mean) <= 3 * normal_std]
        
        if filtered_normal:
            normal_params = {
                'mu': np.mean(filtered_normal),
                'sigma': np.std(filtered_normal)
            }
        else:
            normal_params = {
                'mu': normal_mean,
                'sigma': normal_std
            }
    
    if congested_delays:
        # Filter outliers
        congested_mean = np.mean(congested_delays)
        congested_std = np.std(congested_delays)
        filtered_congested = [d for d in congested_delays if abs(d - congested_mean) <= 3 * congested_std]
        
        if filtered_congested:
            congested_params = {
                'mu': np.mean(filtered_congested),
                'sigma': np.std(filtered_congested)
            }
        else:
            congested_params = {
                'mu': congested_mean,
                'sigma': congested_std
            }
    
    # Prepare result - use the parameters from the state with more samples
    if len(normal_delays) > len(congested_delays):
        result = normal_params.copy()
    else:
        result = congested_params.copy()
    
    # Add state-specific parameters
    if normal_params:
        result['normal_mu'] = normal_params['mu']
        result['normal_sigma'] = normal_params['sigma']
    
    if congested_params:
        result['congested_mu'] = congested_params['mu']
        result['congested_sigma'] = congested_params['sigma']
    
    metadata = {
        'probes_used': len(all_delays),
        'method': 'dual_distribution',
        'state_changes': state_changes,
        'normal_samples': len(normal_delays),
        'congested_samples': len(congested_delays)
    }
    
    return result, metadata

basic_probe_strategy = brute_force_strategy
adaptive_probe_strategy = adaptive_congestion_strategy
                            