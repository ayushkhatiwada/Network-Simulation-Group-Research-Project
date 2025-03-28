import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to path so active_simulator files can import each other
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from active_simulator_v3 import ActiveSimulator_v3
from adaptive_prober import ActiveProber

def is_in_congestion_interval(time, intervals):
    """Check if a given time is in any congestion interval"""
    for start, end in intervals:
        if start <= time <= end:
            return True
    return False

def main():
    # Set up the simulator with some congestion
    intensity = 2.0  # Moderate congestion intensity
    probes_per_second = 10
    
    simulator = ActiveSimulator_v3(max_intensity=intensity)
    simulator.max_probes_per_second = probes_per_second
    simulator.probe_count_per_second = {}
    
    # Create and run the adaptive prober with adjusted parameters
    prober = ActiveProber(simulator, 
                         max_probes_per_second=10,
                         sliding_window=10,    # Window for delay analysis
                         min_samples=3,        # Minimum samples needed
                         confidence_level=0.95, # Statistical confidence
                         learning_rate=0.1)    # For learning congestion multiplier
    prober.probe()
    
    # Get the actual congestion intervals for comparison
    congestion_intervals = simulator.congestion_intervals
    
    # Print congestion intervals for reference
    #print("Actual Congestion Intervals:")
    for i, (start, end) in enumerate(congestion_intervals):
        intensity = simulator.congestion_intensities[i]
        print(f"  {start:.2f}s to {end:.2f}s - Intensity: {intensity:.2f}")
    
    # Print delay metrics over time
    #print("\nDelay Metrics Over Time:")
    delays = [d for t, d in prober.probe_history if d is not None]
    times = [t for t, d in prober.probe_history if d is not None]
    
    window_size = 10  # Size of sliding window for metrics
    for i in range(0, len(times), window_size):
        window_delays = delays[i:i+window_size]
        if window_delays:
            window_start = times[i]
            window_end = times[min(i+window_size-1, len(times)-1)]
            #print(f"\nTime window {window_start:.1f}s - {window_end:.1f}s:")
            #print(f"  Mean Delay: {np.mean(window_delays):.2f} ms")
            #print(f"  Median Delay: {np.median(window_delays):.2f} ms")
            #print(f"  Std Dev: {np.std(window_delays):.2f} ms")
            #print(f"  Min: {min(window_delays):.2f} ms")
            #print(f"  Max: {max(window_delays):.2f} ms")
            if prober.reference_stats:
                ratio = np.mean(window_delays) / prober.reference_stats['mean']
                print(f"  Ratio to baseline: {ratio:.2f}x")
    
    # Print detailed detection results
    #print("\nDetailed Detection Results:")
    for second in range(int(simulator.max_departure_time)):
        detection_result = prober.congestion_detection_results.get(second, {})
        actual_state = "Congested" if is_in_congestion_interval(second, congestion_intervals) else "Normal"
        detected_state = "Congested" if detection_result.get('detected', False) else "Normal"
        reason = detection_result.get('reason', 'no_data')
        
        #print(f"Second {second}:")
        #print(f"  Actual State: {actual_state}")
        #print(f"  Detected State: {detected_state}")
        #print(f"  Detection Reason: {reason}")
        #if 'recent_mean' in detection_result and 'reference_mean' in detection_result:
            #print(f"  Recent Mean Delay: {detection_result['recent_mean']:.2f} ms")
            #print(f"  Reference Mean Delay: {detection_result['reference_mean']:.2f} ms")
    
    # Prepare data for plotting
    seconds = []
    actual_congestion = []
    detected_congestion = []
    
    for second in range(int(simulator.max_departure_time)):
        seconds.append(second)
        
        # Determine if this second is in a congestion interval
        is_congested = is_in_congestion_interval(second, congestion_intervals)
        actual_congestion.append(1 if is_congested else 0)
        
        # Get detection result if available
        detection_result = prober.congestion_detection_results.get(second, {})
        detected = detection_result.get('detected', False)
        
        detected_congestion.append(1 if detected else 0)
    
    # Plot the comparison
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Congestion Detection
    plt.subplot(211)
    plt.step(seconds, actual_congestion, 'b-', where='post', label='Actual Congestion')
    plt.step(seconds, detected_congestion, 'r-', where='post', label='Detected Congestion')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Congestion State')
    plt.title('Congestion Detection Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Delay Metrics and Learned Multiplier
    plt.subplot(212)
    delays = [d for t, d in prober.probe_history if d is not None]
    times = [t for t, d in prober.probe_history if d is not None]
    plt.plot(times, delays, 'b.', alpha=0.3, label='Probe Delays')
    
    if prober.congestion_multiplier:
        plt.axhline(y=prober.reference_stats['mean'] * prober.congestion_multiplier, 
                    color='r', linestyle='--', 
                    label=f'Learned Threshold ({prober.congestion_multiplier:.2f}x)')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Delay (ms)')
    plt.title('Delay Metrics Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('congestion_detection_performance.png')
    plt.show()
    
    # Calculate detection metrics
    true_positives = sum(1 for a, d in zip(actual_congestion, detected_congestion) if a == 1 and d == 1)
    false_positives = sum(1 for a, d in zip(actual_congestion, detected_congestion) if a == 0 and d == 1)
    true_negatives = sum(1 for a, d in zip(actual_congestion, detected_congestion) if a == 0 and d == 0)
    false_negatives = sum(1 for a, d in zip(actual_congestion, detected_congestion) if a == 1 and d == 0)
    
    # Calculate metrics
    accuracy = (true_positives + true_negatives) / len(seconds) if len(seconds) > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\nCongestion Detection Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Negatives: {false_negatives}")

if __name__ == "__main__":
    main() 