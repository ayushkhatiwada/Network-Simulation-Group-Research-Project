import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import seaborn as sns
import math
from scipy import stats


def bin_size_variation():
    bin_sizes = np.array([0.00005,
    0.0001,
    0.0005,
    0.001,
    0.0015,
    0.002,
    0.0025,
    0.003,
    0.0035,
    0.004])
    kl_divergences = np.array([0.0667,
    0.0189,
    0.651,
    1.138,
    1.996,
    2.429,
    2.814,
    3.199,
    3.502,
    3.648])

    # Function to compute a polynomial trend line with a given degree.
    # You can change the degree here; the annotation will display the polynomial as-is.
    def compute_trendline(x, y, degree=2):
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        # Generate a smooth x-range
        x_trend = np.linspace(x.min(), x.max(), 200)
        y_trend = poly(x_trend)
        return x_trend, y_trend, poly

    # ----------------------------
    # Graph 1: KL Divergence vs. Bin Size with Trend Line
    # ----------------------------
    x_trend, kl_trend, poly = compute_trendline(bin_sizes, kl_divergences, degree=2)

    plt.figure(figsize=(8, 6))
    plt.scatter(bin_sizes, kl_divergences, color='blue', label='Data Points')
    plt.plot(x_trend, kl_trend, color='blue', linestyle='--', label='Trend Line')
    plt.xlabel('Bin Size (s)', fontsize=12)
    plt.ylabel('KL Divergence', fontsize=12)
    plt.title('KL Divergence vs. Bin Size', fontsize=14)
    plt.grid(True)
    plt.legend()

    # Annotate the graph with the polynomial equation
    equation_text = poly.__str__()  # Get the string representation of the polynomial
    plt.annotate(equation_text, xy=(0.05, 0.85), xycoords='axes fraction',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig("kl_vs_binsize.png")
    plt.show()

def packet_per_second_variation():


    # Sample data (replace these with your actual values)
    packets_per_second = np.array([10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100])
    kl_divergences    = np.array([1.103,
    0.637,
    0.0089,
    0.0403,
    0.0476,
    0.2725,
    0.105,
    0.0096,
    0.0231,
    0.0408])
    drop_rates        = np.array([34.48,
    28.3,
    23.94,
    23.94,
    24.54,
    25.10,
    25.5,
    22.7,
    22.8,
    24.63])
    affected_bins     = np.array([57,
    98,
    141,
    184,
    215,
    252,
    295,
    327,
    353,
    402])

    def compute_trendline(x, y, degree=1):
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        x_trend = np.linspace(x.min(), x.max(), 200)
        y_trend = poly(x_trend)
        return x_trend, y_trend, poly

    # ----------------------------
    # Figure 1: KL Divergence vs. Packets per Second
    # ----------------------------

    plt.figure(figsize=(12, 6))
    
    # Plot line connecting points
    plt.plot(packets_per_second, kl_divergences, marker='o', linestyle='-', 
             color='blue', linewidth=2, markersize=8)
    
    plt.title('KL Divergence vs. Packets per Second', fontsize=15)
    plt.xlabel('Packets per Second', fontsize=12)
    plt.ylabel('KL Divergence', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Annotate each point with its value
    for x, y in zip(packets_per_second, kl_divergences):
        plt.annotate(f'{y:.4f}', (x, y), xytext=(5, 5), 
                     textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.show()

    #----------------------------
    # Figure 2: Drop Rate vs. Packets per Second
    # ----------------------------
    x_trend, drop_trend, poly_drop = compute_trendline(packets_per_second, drop_rates, degree=1)
    plt.figure(figsize=(8, 6))
    plt.scatter(packets_per_second, drop_rates, color='red', label='Data Points')
    plt.plot(x_trend, drop_trend, linestyle='--', color='red', label='Trend Line')
    plt.xlabel('Packets per Second', fontsize=12)
    plt.ylabel('Drop Rate', fontsize=12)
    plt.title('Drop Rate vs. Packets per Second', fontsize=14)
    equation_text_drop = poly_drop.__str__()
    plt.annotate(equation_text_drop, xy=(0.05, 0.85), xycoords='axes fraction',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("drop_rate_vs_packets.png")
    # plt.show() 

    # ----------------------------
    # Figure 3: Affected Bins vs. Packets per Second
    # ----------------------------
    x_trend, bins_trend, poly_bins = compute_trendline(packets_per_second, affected_bins, degree=1)
    plt.figure(figsize=(8, 6))
    plt.scatter(packets_per_second, affected_bins, color='green', label='Data Points')
    plt.plot(x_trend, bins_trend, linestyle='--', color='green', label='Trend Line')
    plt.xlabel('Packets per Second', fontsize=12)
    plt.ylabel('Affected Bins', fontsize=12)
    plt.title('Affected Bins vs. Packets per Second', fontsize=14)
    equation_text_bins = poly_bins.__str__()
    plt.annotate(equation_text_bins, xy=(0.05, 0.85), xycoords='axes fraction',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("affected_bins_vs_packets.png")
    # plt.show() 

    # Finally, display all figures in separate windows.
    plt.show()

def simulation_time_variation():

    def plot_kl_divergence_vs_time(simulation_times, kl_divergences):
        plt.figure(figsize=(12, 7))
        
        # Scatter plot with color coding based on threshold
        scatter = plt.scatter(simulation_times, kl_divergences, 
                            c=kl_divergences <= 0.05, 
                            cmap='RdYlGn', 
                            s=100)
        
        plt.title('KL Divergence vs. Simulation Time', fontsize=15)
        plt.xlabel('Simulation Time (s)', fontsize=12)
        plt.ylabel('KL Divergence', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set y-axis limit to focus on 0-0.05 range
        plt.ylim(0, 0.075)
        
        # Add horizontal line at 0.05 threshold
        plt.axhline(y=0.05, color='red', linestyle='--', label='Ideal KL Divergence Threshold')
        
        # Count points below 0.05
        good_points = sum(kl_divergences <= 0.05)
        total_points = len(kl_divergences)
        percentage = (good_points / total_points) * 100
        
        plt.annotate(f'Points below 0.05: {good_points}/{total_points} ({percentage:.1f}%)', 
                    xy=(0.05, 0.05), 
                    xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        # Add value labels
        for x, y in zip(simulation_times, kl_divergences):
            plt.annotate(f'{y:.4f}', (x, y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9)
        
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # def plot_kl_divergence_vs_time(simulation_times, kl_divergences):
    #     plt.figure(figsize=(12, 7))
        
    #     # Create bar plot
    #     bars = plt.bar(simulation_times, kl_divergences, color='skyblue', edgecolor='navy')
        
    #     # Add threshold line
    #     plt.axhline(y=0.05, color='red', linestyle='--', label='Ideal KL Divergence Threshold (0.05)')
        
    #     plt.title('KL Divergence vs. Simulation Time', fontsize=15)
    #     plt.xlabel('Simulation Time (s)', fontsize=12)
    #     plt.ylabel('KL Divergence', fontsize=12)
    #     plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
    #     # Set y-axis limit to focus on 0-0.05 range
    #     plt.ylim(0, 0.065)
        
    #     # Add value labels on top of each bar
    #     for bar in bars:
    #         height = bar.get_height()
    #         plt.text(bar.get_x() + bar.get_width()/2., height,
    #                 f'{height:.4f}', 
    #                 ha='center', va='bottom', fontsize=9)
        
    #     # Count points below 0.05
    #     good_points = sum(kl_divergences <= 0.05)
    #     total_points = len(kl_divergences)
    #     percentage = (good_points / total_points) * 100
        
    #     plt.annotate(f'Points below 0.05: {good_points}/{total_points} ({percentage:.1f}%)', 
    #                 xy=(0.05, 0.95), 
    #                 xycoords='axes fraction',
    #                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()



    def plot_valid_samples_vs_time_with_linear_fit(simulation_times, valid_samples):
        plt.figure(figsize=(10, 6))
        
        # Scatter plot of original data
        plt.scatter(simulation_times, valid_samples, color='green', label='Data Points')
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(simulation_times, valid_samples)
        
        # Create line of best fit
        line = slope * simulation_times + intercept
        plt.plot(simulation_times, line, color='red', label=f'Linear Fit (R² = {r_value**2:.4f})')
        
        plt.title('Valid Delay Samples vs. Simulation Time with Linear Fit', fontsize=15)
        plt.xlabel('Simulation Time (s)', fontsize=12)
        plt.ylabel('Number of Valid Delay Samples', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add regression equation to the plot
        equation = f'y = {slope:.4f}x + {intercept:.2f}'
        plt.annotate(equation, 
                    xy=(0.05, 0.95), 
                    xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.legend()
        plt.tight_layout()
        plt.show()

    def scatter_plot_with_drop_rate(simulation_times, kl_divergences, drop_rates):
        plt.figure()  # Creates a new window
        scatter = plt.scatter(simulation_times, kl_divergences, 
                            c=drop_rates, cmap='viridis', 
                            s=100, alpha=0.7)
        plt.colorbar(scatter, label='Drop Rate (%)')
        plt.title('KL Divergence vs. Simulation Time\nColored by Drop Rate', fontsize=15)
        plt.xlabel('Simulation Time (s)', fontsize=12)
        plt.ylabel('KL Divergence', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def parallel_coordinates_plot(data):
        from sklearn.preprocessing import MinMaxScaler
        import pandas as pd
        
        columns = ['Simulation Time', 'KL Divergence', 'Valid Delay Samples', 'Drop Rate']
        
        df = pd.DataFrame({
            'Simulation Time': data['simulation_times'],
            'KL Divergence': data['kl_divergences'],
            'Valid Delay Samples': data['valid_samples'],
            'Drop Rate': data['drop_rates']
        })
        
        scaler = MinMaxScaler()
        df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        
        plt.figure()  # Creates a new window
        
        for i in range(len(df_normalized)):
            plt.plot(columns, df_normalized.iloc[i], 
                    color=plt.cm.viridis(df_normalized['KL Divergence'][i]), 
                    alpha=0.5)
        
        plt.title('Parallel Coordinates Plot', fontsize=15)
        plt.xticks(range(len(columns)), columns, rotation=45)
        plt.ylabel('Normalized Values', fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    # Data from the table
    data = {
        'simulation_times': np.array([1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
        'kl_divergences': np.array([0.0340, 0.0107, 0.0326, 0.0526, 0.0061, 0.0350, 0.0098, 0.0170, 0.0595, 0.0145, 0.0027, 0.0015]),
        'valid_samples': np.array([12, 45, 81, 99, 198, 218, 343, 414, 393, 494, 552, 655]),
        'drop_rates': np.array([33.33, 24.54, 24.65, 24.18, 23.84, 23.58, 24.73, 23.85, 24.66, 25.09, 24.87, 24.19])
    }

    # Create plots in separate windows
    plot_kl_divergence_vs_time(data['simulation_times'], data['kl_divergences'])
    plot_valid_samples_vs_time_with_linear_fit(data['simulation_times'], data['valid_samples'])
    scatter_plot_with_drop_rate(data['simulation_times'], data['kl_divergences'], data['drop_rates'])
    parallel_coordinates_plot(data)

if __name__ == "__main__":
    #bin_size_variation()
    #packet_per_second_variation()
    simulation_time_variation()

