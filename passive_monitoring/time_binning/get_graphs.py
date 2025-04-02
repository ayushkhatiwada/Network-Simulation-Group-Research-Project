import matplotlib.pyplot as plt
import numpy as np


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
    pass