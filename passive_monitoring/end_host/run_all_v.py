import csv
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from passive_monitoring.end_host.end_host_v0 import evolution_0_find_optimal_window
from passive_monitoring.end_host.end_host_v1 import evolution_1_dropped_probes
from passive_monitoring.end_host.end_host_v2 import evolution_2_with_congestion

# Directory to save outputs
output_dir = Path("experiment_results")
output_dir.mkdir(exist_ok=True)

# Store all evolution results
all_results = {}

# Evolution 0
results_0 = evolution_0_find_optimal_window(10, 500, 10, False, None)
df_0 = pd.DataFrame(results_0, columns=["Window Size", "KL Score"])
df_0.to_csv(output_dir / "evolution_0_results.csv", index=False)
plt.figure()
plt.plot(df_0["Window Size"], df_0["KL Score"], marker='o')
plt.axhline(y=0.05, color='gray', linestyle='--', label='Target KL Threshold (0.05)')
plt.title("Evolution 0: KL Divergence vs. Window Size")
plt.xlabel("Window Size")
plt.ylabel("Average KL Divergence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_dir / "evolution_0_plot.png")
all_results["evolution_0"] = df_0

# Evolution 1
results_1 = evolution_1_dropped_probes(20, 1000, 20, drop_probability=0.2)
df_1 = pd.DataFrame(results_1, columns=["Window Size", "KL Score"])
df_1.to_csv(output_dir / "evolution_1_results.csv", index=False)
plt.figure()
plt.plot(df_1["Window Size"], df_1["KL Score"], marker='o')
plt.axhline(y=0.05, color='gray', linestyle='--', label='Target KL Threshold (0.05)')
plt.title("Evolution 1: KL Divergence vs. Window Size (Packet Drop)")
plt.xlabel("Window Size")
plt.ylabel("Average KL Divergence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_dir / "evolution_1_plot.png")
all_results["evolution_1"] = df_1

# Evolution 2
results_2 = evolution_2_with_congestion(50, 1500, 50)
df_2 = pd.DataFrame(results_2, columns=["Window Size", "KL Score"])
df_2.to_csv(output_dir / "evolution_2_results.csv", index=False)
plt.figure()
plt.plot(df_2["Window Size"], df_2["KL Score"], marker='o')
plt.axhline(y=0.05, color='gray', linestyle='--', label='Target KL Threshold (0.05)')
plt.title("Evolution 2: KL Divergence vs. Window Size (Congestion + Drop)")
plt.xlabel("Window Size")
plt.ylabel("Average KL Divergence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_dir / "evolution_2_plot.png")
all_results["evolution_2"] = df_2

plt.close("all")

# Return summary paths
{
    "csv_files": [str(output_dir / f"evolution_{i}_results.csv") for i in range(3)],
    "plot_images": [str(output_dir / f"evolution_{i}_plot.png") for i in range(3)]
}