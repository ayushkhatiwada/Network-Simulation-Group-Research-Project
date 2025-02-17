import networkx as nx
import random
import statistics
import math


# TODO: Shortest (fastest) path is currently based on the initial mean weights of the edges
# shortest (fastest) path should possibly be changed every time step based on the updated sample weights
# Ask Stefano how packets choose their paths from source to destination


# ----------------------------------------------
# 1: Create Graph
# ----------------------------------------------
G = nx.Graph()
nodes = [1, 2, 3, 4, 5]
G.add_nodes_from(nodes)

# Define edges with base_mean & base_standard_deviation delays 
edges = [
    (1, 2, {"base_mean": random.uniform(5, 15), "base_std": 1.5}),
    (2, 3, {"base_mean": random.uniform(5, 15), "base_std": 1.5}),
    (3, 4, {"base_mean": random.uniform(5, 15), "base_std": 1.5}),
    (4, 5, {"base_mean": random.uniform(5, 15), "base_std": 1.5}),
    (1, 3, {"base_mean": random.uniform(10, 20), "base_std": 2.0}),
    (2, 4, {"base_mean": random.uniform(10, 20), "base_std": 2.0})
]

# When adding an edge, set its initial 'mean' and 'std' to the base values
for u, v, params in edges:
    params["mean"] = params["base_mean"]
    params["std"] = params["base_std"]
    G.add_edge(u, v, **params)


# ----------------------------------------------
# 2: Define function to update edge delays over time
# ----------------------------------------------
def update_edge_delays(G, current_time, period=10, amplitude_mean=0.2, amplitude_std=0.2):
    """
    Update each edge's 'mean' and 'std' based on a sine modulation.

    - period: period of the sine wave modulation. Defines how frequently the delays will change
    - amplitude_mean: controls how much the mean delay can change relative to its base value (default is 20%)
    - amplitude_std: controls how much the std delay can change relative to its base value (default is 20%)
    """

    # # data=True grabs all data associated with each edge and puts it into a dictionary
    for u, v, data in G.edges(data=True):
        # Calculate modulation factor using a sine wave.
        modulation = math.sin(2 * math.pi * current_time / period)
        data["mean"] = data["base_mean"] * (1 + amplitude_mean * modulation)
        data["std"] = data["base_std"] * (1 + amplitude_std * modulation)

        # Ensure non-negative values.
        data["mean"] = max(data["mean"], 0.0)
        data["std"] = max(data["std"], 0.0)


# ----------------------------------------------
# 3: Define functions to sample delays and probe a path
# ----------------------------------------------
def sample_edge_delay(G, u, v):
    """
    Sample delay for edge (u, v) using its current mean and standard deviation
    Negative delays are clamped to 0
    """
    mean = G[u][v]['mean']
    std = G[u][v]['std']
    delay = random.gauss(mean, std)
    return max(delay, 0.0)

def probe_path(G, path, num_probes=100):
    """
    Simulates sending probe packets along the given path.
    Sum the delays of each edge (sampled from the current distributions) to get the end-to-end delay.
    """
    delays = []

    for _ in range(num_probes):
        total_delay = 0.0

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            total_delay += sample_edge_delay(G, u, v)

        delays.append(total_delay)

    return delays


# ----------------------------------------------
# 4: Simulate the dynamic network over time
# ----------------------------------------------
SOURCE, DESTINATION = 1, 5
TIME_STEPS = 10  # Number of time steps in the simulation
NUM_PROBES = 100

for t in range(TIME_STEPS):
    # Update edge delay distributions for the current time step.
    update_edge_delays(G, current_time=t, period=TIME_STEPS, amplitude_mean=0.9, amplitude_std=0.9)
    
    # Compute the shortest path based on the current 'mean' delays.
    path = nx.shortest_path(G, source=SOURCE, target=DESTINATION, weight='mean')
    
    # Simulate probing along this path.
    delays = probe_path(G, path, num_probes=NUM_PROBES)
    avg_delay = statistics.mean(delays)
    delay_std = statistics.stdev(delays)
    
    print(f"Time Step {t}:")
    print(f"  Selected path: {path}")
    print(f"  Average delay: {avg_delay:.2f} ms, Std Dev: {delay_std:.2f} ms\n")
