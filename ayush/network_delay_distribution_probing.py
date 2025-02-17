import networkx as nx
import random
import statistics


"""
The following file:
- Creates a network graph with nodes and edges, where each edge has a delay distribution (mean and standard deviation).
- Computes the shortest path between a source and destination based on the mean delay of the edges.
- Simulates active probing by sending packets along the path, sampling edge delays from normal distributions.
- Outputs the average end-to-end delay and its variation (standard deviation) across multiple probes.
"""

G = nx.Graph()
nodes = [1, 2, 3, 4, 5]
G.add_nodes_from(nodes)

# Define edges with delay distribution parameters.
# Each edge gets a 'mean' delay and a 'std' deviation (jitter).
edges = [
    (1, 2, {"mean": random.uniform(5, 15), "std": 1.5}),
    (2, 3, {"mean": random.uniform(5, 15), "std": 1.5}),
    (3, 4, {"mean": random.uniform(5, 15), "std": 1.5}),
    (4, 5, {"mean": random.uniform(5, 15), "std": 1.5}),
    (1, 3, {"mean": random.uniform(10, 20), "std": 2.0}),
    (2, 4, {"mean": random.uniform(10, 20), "std": 2.0})
]

for u, v, params in edges:
    G.add_edge(u, v, **params)

# Compute the shortest path based on mean delay
# Use mean delay as the weight for each edge.
SOURCE, DESTINATION = 1, 5
path = nx.shortest_path(G, source=SOURCE, target=DESTINATION, weight='mean')
print("Selected path: ", path)


def sample_edge_delay(G, u, v):
    """
    Samples the delay for the edge between u and v from a normal distribution.
    Negative delays are clamped to 0
    """
    mean = G[u][v]['mean']
    std = G[u][v]['std']
    delay = random.gauss(mean, std)
    return max(delay, 0.0)


def probe_path(G, path, num_probes=100):
    """
    Simulates sending probe packets along the given path.
    For each probe, sample the delay at each edge and sum to get the total end-to-end delay.
    """
    delays = []

    for _ in range(num_probes):
        total_delay = 0.0

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            total_delay += sample_edge_delay(G, u, v)

        delays.append(total_delay)

    return delays


num_probes = 100
delays = probe_path(G, path, num_probes=num_probes)


# Analyze the results from the active probing
avg_delay = statistics.mean(delays)
delay_std = statistics.stdev(delays)

print(f"\nProbing Results over {num_probes} probes:")
print(f"Average end-to-end delay: {avg_delay:.2f} ms")
print(f"Delay variation (std. dev.): {delay_std:.2f} ms")
