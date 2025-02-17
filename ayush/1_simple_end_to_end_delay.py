import networkx as nx
import random
import statistics


"""
The following file:
- Simulates end-to-end network delay using a graph model with nodes and weighted edges (edge weights represent link delays).
- Calculates the shortest path between a source and destination based on delay.
- Simulates sending probe packets along the selected path, adding random noise to each hop to mimic real-world delay variations.
- (added noise follows a normal distribution)
- Computes and outputs the average end-to-end delay and its variation across multiple probes.
"""

# Step 1: Create a simple graph
G = nx.Graph()
nodes = [1, 2, 3, 4, 5]
G.add_nodes_from(nodes)

# define edges and give them a random delay (weight)
edges = [
    (1, 2, random.uniform(5, 15)),
    (2, 3, random.uniform(5, 15)),
    (3, 4, random.uniform(5, 15)),
    (4, 5, random.uniform(5, 15)),
    (1, 3, random.uniform(10, 20)),
    (2, 4, random.uniform(10, 20))
]

for u, v, delay in edges:
    G.add_edge(u, v, delay=delay)


# Step 2: Determine the shortest route from source to destination using delay as the weight
SOURCE, DESTINATION = 1, 5
path = nx.shortest_path(G, source=SOURCE, target=DESTINATION, weight ='delay')
print("Selected path: ", path)


# Step 3: Simulate sending probe packets along the path
def measure_path_delay(G, path, num_probes=100):
    delays = []

    for _ in range(num_probes):
        total_delay = 0

        # Simulate each hop delay with added random noise (10% standard deviation)
        # Noise added is sampled from a normal distribution
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            base_delay = G[u][v]['delay']
            noise = random.gauss(0, base_delay * 0.1)
            total_delay += base_delay + noise

        delays.append(total_delay)

    return delays

num_probes = 100
delays = measure_path_delay(G, path, num_probes=num_probes)


# Step 4: Analyse the results
avg_delay = statistics.mean(delays)
delay_variation = statistics.stdev(delays)

print(f"Average end-to-end delay over {num_probes} probes: {avg_delay:.2f} ms")
print(f"Delay variation (std. dev.): {delay_variation:.2f} ms")
