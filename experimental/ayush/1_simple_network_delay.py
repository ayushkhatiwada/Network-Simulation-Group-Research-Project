import networkx as nx
import random
import statistics


"""
Simulate network delay & probing.
- Create a simple graph that represents a network.
- Weights of graph represent the delay from node to node.
- Find the shortest path from a source to a destination based on delays.
- Simulate 100 probes and calculate average delay and variation.
"""

# Step 1: Create simple graph
G = nx.Graph()
nodes = [1, 2, 3, 4, 5]
G.add_nodes_from(nodes)


# Step 2: Add edges with random weights representing delay (ms)
edges = [
    (1, 2, random.uniform(1, 20)),
    (2, 3, random.uniform(1, 20)),
    (3, 4, random.uniform(1, 20)),
    (4, 5, random.uniform(1, 20)),
]
for u, v, delay in edges:
    G.add_edge(u, v, delay=delay)


# Step 3: Find the shortest path based on delay
# Only 1 path exists for now, but this can change in the future
SOURCE, DESTINATION = 1, 5
path = nx.shortest_path(G, source=SOURCE, target=DESTINATION, weight='delay')
print("Selected path:", path)


# Step 4: Simulate probes and calculate total delay
def measure_path_delay(G, path, num_probes):    
    total_delay = 0
    
    # Loop through consecutive nodes in the path and sum delays
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        total_delay += G[u][v]['delay']
    
    # Return the total delay repeated for all probes
    return [total_delay] * num_probes

num_probes = 100
delays = measure_path_delay(G, path, num_probes)


# Step 5: Calculate and print the results
avg_delay = statistics.mean(delays)
delay_variation = statistics.stdev(delays)

print(f"Avg delay over {num_probes} probes: {avg_delay:.2f} ms")
print(f"Delay variation (std. dev.): {delay_variation:.2f} ms")
