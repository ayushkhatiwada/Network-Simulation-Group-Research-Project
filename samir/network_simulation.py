import networkx as nx
import random
import matplotlib.pyplot as plt
import time

def create_network(num_nodes):
    G = nx.connected_watts_strogatz_graph(num_nodes, k=4, p=0.5)
    # Assign random delays to each edge
    for u, v in G.edges():
        G[u][v]['delay'] = random.uniform(0.1, 1.0) 
    return G

# Tweak delay distribution on a timescale
def tweak_delays(G, time_scale):
    for u, v in G.edges():
        if time_scale % 2 == 0:  
            G[u][v]['delay'] += random.uniform(-0.05, 0.05) 
            # Ensure delay doesn't go below 0.1
            if G[u][v]['delay'] < 0.1:
                G[u][v]['delay'] = 0.1

def route_packet(G, source, destination):
    try:
        # Find the shortest path based on delay
        path = nx.shortest_path(G, source, destination, weight='delay')  
        total_delay = sum(G[u][v]['delay'] for u, v in zip(path[:-1], path[1:]))  # Sum of edge delays
        print(f"Packet from {source} to {destination}: Path = {path}, Total Delay = {total_delay:.2f}s")
        return path  
    except nx.NetworkXNoPath:
        print(f"No path from {source} to {destination}")
        return []

def simulate_network(num_nodes, total_time):
    G = create_network(num_nodes)  

    plt.ion()  
    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(G) 

    for time_scale in range(total_time):
        tweak_delays(G, time_scale)  

        source, target = random.sample(list(G.nodes()), 2)

        packet_path = route_packet(G, source, target)

        ax.clear()

        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=12, ax=ax)
        
        labels = {(u, v): f"{G[u][v]['delay']:.2f}s" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red', ax=ax)

        if packet_path:
            path_edges = [(packet_path[i], packet_path[i+1]) for i in range(len(packet_path)-1)]
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='orange', width=2, ax=ax)

            for i in range(len(packet_path) - 1):
                start_node = packet_path[i]
                end_node = packet_path[i+1]
                
                ax.annotate('', xy=pos[end_node], xytext=pos[start_node],
                            arrowprops=dict(facecolor='green', edgecolor='green', arrowstyle='->', lw=2))

        ax.set_title(f"Network with Delays (Time Step: {time_scale})")

        plt.draw()
        plt.pause(1) 

    plt.ioff()
    plt.show()

# Set a random seed for reproducibility
random.seed(42)

simulate_network(5, 10)