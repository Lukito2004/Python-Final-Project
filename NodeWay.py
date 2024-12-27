import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random
from collections import defaultdict

# Read and create the graph
df = pd.read_csv("dataset4.csv")


grouped = df.groupby("street")

pos = {key.lower(): group for key, group in grouped}
dic = {}

# Group and aggregate data
df = df.groupby(["fromSt", "toSt"]).agg({
    "Vol": "mean",
    "street": "first"  # keeps the first street name encountered
}).reset_index()

# Remove rows containing 'Dead End' in any of the specified columns
df = df[~df['fromSt'].str.contains("Dead End", na=False)]
df = df[~df['toSt'].str.contains("Dead End", na=False)]
df = df[~df['street'].str.contains("Dead End", na=False)]
df = df[~df['fromSt'].str.contains("Dead end", na=False)]
df = df[~df['toSt'].str.contains("Dead end", na=False)]
df = df[~df['street'].str.contains("Dead end", na=False)]


# Create a directed graph
G = nx.Graph()

# Add unique nodes (both 'fromSt' and 'toSt')
unique_nodes = set(df['fromSt'].unique()) | set(df['toSt'].unique())
G.add_nodes_from(unique_nodes)

# Add edges to the graph
for _, row in df.iterrows():
    G.add_edge(row['fromSt'].lower(), row['toSt'].lower(), street=row['street'].lower(), volume=row['Vol'])

class PathLearner:
    def __init__(self, graph):
        self.graph = graph

    def find_shortest_path(self, start_node, end_node):
        """
        Find the shortest path using Dijkstra's algorithm.
        """
        try:

            path = nx.shortest_path(self.graph, start_node, end_node, weight='weight')

            return path

        except nx.NetworkXNoPath:
            print(f"No path found between {start_node} and {end_node}.")
            return None

        except nx.NodeNotFound as e:
            print(e)
            return None


    def visualize_path(self, path, title="Shortest Path"):
        """Visualize the path in the graph"""
        plt.figure(figsize=(15, 10))

        # Draw the full graph
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos,
                node_size=1,
                with_labels=False,
                edge_color='lightgray',
                width=0.5,
                alpha=0.3)

        # Highlight the path
        if path:
            path_edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_nodes(self.graph, pos,
                                   nodelist=path,
                                   node_color='r',
                                   node_size=50)
            nx.draw_networkx_edges(self.graph, pos,
                                   edgelist=path_edges,
                                   edge_color='r',
                                   width=2)

            # Add labels for start and end nodes
            labels = {path[0]: 'Start', path[-1]: 'End'}
            nx.draw_networkx_labels(self.graph, pos,
                                    labels,
                                    font_size=12)

        plt.title(title)
        plt.show()


# Create PathLearner instance
learner = PathLearner(G)


# Example usage
def find_and_visualize_path(start_node, end_node):
    """Find, learn from, and visualize a path"""

    path = learner.find_shortest_path(start_node, end_node)
    path_graph = nx.Graph()
    edge_labels = {}

    # Add edges and create edge labels
    for i in range(len(path) - 1):
        path_graph.add_edge(path[i], path[i + 1])
        edge_labels[(path[i], path[i + 1])] = G[path[i]][path[i + 1]]['street']

    # Draw the path
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(path_graph, k=2)  # Increased k for more spacing

    # Draw edges
    nx.draw_networkx_edges(path_graph, pos, edge_color='red', width=2)

    # Draw nodes
    nx.draw_networkx_nodes(path_graph, pos, node_color='lightblue',
                           node_size=500)

    # Add node labels
    node_labels = {node: node for node in path_graph.nodes()}
    nx.draw_networkx_labels(path_graph, pos, node_labels)

    # Add edge labels
    nx.draw_networkx_edge_labels(path_graph, pos, edge_labels=edge_labels)

    plt.title(f"Shortest Path from {start_node} to {end_node}")
    plt.axis('off')
    plt.show()

    # Print path information with street names
    print("Path with street names:")
    for i in range(len(path) - 1):
        street_name = G[path[i]][path[i + 1]]['street']
        print(G[path[i]][path[i + 1]]['volume'])
        print(f"{path[i]} -> {street_name} -> {path[i + 1]}")

    path_length = sum(G[path[i]][path[i + 1]]['volume'] for i in range(len(path) - 1))
    print(f"\nTotal distance (weight): {path_length}")


# Get list of nodes
nodes = list(G.nodes())

# Example: Find path between two random nodes
if len(nodes) >= 2:
    print("For example: Rhinelander Avenue, 6 Avenue Line")
    start = input("Starting Street name: ").lower()
    end = input("Destination Street name: ").lower()
    find_and_visualize_path(start, end)

# Statistics about the graph
print("\nGraph Statistics:")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")