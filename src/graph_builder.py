import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def claim_graph_builder(adjacency_matrix):
    G = nx.from_pandas_adjacency(adjacency_matrix)
    return G


if __name__ == '__main__':
    adjacency_matrix = pd.read_csv('adjacency_matrix.csv', index_col=0)
    G = claim_graph_builder(adjacency_matrix)
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # or nx.kamada_kawai_layout(G), nx.circular_layout(G), etc.
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.show()
