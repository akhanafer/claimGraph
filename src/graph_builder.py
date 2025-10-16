import networkx as nx
import pandas as pd


def claim_graph_builder(edge_list):
    G = nx.from_pandas_edgelist(edge_list, source='domain', target='claim', edge_attr='relation', create_using=nx.DiGraph)

    domains = list(edge_list['domain'].unique())
    claims = list(edge_list['claim'].unique())

    for domain in domains:
        G.nodes[domain]['type'] = 'domain'
        G.nodes[domain]['color'] = 'blue'

    for claim in claims:
        G.nodes[claim]['type'] = 'claim'
        G.nodes[claim]['color'] = 'red'

    nx.write_graphml(G, "source_to_claim_graph.graphml")


if __name__ == '__main__':
    # source_to_source_edge_list = pd.read_csv("edge_list_2_hop.csv")
    source_to_claim_edge_list = pd.read_csv("storage/source_to_claim_edge_list_2_hop.csv")
    G = claim_graph_builder(source_to_claim_edge_list)
