import ast

import networkx as nx
import pandas as pd
import plotly.graph_objects as go


def claim_graph_builder(edge_list):
    G = nx.from_pandas_edgelist(
        edge_list,
        source='source_id_source',
        target='source_id_target',
        edge_attr='claim_info',
    )
    nx.set_node_attributes(G, pd.Series(edge_list.url_target.values, index=edge_list.source_id_target).to_dict(), 'url_target')
    return G


if __name__ == '__main__':
    edge_list = pd.read_csv("edge_list.csv")
    G = claim_graph_builder(edge_list)

    # Get positions for nodes
    pos = nx.spring_layout(G, k=1.5, seed=42)

    # Build edge traces (just the lines)
    edge_x = []
    edge_y = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=1, color="#888"), mode="lines", hoverinfo="none"  # no hover on the line itself
    )

    # Build invisible midpoint markers with hover text for edges
    mid_x = []
    mid_y = []
    mid_text = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        claim_id, claim_text = ast.literal_eval(data["claim_info"])
        mid_text.append(f"Claim ID: {claim_id}<br>Claim: {claim_text[:200]}")

    edge_hover_trace = go.Scatter(
        x=mid_x,
        y=mid_y,
        mode="markers",
        marker=dict(size=10, color="rgba(0,0,0,0)"),  # invisible markers
        text=mid_text,
        hoverinfo="text",
    )

    # Build node traces
    node_x = []
    node_y = []
    node_labels = []  # labels drawn on plot
    node_hover = []  # attributes shown on hover

    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # label shown on graph
        node_labels.append(str(node))

        # attributes shown on hover
        url = data.get("url_target", "N/A")
        node_hover.append(f"Node: {node}<br>URL: {url}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_labels,  # visible labels
        hovertext=node_hover,  # hover attributes
        hoverinfo="text",
        marker=dict(size=20, color="lightblue", line=dict(width=2, color="darkblue")),
    )

    # Create figure
    fig = go.Figure(
        data=[edge_trace, edge_hover_trace, node_trace],
        layout=go.Layout(
            title=dict(text="Claim Graph Visualization", font=dict(size=20)),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    fig.show()
