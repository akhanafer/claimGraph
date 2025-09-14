import ast

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
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


def create_spaced_layout(G, edge_list, spacing_multiplier=8.0):
    """Create a layout with visual separation between source groups in a perfect circle"""

    # Get unique source nodes
    source_nodes = edge_list['source_id_source'].unique()

    # Group nodes by their primary source (the one they appear most with)
    node_to_primary_source = {}

    # First, assign each source node to itself
    for source in source_nodes:
        if source in G.nodes():
            node_to_primary_source[source] = source

    # Then assign target nodes to their most frequent source
    for node in G.nodes():
        if node not in node_to_primary_source:
            # Find which sources this node connects to
            connected_sources = edge_list[(edge_list['source_id_target'] == node) | (edge_list['source_id_source'] == node)][
                'source_id_source'
            ].values

            if len(connected_sources) > 0:
                # Use the first source (or most frequent if you want to get fancy)
                node_to_primary_source[node] = connected_sources[0]
            else:
                # Fallback - shouldn't happen with proper data
                node_to_primary_source[node] = source_nodes[0]

    # Calculate center position for each source group in a perfect circle
    group_positions = {}
    angle_step = 2 * np.pi / len(source_nodes)

    for i, source in enumerate(source_nodes):
        angle = i * angle_step
        center_x = spacing_multiplier * np.cos(angle)
        center_y = spacing_multiplier * np.sin(angle)
        group_positions[source] = (center_x, center_y)

    # Position nodes within each group
    adjusted_pos = {}

    for source in source_nodes:
        # Get all nodes in this group
        group_nodes = [
            node for node, primary_source in node_to_primary_source.items() if primary_source == source and node in G.nodes()
        ]

        if not group_nodes:
            continue

        group_center_x, group_center_y = group_positions[source]

        # Place the source node at the center of the group
        if source in G.nodes():
            adjusted_pos[source] = (group_center_x, group_center_y)

        # Get target nodes (non-source nodes in this group)
        target_nodes = [node for node in group_nodes if node != source]

        if target_nodes:
            # Arrange target nodes in a circle around the source node
            target_radius = 2.0  # Radius around the source node
            target_angle_step = 2 * np.pi / len(target_nodes) if len(target_nodes) > 1 else 0

            for j, target in enumerate(target_nodes):
                target_angle = j * target_angle_step
                target_x = group_center_x + target_radius * np.cos(target_angle)
                target_y = group_center_y + target_radius * np.sin(target_angle)
                adjusted_pos[target] = (target_x, target_y)

    return adjusted_pos, node_to_primary_source


def get_node_colors(G, edge_list, node_to_primary_source):
    """Assign colors to nodes based on their primary source groups"""

    # Get unique source nodes
    source_nodes = edge_list['source_id_source'].unique()

    # Create color palette
    colors = px.colors.qualitative.Set3
    if len(source_nodes) > len(colors):
        colors = colors * (len(source_nodes) // len(colors) + 1)

    # Create color mapping for sources
    source_color_map = {source: colors[i] for i, source in enumerate(source_nodes)}

    # Assign colors to nodes based on their primary source
    node_colors = []
    node_color_map = {}

    for node in G.nodes():
        primary_source = node_to_primary_source[node]
        color = source_color_map[primary_source]
        node_colors.append(color)
        node_color_map[node] = color

    return node_colors, node_color_map, source_color_map


if __name__ == '__main__':
    edge_list = pd.read_csv("edge_list_2_hop.csv")
    G = claim_graph_builder(edge_list)

    # Get positions for nodes with group separation
    pos, node_to_primary_source = create_spaced_layout(G, edge_list, spacing_multiplier=10.0)

    # Get node colors based on primary source groups
    node_colors, node_color_map, source_color_map = get_node_colors(G, edge_list, node_to_primary_source)

    # Build edge traces with different styles for inter-group vs intra-group edges
    intra_group_edge_x = []
    intra_group_edge_y = []
    inter_group_edge_x = []
    inter_group_edge_y = []

    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        # Check if this edge connects nodes from different groups
        if node_to_primary_source[u] == node_to_primary_source[v]:
            # Same group - normal edge
            intra_group_edge_x += [x0, x1, None]
            intra_group_edge_y += [y0, y1, None]
        else:
            # Different groups - inter-group edge
            inter_group_edge_x += [x0, x1, None]
            inter_group_edge_y += [y0, y1, None]

    # Create traces for different edge types
    intra_group_edge_trace = go.Scatter(
        x=intra_group_edge_x,
        y=intra_group_edge_y,
        line=dict(width=2, color="#888888"),
        mode="lines",
        hoverinfo="none",
        name="Intra-group edges",
        showlegend=False,
    )

    inter_group_edge_trace = go.Scatter(
        x=inter_group_edge_x,
        y=inter_group_edge_y,
        line=dict(width=3, color="#444444", dash="dash"),
        mode="lines",
        hoverinfo="none",
        name="Inter-group edges",
        showlegend=False,
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

        # Add group information to hover text
        u_group = node_to_primary_source[u]
        v_group = node_to_primary_source[v]
        edge_type = "Intra-group" if u_group == v_group else "Inter-group"

        mid_text.append(
            f"Edge Type: {edge_type}<br>"
            f"From: {u} (Group: {u_group})<br>"
            f"To: {v} (Group: {v_group})<br>"
            f"Claim ID: {claim_id}<br>"
            f"Claim: {claim_text[:200]}"
        )

    edge_hover_trace = go.Scatter(
        x=mid_x,
        y=mid_y,
        mode="markers",
        marker=dict(size=15, color="rgba(0,0,0,0)"),
        text=mid_text,
        hoverinfo="text",
        name="Edge Info",
        showlegend=False,
    )

    # Build node traces - create one trace per source group for legend
    traces = [intra_group_edge_trace, inter_group_edge_trace, edge_hover_trace]

    source_nodes = edge_list['source_id_source'].unique()

    for source in source_nodes:
        # Get all nodes that belong to this source group
        group_nodes = [
            node for node, primary_source in node_to_primary_source.items() if primary_source == source and node in G.nodes()
        ]

        if not group_nodes:
            continue

        # Get positions and data for nodes in this group
        group_x = []
        group_y = []
        group_labels = []
        group_hover = []

        for node in group_nodes:
            x, y = pos[node]
            group_x.append(x)
            group_y.append(y)
            group_labels.append(str(node))

            # Get node data
            node_data = G.nodes[node]
            url = node_data.get("url_target", "N/A")

            # Determine if this node is a source node
            is_source = node in source_nodes
            node_type = "Source" if is_source else "Target"

            group_hover.append(f"Node: {node}<br>" f"Type: {node_type}<br>" f"Primary Group: {source}<br>" f"URL: {url}")

        # Create trace for this source group
        node_trace = go.Scatter(
            x=group_x,
            y=group_y,
            mode="markers+text",
            text=group_labels,
            hovertext=group_hover,
            hoverinfo="text",
            marker=dict(
                size=25,  # All nodes same size, all circles
                color=source_color_map[source],
                line=dict(width=2, color="white"),
                symbol='circle',  # All nodes are circles
            ),
            textposition="middle center",
            textfont=dict(size=10, color="white", family="Arial Black"),
            name=f"Group: {source}",
            legendgroup=f"group_{source}",
        )
        traces.append(node_trace)

    # Create figure
    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(text="Claim Graph - Grouped by Source with Visual Separation", font=dict(size=24), x=0.5),
            showlegend=True,
            legend=dict(x=1.02, y=1, bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(0,0,0,0.3)", borderwidth=1),
            hovermode="closest",
            margin=dict(b=20, l=5, r=180, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            # plot_bgcolor='white',
            # width=1400,
            # height=900
        ),
    )

    # Add annotations to explain the visual encoding
    fig.add_annotation(
        text="● All nodes are circles<br>"
        "— Solid lines: same group<br>"
        "- - Dashed lines: different groups<br>"
        "Same color = same source group",
        xref="paper",
        yref="paper",
        x=1.02,
        y=0.02,
        showarrow=False,
        font=dict(size=11),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(0,0,0,0.3)",
        borderwidth=1,
    )

    fig.show()

    # Print summary information
    print("\nGraph Summary:")
    print(f"Total nodes: {len(G.nodes())}")
    print(f"Total edges: {len(G.edges())}")
    print(f"Source groups: {len(source_nodes)}")

    # Print group assignments
    print("\nGroup assignments:")
    for source in source_nodes:
        group_members = [
            node for node, primary_source in node_to_primary_source.items() if primary_source == source and node in G.nodes()
        ]
        print(f"  Group '{source}': {group_members}")

    # Count inter-group vs intra-group edges
    intra_count = len(intra_group_edge_x) // 3  # Each edge has 3 elements (x0, x1, None)
    inter_count = len(inter_group_edge_x) // 3
    print("\nEdge types:")
    print(f"  Intra-group edges: {intra_count}")
    print(f"  Inter-group edges: {inter_count}")
