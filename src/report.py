import logging
from collections import Counter
from typing import Optional

import matplotlib
import pandas as pd
import plotly.graph_objects as go
import requests
from ollama import chat
from pyvis.network import Network

from src.utils.utils import log_event

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

SYSTEM_PROMPT = '''
You are an agent responsible for providing detailed information about a research topic,
claim, or question based on a user input. Your task is to generate a comprehensive report
summary that includes relevant claims, sources, and a synthesized overview of the topic.

You will be given a comma separated list of claims and their associated sources and relevancy score.
Each claim may have multiple sources. Below is an example of the input format:

url,claim,similarity
https://dummy-url.com, some dummy claim, 0.9657629800072585
https://some-other-url.com, some other dummy claim, 0.4778959003888945
https://another-dummy-url.com, some other dummy claim, 0.7598723400872585

Format your report in the following way, with each section having its own heading in markdown:
1. Title: A concise title that encapsulates the main theme of the report.
2. Introduction: A brief introduction that outlines the scope and purpose of the report,
along with the strongest arguments and claims.
3. Body: A detailed body that delves into the various claims, providing context,
analysis, and supporting evidence from the sources.
4. Conclusion: A summary that encapsulates the key findings and insights derived from the report.
5. Sources: A comprehensive numbered list of all sources referenced in the report.

Any claim made in the report that comes from the Sources section should point to the corresponding numbered source in the list
like so [1], [2], etc.
'''

PROMPT = '''
Generate a report summary based on the following claims and sources:

Claims and Sources:
{claims_data}

User Prompt:
{user_prompt}

'''


def generate_report_summary(
    user_prompt: str, claims_data: pd.DataFrame, hops: int = 1, model: str = 'gpt-oss:20b', write_file_path: Optional[str] = None
) -> str:
    augmented_user_prompt = PROMPT.format(claims_data=claims_data.to_csv(index=False), user_prompt=user_prompt)
    log_event(logger, logging.INFO, 'generate_report_summary', system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt, model=model)

    response = chat(
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': augmented_user_prompt},
        ],
        model=model,
    )
    summary = response.message.content
    with open(f'{write_file_path}/report.txt', 'w') as f:
        f.write(summary)
    return summary


def plot_claim_source_pyvis(df, write_file_path: Optional[str] = None):
    net = Network(height="600px", width="100%", directed=True, notebook=False)

    # Count sources per claim for sizing
    claim_source_counts = Counter(df['claim_id'])

    # Normalize tone values for gradient
    print(df.columns)
    if 'tone' in df.columns:
        tones = df['tone'].astype(float)
        min_tone, max_tone = tones.min(), tones.max()
        # Avoid division by zero
        if min_tone == max_tone:
            min_tone, max_tone = min_tone - 1, max_tone + 1

        def tone_to_color(tone):
            # Map tone to [0,1] for colormap
            norm = (tone - min_tone) / (max_tone - min_tone)
            # Use matplotlib's RdYlGn colormap
            rgb = matplotlib.cm.RdYlGn(norm)[:3]
            return matplotlib.colors.rgb2hex(rgb)

    else:
        # Fallback: all orange
        def tone_to_color(tone):
            return "orange"

    # Add nodes and edges
    for _, row in df.iterrows():
        claim_id = f"claim_{row['claim_id']}"
        source_id = f"src_{row['domain']}"
        # Source node: gradient color by tone, label only on hover
        tone_val = float(row['tone']) if 'tone' in row and not pd.isnull(row['tone']) else 0
        source_color = tone_to_color(tone_val)
        net.add_node(
            source_id, label="", color=source_color, title=f"Source: {row['domain']}<br>Tone: {tone_val}"  # No label shown
        )
        # Claim node: blue, label only on hover, size by number of sources
        net.add_node(
            claim_id,
            label="",  # No label shown
            color="royalblue",
            size=12 + 3 * claim_source_counts[row['claim_id']],  # Reduced scaling for claim node size
            title=f"Claim: {row['claim']}<br>Sources: {claim_source_counts[row['claim_id']]}",
        )
        net.add_edge(source_id, claim_id, title=row['url'])

    # Add legend nodes (not connected, just for legend)
    # Source legend: show gradient extremes
    net.add_node(
        "legend_source_min",
        label="Source (Low Tone)",
        color=tone_to_color(min_tone),
        shape="dot",
        x=-200,
        y=-200,
        physics=False,
        title=f"Source node (tone={min_tone:.2f}, red)",
    )
    net.add_node(
        "legend_source_max",
        label="Source (High Tone)",
        color=tone_to_color(max_tone),
        shape="dot",
        x=-200,
        y=-250,
        physics=False,
        title=f"Source node (tone={max_tone:.2f}, green)",
    )
    net.add_node(
        "legend_claim", label="Claim", color="royalblue", shape="dot", x=-200, y=-300, physics=False, title="Claim node (blue)"
    )

    net.show_buttons(filter_=['physics'])
    net.save_graph(f'{write_file_path}/pyvis_graph.html')
    return write_file_path


def get_tone_chart(url: str):
    tone_chart_json = requests.get(url=url).json()
    bins = []
    counts = []
    hover_texts = []
    for item in tone_chart_json['tonechart']:
        bins.append(item['bin'])
        counts.append(item['count'])

        if item['count'] > 0 and item['toparts']:
            articles_html = []
            for article in item['toparts']:
                link = f"<a href='{article['url']}' target='_blank'>{article['title']}</a>"
                articles_html.append(link)
            hover_text = f"Tone Bin: {item['bin']}<br>Count: {item['count']}<br><br>Articles:<br>" + "<br><br>".join(
                articles_html
            )
        else:
            hover_text = f"Tone Bin: {item['bin']}<br>Count: {item['count']}<br><br>No articles"

        hover_texts.append(hover_text)

    fig = go.Figure(data=[go.Bar(x=bins, y=counts, hovertemplate='%{hovertext}<extra></extra>', hovertext=hover_texts)])

    # Update layout
    fig.update_layout(
        title='Article Count by Tone Bin',
        xaxis_title='Tone Bin',
        yaxis_title='Count',
        hovermode='closest',
        height=600,
        width=1000,
        template='plotly_white',
    )
    return fig


def get_timeline_tone_chart(url: str):
    timeline_tone_chart_json = requests.get(url=url).json()
    # Extract timeline
    timeline = timeline_tone_chart_json["timeline"][0]["data"]

    # Build DataFrame
    df = pd.DataFrame(timeline)
    df["date"] = pd.to_datetime(df["date"])

    # Plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["value"],
            mode="lines+markers",
            name=timeline_tone_chart_json["timeline"][0]["series"],
            line=dict(color="royalblue", width=2),
            marker=dict(size=5),
        )
    )

    fig.update_layout(
        title=f"Article Tone Over Time — {timeline_tone_chart_json['query_details']['title']}",
        xaxis_title="Date",
        yaxis_title="Average Tone",
        template="plotly_white",
        height=500,
        width=1000,
    )

    return fig
