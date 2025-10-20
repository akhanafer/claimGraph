import asyncio
import os

import pandas as pd
import streamlit as st

from src.embeddings import embed_claims, get_similar_claims
from src.main import main
from src.report import (
    generate_report_summary,
    get_timeline_tone_chart,
    get_tone_chart,
    plot_claim_source_pyvis,
)

st.set_page_config(page_title="CT.ai", layout="wide")

st.sidebar.header("Input Options")


# --- Refactored Info Messages ---
INFO_MESSAGES = {
    "Provide a topic and number of hops": (
        "This option is for when you have a general topic, claim or question in mind that you want to explore. "
        "It will scrape the web and look for articles that relate to your topic. "
        "For every claim in the article, we will look for sources, "
        "then extract claims from those sources and look for their respective sources. We will repeat this "
        "up to the number of hops you configured. "
        "The higher the number the longer this will take."
    ),
    "Provide a URL": (
        "This option is for when you have a specific article or webpage in mind who's claims you want to analyze. "
        "It will scrape the provided URL and extract claims from the content. "
        "For every claim in the article, we will look for sources, "
        "then extract claims from those sources and look for their respective sources. We will repeat this "
        "up to the number of hops you configured. "
        "The higher the number the longer this will take."
    ),
}


def show_info(option, submit_button):
    if not submit_button and option in INFO_MESSAGES:
        st.info(INFO_MESSAGES[option])


# --- Sidebar Input Logic ---
option = st.sidebar.radio(
    "Choose how to start your analysis:", ("Select from an existing topic", "Provide a topic and number of hops", "Provide a URL")
)

# --- Reset session state on option change ---
if "last_option" not in st.session_state:
    st.session_state["last_option"] = option

if option != st.session_state["last_option"]:
    for key in ["summary", "tone_chart_fig", "timeline_tone_chart_fig"]:
        st.session_state[key] = None
    st.session_state["last_option"] = option

existing_topics = []
user_prompt = None
num_hops = None
url = None
submit_button = False

# --- Session State Initialization ---
for key in ["summary", "tone_chart_fig", "timeline_tone_chart_fig"]:
    if key not in st.session_state:
        st.session_state[key] = None

if option == "Select from an existing topic":
    selected_topic = st.sidebar.selectbox("Choose a topic:", existing_topics)
    submit_button = st.sidebar.button("Analyze", key="topic_btn", type="primary")
    if submit_button:
        user_prompt = selected_topic
elif option == "Provide a topic and number of hops":
    user_prompt = st.sidebar.text_area(
        "Enter your topic, claim, or question:", placeholder="Enter your research topic, claim or question...", height=100
    )
    num_hops = st.sidebar.number_input("Number of Hops", min_value=1, max_value=10, value=3, step=1)
    submit_button = st.sidebar.button("Analyze", key="prompt_btn", type="primary")
elif option == "Provide a URL":
    url = st.sidebar.text_input("URL", placeholder="https://example.com")
    user_prompt = st.sidebar.text_area(
        "What are you researching?", placeholder="Enter your research topic, claim or question...", height=100
    )
    num_hops = st.sidebar.number_input("Number of Hops", min_value=1, max_value=10, value=3, step=1)
    submit_button = st.sidebar.button("Analyze", key="url_btn", type="primary")

st.title("CT.ai")
show_info(option, submit_button)

# --- Main Analysis Logic ---
if submit_button:
    write_file_path = f'storage/{user_prompt.strip().replace(" ", "_")}_{num_hops}_hops'
    if not os.path.exists(write_file_path):
        os.makedirs(write_file_path)
    model = 'mistral:7b'
    st.subheader(f'{user_prompt} - Report Summary')
    st.session_state['tone_chart_fig'] = get_tone_chart(
        url=f'https://api.gdeltproject.org/api/v2/doc/doc?query={user_prompt} sourcelang:english&mode=ToneChart&format=json'
    )  # TODO: Make URLs clickable in the bar chart hover text
    st.session_state['timeline_tone_chart_fig'] = get_timeline_tone_chart(
        url=f'https://api.gdeltproject.org/api/v2/doc/doc?query={user_prompt} sourcelang:english&mode=TimelineTone&format=json'
    )
    if option == "Select from an existing topic" and user_prompt:
        with open(f'streamlit_reports/{user_prompt.replace(" ", "_")}.txt', 'r') as f:
            st.session_state['summary'] = f.read()
    elif option == "Provide a topic and number of hops" and user_prompt:
        claims_df = pd.read_pickle('storage/claim_embeddings.pkl')
        related_claims = get_similar_claims(user_prompt, claims_df, top_k=20)
        summary = generate_report_summary(user_prompt, related_claims[['url', 'claim', 'similarity']], model=model)
        st.session_state['summary'] = summary
    elif option == "Provide a URL" and url and user_prompt:
        articles_pd = pd.DataFrame({'source_id': [1], 'tone': [3], 'url': [url]})
        source_to_source, source_to_claim = asyncio.run(main(articles_pd, hops=num_hops, write_file_path=write_file_path))
        claim_embeddings = embed_claims(source_to_claim, write_file_path=write_file_path)
        st.session_state['claim_source_graph_fig'] = plot_claim_source_pyvis(source_to_claim, write_file_path=write_file_path)
        related_claims = get_similar_claims(user_prompt, claim_embeddings, top_k=20)
        summary = generate_report_summary(
            user_prompt, related_claims[['url', 'claim', 'similarity']], model='gpt-oss:20b', write_file_path=write_file_path
        )

        st.session_state['summary'] = summary

# --- Output Display Logic ---
col1, col2 = st.columns(2)
if st.session_state['tone_chart_fig']:
    st.subheader('Tone Charts for sources related to your prompt', divider=True)
    with col1:
        st.plotly_chart(st.session_state['tone_chart_fig'])
if st.session_state['timeline_tone_chart_fig']:
    with col2:
        st.plotly_chart(st.session_state['timeline_tone_chart_fig'])
if st.session_state.get('claim_source_graph_fig') is not None:
    with open(f'storage/{user_prompt.strip().replace(" ", "_")}_{num_hops}_hops/pyvis_graph.html', 'r') as f:
        html_content = f.read()
    st.subheader("Claim-Source Graph", divider=True)
    st.components.v1.html(html_content, height=600, scrolling=True)
if st.session_state['summary']:
    st.subheader("LLM Report", divider=True)
    st.markdown(st.session_state['summary'])
elif option == "Provide a topic and number of hops" and not user_prompt:
    st.info("👈 Please enter a research topic, claim or question in the sidebar to begin analysis")
elif option == "Provide a URL" and not url:
    st.info("👈 Please enter a URL in the sidebar to begin analysis")
else:
    st.info("👈 Select a topic or provide your own to start")
