import asyncio
import os

import pandas as pd
import streamlit as st

from src.claim_extraction import main
from src.embeddings import embed_claims, embed_user_query, get_similar_claims
from src.report import generate_report_summary

# Configure the page
st.set_page_config(page_title="CT.ai", layout="wide")


# Sidebar for inputs
st.sidebar.header("Input Options")

# Option selector
option = st.sidebar.radio(
    "Choose how to start your analysis:", ("Select from an existing topic", "Provide a topic and number of hops", "Provide a URL")
)

# Placeholder list of existing topics
existing_topics = ["American Sentiment on Israel-Palestine war"]

# Input fields based on option
user_prompt = None
num_hops = None
url = None

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

# Main content area
st.title("CT.ai")

# Show info message for topic + hops option
if option == "Provide a topic and number of hops":
    st.info(
        "This option is for when you have a general topic, claim or question in mind that you want to explore. "
        "It will scrape the web and look for articles that relate to your topic. "
        "For every claim in the article, we will look for sources, "
        "then extract claims from those sources and look for their respective sources. We will repeat this"
        " up to the number of hops you configured. "
        "The higher the number the longer this will take."
    )

if option == "Provide a URL":
    st.info(
        "This option is for when you have a specific article or webpage in mind who's claims you want to analyze. "
        "It will scrape the provided URL and extract claims from the content. "
        "For every claim in the article, we will look for sources, "
        "then extract claims from those sources and look for their respective sources. We will repeat this"
        " up to the number of hops you configured. "
        "The higher the number the longer this will take."
    )

# Display results based on inputs using session_state
if 'summary' not in st.session_state:
    st.session_state['summary'] = None

if submit_button:
    model = 'llama3.1:8b'
    st.subheader("Report Summary")
    if option == "Select from an existing topic" and user_prompt:
        if os.path.exists(f'streamlit_reports/{user_prompt.replace(" ", "_")}.txt'):
            with open(f'streamlit_reports/{user_prompt.replace(" ", "_")}.txt', 'r') as f:
                report = f.read()
                st.session_state['summary'] = report
        else:
            claims_df = pd.read_pickle('storage/claim_embeddings.pkl')
            related_claims = get_similar_claims(user_prompt, claims_df, top_k=20)
            summary = generate_report_summary(user_prompt, related_claims[['url', 'claim', 'similarity']], model=model)
            with open(f'streamlit_reports/{user_prompt.replace(" ", "_")}.txt', 'w') as f:
                f.write(summary)
            st.session_state['summary'] = summary
    elif option == "Provide a topic and number of hops" and user_prompt:
        claims_df = pd.read_pickle('storage/claim_embeddings.pkl')
        related_claims = get_similar_claims(user_prompt, claims_df, top_k=20)
        summary = generate_report_summary(user_prompt, related_claims[['url', 'claim', 'similarity']], model=model)
        st.session_state['summary'] = summary
    elif option == "Provide a URL" and url and user_prompt:
        articles_pd = pd.DataFrame(
            {
                'source_id': [1],
                'url': [url],
            }
        )
        source_to_source, source_to_claim = asyncio.run(main(articles_pd, hops=num_hops, prompt=user_prompt))
        claim_embeddings = embed_claims(source_to_claim)
        query_embedding = embed_user_query(user_prompt)
        related_claims = get_similar_claims(query_embedding, claim_embeddings, top_k=10)
        summary = generate_report_summary(user_prompt, related_claims[['url', 'claim', 'similarity']], model=model)
        with open(f'streamlit_reports/{user_prompt.replace(" ", "_")}.txt', 'w') as f:
            f.write(summary)
        st.session_state['summary'] = summary

if st.session_state['summary']:
    st.markdown(st.session_state['summary'])
elif option == "Provide a topic and number of hops" and not user_prompt:
    st.info("👈 Please enter a research topic, claim or question in the sidebar to begin analysis")
elif option == "Provide a URL" and not url:
    st.info("👈 Please enter a URL in the sidebar to begin analysis")
else:
    st.info("👈 Select a topic or provide your own to start")
