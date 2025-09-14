import logging
from typing import Callable, List, Optional, Tuple
from uuid import uuid4

import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ollama import chat
from pydantic import BaseModel, ValidationError

from src.gdelt_api_client import full_text_search
from src.prompts import (
    CLAIM_EXTRACTION_PROMPT,
    CLAIM_EXTRACTION_SYSTEM_PROMPT,
    FORMAT_QUERY_RETRY_PROMPT,
    STRUCTURED_OUTPUT_PROMPT,
)
from src.pydantic_models.gdelt_api_params import (
    FullTextSearchParams,
    FullTextSearchQueryCommands,
)
from src.utils import log_event

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class TextPassage(BaseModel):
    subject: str
    claims: List[str]


def apply_and_explode(df: pd.DataFrame, column: str, func: Callable, new_columns: List[str]) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy['info_tuple'] = df_copy[column].apply(func)
    df_exploded = df_copy.explode('info_tuple').reset_index(drop=True)
    if df_exploded.empty:  # If the exploded DataFrame is empty e.g., no HTML fetched
        return pd.DataFrame(columns=df.columns.tolist() + new_columns)
    df_exploded[new_columns] = pd.DataFrame(df_exploded['info_tuple'].tolist(), index=df_exploded.index)
    return df_exploded.drop(columns=['info_tuple'])


def get_text_from_source(content_source_df: pd.DataFrame, keep_html_col: bool = False) -> pd.DataFrame:
    content_text_df = content_source_df.copy()
    content_text_df = content_text_df.dropna(subset=['url'])  # Happens when no sources found
    content_text_df['raw_html'] = content_text_df['url'].apply(_fetch_source_html)  # TODO: How to deal with failed HTML requests
    content_text_df['content_text'] = content_text_df['raw_html'].apply(_extract_text_from_html)
    if not keep_html_col:
        content_text_df = content_text_df.drop(columns=['raw_html'])
    return content_text_df


def chunk_text(text_df: pd.DataFrame, chunk_size: int, chunk_overlap: int) -> pd.DataFrame:
    log_event(logger, logging.INFO, 'Chunking text', chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def recursive_text_split(text: str) -> List[str]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return [(uuid4().hex, chunk) for chunk in text_splitter.split_text(text)]

    text_df_copy_exploded = apply_and_explode(text_df, 'content_text', recursive_text_split, ['chunk_id', 'chunk'])

    return text_df_copy_exploded


def get_chunk_claims(
    chunk_df: pd.DataFrame,
    claim_model: str = 'gpt-oss:20b',
    structured_output_model: Optional[str] = None,  # Needed for models that don't support structured output
) -> TextPassage:
    log_event(
        logger,
        logging.INFO,
        'Extracting claims from chunks',
        claim_model=claim_model,
        structured_output_model=structured_output_model,
    )

    def extract_claims_from_chunk(chunk: str) -> List[Tuple[str, str]]:
        response = chat(
            messages=[
                {'role': 'system', 'content': CLAIM_EXTRACTION_SYSTEM_PROMPT},
                {'role': 'system', 'content': 'provide your answer in a list like so "1. subject: <subject>, claim: <claim>"'},
                {'role': 'user', 'content': CLAIM_EXTRACTION_PROMPT.format(passage=chunk)},
            ],
            model=claim_model,
            format=TextPassage.model_json_schema() if not structured_output_model else None,
        )

        if structured_output_model:
            response = chat(
                messages=[
                    {
                        'role': 'system',
                        'content': STRUCTURED_OUTPUT_PROMPT,
                    },
                    {'role': 'user', 'content': response.message.content},
                ],
                model=structured_output_model,
                format=TextPassage.model_json_schema(),
            )

        try:
            passage_claims = TextPassage.model_validate_json(response.message.content).claims
        except ValidationError as e:
            log_event(logger, logging.ERROR, 'Error parsing model response', error=str(e), response=response.message.content)
            raise e

        return [(uuid4().hex, claim) for claim in passage_claims]

    chunk_df_exploded = apply_and_explode(chunk_df, 'chunk', extract_claims_from_chunk, ['claim_id', 'claim'])
    return chunk_df_exploded


def get_claim_sources(claim_df: pd.DataFrame) -> pd.DataFrame:
    def _perform_full_text_search(claim: str, max_retry: int = 1) -> List[Tuple[str, str]]:
        retry_count = 0
        retry_prompt = []
        while retry_count < max_retry:
            sources_df = full_text_search(
                url_parameters=FullTextSearchParams(query=claim, maxrecords=1),
                query_commands=FullTextSearchQueryCommands(domain_exclude='middleeasteye.net'),
                retry_prompt=retry_prompt,
            )
            need_retry = sources_df['warning'].iloc[0] != ''
            if need_retry:
                retry_prompt += [
                    {
                        'role': 'assistant',
                        'content': sources_df['query_without_commands'].iloc[0],
                    },
                    {
                        'role': 'user',
                        'content': FORMAT_QUERY_RETRY_PROMPT.format(warning=sources_df['warning'].iloc[0]),
                    },
                ]
                retry_count += 1
            else:
                break
        sources_list = sources_df[['warning', 'query_with_commands', 'query_without_commands', 'url']].values.tolist()
        id_resource_pair = [
            (uuid4().hex, url, query_with_commands, query_without_commands, warning)
            for warning, query_with_commands, query_without_commands, url in sources_list
        ]
        return id_resource_pair

    claim_df_exploded = apply_and_explode(
        claim_df,
        'claim',
        _perform_full_text_search,
        ['source_id', 'url', 'query_with_commands', 'query_without_commands', 'warning'],
    )
    return claim_df_exploded


def create_edge_list(chunk_claims_df: pd.DataFrame, claim_source_df: pd.DataFrame) -> pd.DataFrame:
    joint_df = pd.merge(chunk_claims_df, claim_source_df, on='claim_id', how='inner', suffixes=('_source', '_target'))
    joint_df = joint_df[['source_id_source', 'source_id_target', 'claim_id', 'claim_source', 'url_target']]
    joint_df['claim_info'] = list(zip(joint_df['claim_id'], joint_df['claim_source']))
    return joint_df


def _fetch_source_html(url: str) -> str:
    try:
        log_event(logger, logging.INFO, 'Fetching source HTML %s', url=url)
        response = requests.get(url)
    except requests.RequestException as e:
        log_event(logger, logging.ERROR, 'Failed to fetch source HTML', error=str(e), url=url)
        return ''
    return response.text


def _extract_text_from_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, 'html.parser')
    for script in soup(['script', 'style']):
        script.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split('  '))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text


def process_single_row(
    series: pd.Series,
    claim_model: str = 'gpt-oss:20b',
    structured_output_model: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> tuple:
    log_event(logger, logging.INFO, 'Processing single row', source_id=series['source_id'], url=series['url'])
    text_df = get_text_from_source(pd.DataFrame([series]))
    text_chunks_df = chunk_text(text_df, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunk_claims_df = get_chunk_claims(text_chunks_df, claim_model=claim_model, structured_output_model=structured_output_model)
    claim_source_df = get_claim_sources(chunk_claims_df[['claim_id', 'claim']])
    edge_list = create_edge_list(chunk_claims_df, claim_source_df)

    return edge_list, claim_source_df


def loop(
    content_source_df: pd.DataFrame,
    claim_model: str = 'gpt-oss:20b',
    structured_output_model: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
):
    results = content_source_df.apply(
        lambda row: process_single_row(
            row,
            claim_model=claim_model,
            structured_output_model=structured_output_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
        axis=1,
    )

    edge_lists = [result[0] for result in results]
    claim_source_dfs = [result[1] for result in results]

    return pd.concat(claim_source_dfs, ignore_index=True), pd.concat(edge_lists, ignore_index=True)


def main(content_source_df: pd.DataFrame, hops: int = 2) -> pd.DataFrame:
    edge_lists = []
    for _ in range(hops):
        claim_source_df, edge_list = loop(
            content_source_df=content_source_df,
            claim_model='mistral:7b',
        )
        content_source_df = claim_source_df[['source_id', 'url']]
        edge_lists.append(edge_list)
        log_event(logger, logging.INFO, 'Completed hop', new_edges=len(edge_list))

    return pd.concat(edge_lists, ignore_index=True)


if __name__ == '__main__':
    articles_pd = pd.DataFrame(
        {
            'source_id': [1],
            'timestamp': ['20250824151500'],
            'language': ['en'],
            'url': ['https://www.middleeasteye.net/news/poll-finds-60-percent-americans-oppose-military-aid-israel'],
        }
    )

    edge_list = main(articles_pd, hops=2)
    edge_list.to_csv('edge_list_2_hop.csv', index=False)
