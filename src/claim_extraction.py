import logging
from typing import List, Optional, Tuple
from uuid import uuid4

import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ollama import chat
from pydantic import BaseModel, ValidationError

from src.gdelt_api_client import full_text_search
from src.prompts import CLAIM_EXTRACTION_PROMPT, CLAIM_EXTRACTION_SYSTEM_PROMPT
from src.pydantic_models.gdelt_api_params import (
    FullTextSearchParams,
    FullTextSearchQueryCommands,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class TextPassage(BaseModel):
    subject: str
    claims: List[str]


def get_text_from_source(content_source_df: pd.DataFrame, keep_html_col: bool = False) -> pd.DataFrame:
    content_text_df = content_source_df.copy()
    content_text_df['raw_html'] = content_text_df['url'].apply(_fetch_source_html)
    content_text_df['content_text'] = content_text_df['raw_html'].apply(_extract_text_from_html)
    if not keep_html_col:
        content_text_df = content_text_df.drop(columns=['raw_html'])
    return content_text_df


def chunk_text(text_df: pd.DataFrame, chunk_size: int, chunk_overlap: int) -> pd.DataFrame:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    text_df_copy = text_df.copy()
    text_df_copy['chunk'] = text_df_copy['content_text'].apply(
        lambda text: [(uuid4().hex, chunk) for chunk in text_splitter.split_text(text)]
    )
    text_df_copy_exploded = text_df_copy.explode('chunk').reset_index(drop=True)
    text_df_copy_exploded[['chunk_id', 'chunk']] = pd.DataFrame(
        text_df_copy_exploded['chunk'].tolist(), index=text_df_copy_exploded.index
    )

    return text_df_copy_exploded


def get_chunk_claims(
    chunk_df: pd.DataFrame,
    claim_model: str = 'gpt-oss:20b',
    structured_output_model: Optional[str] = None,  # Needed for models that don't support structured output
) -> TextPassage:
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
                        'content': 'You are simply responsible for getting the subject'
                        ' and claim from the provided text to comply with the output format given to you',
                    },
                    {'role': 'user', 'content': response.message.content},
                ],
                model=structured_output_model,
                format=TextPassage.model_json_schema(),
            )

        try:
            passage_claims = TextPassage.model_validate_json(response.message.content).claims
        except ValidationError as e:
            logger.error(f'Error parsing model response: {e}')
            raise e

        return [(uuid4().hex, claim) for claim in passage_claims]

    chunk_df_copy = chunk_df.copy()
    chunk_df_copy['claim_tuples'] = chunk_df_copy['chunk'].apply(extract_claims_from_chunk)
    chunk_df_copy_exploded = chunk_df_copy.explode('claim_tuples').reset_index(drop=True)
    chunk_df_copy_exploded[['claim_id', 'claim']] = pd.DataFrame(
        chunk_df_copy_exploded['claim_tuples'].tolist(), index=chunk_df_copy_exploded.index
    )

    return chunk_df_copy_exploded.drop(columns=['claim_tuples'])


def get_claim_sources(claim_df: pd.DataFrame) -> pd.DataFrame:
    def _perform_full_text_search(claim: str) -> List[Tuple[str, str]]:
        sources_df = full_text_search(
            url_parameters=FullTextSearchParams(query=claim), query_commands=FullTextSearchQueryCommands()
        )
        sources_list = sources_df[['formatted_query', 'url']].values.tolist()
        id_resource_pair = [(uuid4().hex, url, formatted_query) for formatted_query, url in sources_list]
        return_value = id_resource_pair if len(id_resource_pair) > 0 else [(0, '', '')]
        return return_value

    claim_df_copy = claim_df.copy()
    claim_df_copy['source_tuples'] = claim_df_copy['claim'].apply(_perform_full_text_search)
    claim_df_copy_exploded = claim_df_copy.explode('source_tuples').reset_index(drop=True)
    claim_df_copy_exploded[['source_id', 'source', 'formatted_query']] = pd.DataFrame(
        claim_df_copy_exploded['source_tuples'].tolist(), index=claim_df_copy_exploded.index
    )

    return claim_df_copy_exploded.drop(columns=['source_tuples'])


def _fetch_source_html(url: str) -> str:
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    logger.error(f'Error fetching source HTML: {response.status_code}')
    return ''


def _extract_text_from_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, 'html.parser')
    for script in soup(['script', 'style']):
        script.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split('  '))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text


def main(
    content_source_df: pd.DataFrame,
    claim_model: str = 'gpt-oss:20b',
    structured_output_model: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> pd.DataFrame:

    text_df = get_text_from_source(content_source_df)
    chunked_df = chunk_text(text_df, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunk_claims_df = get_chunk_claims(chunked_df, claim_model=claim_model, structured_output_model=structured_output_model)
    claim_source_df = get_claim_sources(chunk_claims_df[['claim_id', 'claim']])

    return chunk_claims_df, claim_source_df


if __name__ == '__main__':
    articles_pd = pd.DataFrame(
        {
            'resource_id': [1],
            'timestamp': ['20250824151500'],
            'language': ['en'],
            'url': ['https://www.middleeasteye.net/news/poll-finds-60-percent-americans-oppose-military-aid-israel'],
        }
    )

    chunk_claims_df, claim_source_df = main(articles_pd, claim_model='mistral:7b')
    chunk_claims_df.to_csv('chunk_claims.csv', index=False)
    claim_source_df.to_csv('claim_sources.csv', index=False)
