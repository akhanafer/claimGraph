import io
import logging

import pandas as pd
import requests
from ollama import chat

from src.prompts import FORMAT_QUERY_PROMPT, FORMAT_QUERY_SYSTEM_PROMPT
from src.pydantic_models.gdelt_api_params import (
    FullTextSearchParams,
    FullTextSearchQueryCommands,
)

GDELT_FULL_TEXT_SEARCH_BASE_URL = 'https://api.gdeltproject.org/api/v2/doc/doc?'
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def full_text_search(url_parameters: FullTextSearchParams, query_commands: FullTextSearchQueryCommands) -> pd.DataFrame:
    try:
        formatted_query = format_query(url_parameters.query)
        query = _construct_query(query=formatted_query, query_commands=query_commands).strip()

        url_parameters.query = query

        response = requests.get(url=GDELT_FULL_TEXT_SEARCH_BASE_URL, params=url_parameters.model_dump().items())

        logger.info(f'Request URL: {response.url}')

        response_csv = io.StringIO(response.text)
        response_pdf = pd.read_csv(response_csv, skiprows=1, names=['url', 'mobile_url', 'date', 'title']).drop(
            columns=['mobile_url']
        )
        response_pdf['formatted_query'] = formatted_query
        return response_pdf

    except requests.RequestException as e:
        logger.exception(f'URL list GET request failed: {e}')

        return None


def format_query(query: str, model: str = 'mistral:7b') -> str:
    response = chat(
        messages=[
            {'role': 'system', 'content': FORMAT_QUERY_SYSTEM_PROMPT},
            {'role': 'user', 'content': FORMAT_QUERY_PROMPT.format(claim=query)},
        ],
        model=model,
    )
    return response.message.content


def _construct_query(query: str, query_commands: FullTextSearchQueryCommands) -> str:
    final_query = f'{query} '
    for field, value in query_commands.model_dump().items():
        if value:
            final_query += f'{field}:{value} '

    logger.info(f'Final query: {final_query}')
    return final_query


if __name__ == "__main__":
    url_params = FullTextSearchParams(
        query='"US military aid" AND Israel AND poll AND "oppose" AND "60 percent" AND Americans',
    )

    query_commands = FullTextSearchQueryCommands(domain='-middleeasteye.net')

    response_pdf = full_text_search(
        url_parameters=url_params,
        query_commands=query_commands,
    )

    response_pdf.to_csv('articlesV2.csv', index=False)
