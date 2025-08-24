import io
import logging

import pandas as pd
import requests

from src.pydantic_models.gdelt_api_params import (
    FullTextSearchParams,
    FullTextSearchQueryCommands,
)

GDELT_FULL_TEXT_SEARCH_BASE_URL = 'https://api.gdeltproject.org/api/v1/search_ftxtsearch/search_ftxtsearch?'
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def full_text_search(url_parameters: FullTextSearchParams, query_commands: FullTextSearchQueryCommands) -> pd.DataFrame:
    try:
        query = _construct_query(query=url_parameters.query, query_commands=query_commands).strip()

        url_parameters.query = query

        response = requests.get(url=GDELT_FULL_TEXT_SEARCH_BASE_URL, params=url_parameters.model_dump().items())

        logger.info(f'Request URL: {response.url}')

        response_csv = io.StringIO(response.text)
        response_pdf = pd.read_csv(response_csv, header=None, names=['timestamp', 'language', 'url'])

        return response_pdf

    except requests.RequestException as e:
        logger.exception(f'URL list GET request failed: {e}')

        return None


def _construct_query(query: str, query_commands: FullTextSearchQueryCommands) -> str:
    final_query = f'{query} '
    for field, value in query_commands.model_dump().items():
        if value:
            final_query += f'{field}:{value} '

    logger.info(f'Final query: {final_query}')
    return final_query


if __name__ == "__main__":
    url_params = FullTextSearchParams(
        query='gaza starvation',
        dropdup=True,
        maxrows=10
    )

    query_commands = FullTextSearchQueryCommands(
        lastminutes=100000,
        sortby='rel',
    )

    response_pdf = full_text_search(
        url_parameters=url_params,
        query_commands=query_commands,
    )

    print(response_pdf.iloc[2]['url'])
