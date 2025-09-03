import io
import logging

import pandas as pd
import requests
from ollama import chat

from src.exceptions import GDELTAPIRequestError
from src.prompts import FORMAT_QUERY_PROMPT, FORMAT_QUERY_SYSTEM_PROMPT
from src.pydantic_models.gdelt_api_params import (
    FullTextSearchParams,
    FullTextSearchQueryCommands,
)
from src.utils import log_event

GDELT_FULL_TEXT_SEARCH_BASE_URL = 'https://api.gdeltproject.org/api/v2/doc/doc?'
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)


def full_text_search(url_parameters: FullTextSearchParams, query_commands: FullTextSearchQueryCommands) -> pd.DataFrame:
    log_event(
        logger,
        logging.INFO,
        'Performing GDELT full text search %s',
        parameters=url_parameters.model_dump(),
        query_commands=query_commands.model_dump(),
    )

    formatted_query = format_query(url_parameters.query)
    query = _construct_query(query=formatted_query, query_commands=query_commands).strip()

    url_parameters.query = query

    try:
        response = requests.get(url=GDELT_FULL_TEXT_SEARCH_BASE_URL, params=url_parameters.model_dump().items())
    except requests.RequestException as e:
        log_event(logger, logging.ERROR, 'GDELT API request failed %s', error=str(e), final_query=query)
        raise GDELTAPIRequestError(f"GDELT API request failed: {e}")

    log_event(logger, logging.INFO, 'GDELT API response %s', response_text=response.text, url=response.url)
    response_text = response.text
    response_csv = io.StringIO(response_text)
    response_pdf = pd.read_csv(response_csv, skiprows=1, names=['url', 'mobile_url', 'date', 'title']).drop(
        columns=['mobile_url']
    )

    if response_pdf.empty:
        log_event(
            logger,
            logging.WARNING,
            'GDELT Full Text Search returned no results %s',
            url=response.url,
            response_content=response.text,
        )
        response_pdf = pd.DataFrame(
            {'url': [None], 'date': [None], 'title': [None], 'formatted_query': [formatted_query], 'warning': [response_text]}
        )

    else:
        response_pdf['formatted_query'] = formatted_query
        response_pdf['warning'] = ''
    return response_pdf


def format_query(query: str, model: str = 'mistral:7b') -> str:
    response = chat(
        messages=[
            {'role': 'system', 'content': FORMAT_QUERY_SYSTEM_PROMPT},
            {'role': 'user', 'content': FORMAT_QUERY_PROMPT.format(claim=query)},
        ],
        model=model,
    ).message.content

    log_event(logger, logging.INFO, 'Generated formatted query %s', formatted_query=response, model=model)

    return response


def _construct_query(query: str, query_commands: FullTextSearchQueryCommands) -> str:
    final_query = f'{query} '
    for field, value in query_commands.model_dump().items():
        if value:
            final_query += f'{field}:{value} '

    return final_query


if __name__ == "__main__":
    url_params = FullTextSearchParams(
        query='"US military aid" AND Israel AND poll AND "oppose" AND "60 percent" AND Americans',
    )

    query_commands = FullTextSearchQueryCommands(domain='middleeasteye.net')

    response_pdf = full_text_search(
        url_parameters=url_params,
        query_commands=query_commands,
    )

    response_pdf.to_csv('articlesV2.csv', index=False)
