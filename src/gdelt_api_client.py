import io
import logging
from typing import Dict, List, Optional

import pandas as pd
import requests
from ollama import chat

from src.consts import GDELT_API_ERROR, NO_RESULTS_WARNING
from src.exceptions import GDELTAPIRequestError
from src.prompts import FORMAT_QUERY_PROMPT, FORMAT_QUERY_SYSTEM_PROMPT
from src.pydantic_models.gdelt_api_params import (
    FullTextSearchParams,
    FullTextSearchQueryCommands,
)
from src.utils.utils import log_event

GDELT_FULL_TEXT_SEARCH_BASE_URL = 'https://api.gdeltproject.org/api/v2/doc/doc?'
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", filename='app.log')
logging.getLogger("httpx").setLevel(logging.WARNING)


def full_text_search(
    url_parameters: FullTextSearchParams,
    query_commands: FullTextSearchQueryCommands,
    text_to_gdelt_query_model: str,
    retry_prompt: Optional[List[Dict[str, str]]] = [],
) -> pd.DataFrame:
    log_event(
        logger,
        logging.INFO,
        'Performing GDELT full text search %s',
        parameters=url_parameters.model_dump(),
        query_commands=query_commands.model_dump(),
    )

    gdelt_query = _convert_text_to_gdelt_query_format(
        url_parameters.query, retry_prompt=retry_prompt, model=text_to_gdelt_query_model
    )
    query_with_commands = _add_query_commands_to_gdelt_query(query=gdelt_query, query_commands=query_commands).strip()
    request_parameters = url_parameters.model_copy(update={'query': query_with_commands})

    try:
        response = requests.get(url=GDELT_FULL_TEXT_SEARCH_BASE_URL, params=request_parameters.model_dump().items())
    except requests.RequestException as e:
        log_event(
            logger,
            logging.ERROR,
            'GDELT API request failed %s',
            error=str(e),
        )
        raise GDELTAPIRequestError(f"GDELT API request failed: {e}")

    log_event(
        logger,
        logging.INFO,
        'GDELT API request completed with no error',
        status_code=response.status_code,
        url=response.url,
        query=url_parameters.query,
    )
    return _process_response(response=response, query_with_commands=query_with_commands, query_without_commands=gdelt_query)


def _process_response(response: requests.Response, query_with_commands: str, query_without_commands: str) -> pd.DataFrame:
    content_type = response.headers.get('Content-Type', '').lower()
    is_csv = 'text/csv' in content_type or 'application/csv' in content_type or 'text/plain' in content_type
    if not is_csv:
        log_event(
            logger,
            logging.WARNING,
            'GDELT Full Text Search returned no results',
            content_type=content_type,
            response_text=response.text,
        )
        return _create_error_result(query_with_commands, query_without_commands, f'{GDELT_API_ERROR}: {response.text}')

    try:
        response_csv = io.StringIO(response.text)
        response_pdf = pd.read_csv(response_csv)

        if not response_pdf.empty:
            log_event(
                logger,
                logging.INFO,
                'GDELT Full Text Search successful and returned results',
            )
            response_pdf = _unpivot_df(response_pdf)
            response_pdf['query_with_commands'] = query_with_commands
            response_pdf['query_without_commands'] = query_without_commands
            response_pdf['warning'] = ''
            response_pdf = response_pdf.groupby('tone', group_keys=False).sample(
                n=1, random_state=42, replace=True
            )  # TODO: Make n configurable
            log_event(logger, logging.INFO, 'Sampled results from each tone category', num_samples=len(response_pdf))
            return response_pdf
        else:
            log_event(
                logger, logging.WARNING, 'GDELT Full Text Search successful but found no results', response_text=response.text
            )
            return _create_empty_result(query_with_commands, query_without_commands, NO_RESULTS_WARNING)
    except pd.errors.ParserError as e:
        log_event(
            logger,
            logging.ERROR,
            'Failed to parse GDELT Full Text Search response',
            error=str(e),
        )
        raise pd.errors.ParserError(f"Failed to parse GDELT Full Text Search response: {e}")
    except Exception as e:
        log_event(logger, logging.ERROR, f'Unexpected error parsing response: {e}')
        raise Exception(f"Unexpected error parsing response: {e}")


def _unpivot_df(df: pd.DataFrame) -> pd.DataFrame:
    url_cols = [col for col in df.columns if "TopArtURL" in col]
    title_cols = [col for col in df.columns if "TopArtTitle" in col]

    urls = df.melt(id_vars=["Label", "Count"], value_vars=url_cols, var_name="url_num", value_name="url")
    titles = df.melt(id_vars=["Label", "Count"], value_vars=title_cols, var_name="title_num", value_name="title")

    result = pd.concat([urls[["Label", "url"]], titles["title"]], axis=1)

    result = result.rename(columns={"Label": "tone"})

    # Drop rows with NaN URLs, this can happen if GDELT
    # returns rows where some TopArtURL columns are empty
    result = result.dropna(subset=["url"]).reset_index(drop=True)
    return result


def _create_empty_result(query_with_commands: str, query_without_commands: str, warning: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'url': [None],
            'tone': [None],
            'date': [None],
            'title': [None],
            'query_with_commands': [query_with_commands],
            'query_without_commands': [query_without_commands],
            'warning': [warning],
        }
    )


def _create_error_result(query_with_commands: str, query_without_commands: str, error_text: str) -> pd.DataFrame:
    """Create DataFrame for error responses."""
    return pd.DataFrame(
        {
            'url': [None],
            'tone': [None],
            'date': [None],
            'title': [None],
            'query_with_commands': [query_with_commands],
            'query_without_commands': [query_without_commands],
            'warning': [error_text],
        }
    )


def _convert_text_to_gdelt_query_format(query: str, model: str, retry_prompt: Optional[List[Dict[str, str]]] = []) -> str:
    if retry_prompt:
        log_event(logger, logging.INFO, 'Retrying GDELT query formatting with additional prompt', retry_prompt=retry_prompt)
    response = chat(
        messages=[
            {'role': 'system', 'content': FORMAT_QUERY_SYSTEM_PROMPT},
            {'role': 'user', 'content': FORMAT_QUERY_PROMPT.format(claim=query)},
            *retry_prompt,
        ],
        model=model,
    ).message.content

    log_event(logger, logging.INFO, 'Generated formatted query %s', formatted_query=response, model=model)

    return response


def _add_query_commands_to_gdelt_query(query: str, query_commands: FullTextSearchQueryCommands) -> str:
    final_query = f'{query} '
    for field, value in query_commands.model_dump().items():
        if value:
            if '_exclude' in field:
                field = field.replace('_exclude', '')
                final_query += f'-{field}:{value} '
            else:
                final_query += f'{field}:{value} '

    return final_query
