# TODO: Handle CSV Column Names
import logging
from typing import Any, Callable, List, Optional, Tuple
from uuid import uuid4

import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ollama import chat
from pydantic import BaseModel, ValidationError

from src.exceptions import PydanticValidationError
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


def apply_and_explode(
    df: pd.DataFrame, column: str, func: Callable[[Any], List[Tuple]], new_columns: List[str], **kwargs
) -> pd.DataFrame:
    '''
    Applies function to each row in df for the specified column then explodes

    This function applies the given function `func` tto each entry in the specified `column` of the DataFrame `df`.
    The function is expected to return a list of tuples. The DataFrame is then exploded based on these tuples,
    and the tuples are split into separate columns as specified in `new_columns`.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to process.
    column : str
        The name of the column in `df` to which `func` will be applied.
    func : Callable[[Any], List[Tuple]]
        A function that takes a single argument (the value from the specified column) and returns a list of tuples.
    new_columns : List[str]
        A list of new column names to create from the elements of the tuples returned by `func`.
    **kwargs : dict
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the exploded tuples split into separate columns.

    Raises
    ------
    KeyError
        If the specified `column` does not exist in `df`.
    ValueError
        If `df` is empty.
        If any of the `new_columns` already exist in `df`.
        If the length of the tuples returned by `func` does not match the length of `new_columns`.

    Examples
    --------
    >>> import pandas as pd
    >>> data = {'number': [1, 2, 3]}
    >>> df = pd.DataFrame(data)
    >>> def exponents(x):
    ...     return [(i, x**i) for i in range(1, 4)]
    >>> new_columns = ['exponent', 'result']
    >>> result = apply_and_explode(df, 'number', exponents, new_columns)
    >>> print(result)
         number  exponent  result
    0       1         1       1
    1       1         2       1
    2       1         3       1
    3       2         1       2
    4       2         2       4
    5       2         3       8
    6       3         1       3
    7       3         2       9
    8       3         3      27
    '''
    df_copy = df.copy()
    if column not in df_copy.columns:
        raise KeyError(f"Column '{column}' does not exist in DataFrame.")
    if df_copy.empty:
        raise ValueError("Input DataFrame is empty.")
    if any(col in new_columns for col in df_copy.columns):
        raise ValueError("One or more new_columns already exist in DataFrame.")
    df_copy['info_tuple'] = df_copy[column].apply(func, **kwargs)
    if len(df_copy['info_tuple'].iloc[0]) == 0:  # Don't explode if the function returns empty lists
        df_copy[new_columns] = None
        df_copy = df_copy.drop(columns=['info_tuple'])
        return df_copy

    df_exploded = df_copy.explode('info_tuple').reset_index(drop=True)
    if len(new_columns) != len(df_exploded['info_tuple'].iloc[0]):
        raise ValueError(f"Function return length does not match new_columns length for column '{column}'.")
    df_exploded[new_columns] = pd.DataFrame(df_exploded['info_tuple'].tolist(), index=df_exploded.index)
    return df_exploded.drop(columns=['info_tuple'])


def get_text_from_source(source: str) -> str:
    '''
    Fetches and extracts text content from the given source URL.

    Parameters
    ----------
    source : str
        The URL of the source from which to fetch and extract text.

    Returns
    -------
    str
        The extracted text content with no HTML tags.

    Raises
    ------
    ValueError
        If the input source URL is empty.
    '''
    log_event(logger, logging.INFO, 'Getting text from source URL', source=source)
    raw_html = _fetch_source_html(source)
    content_text = _extract_text_from_html(raw_html)
    return content_text


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> pd.DataFrame:
    '''
    Chunks the input text into smaller segments for processing.

    Parameters
    ----------
    text : str
        The input text to chunk.
    chunk_size : int
        The size of each chunk.
    chunk_overlap : int
        The overlap between chunks.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the text chunks. Each row represents a chunk with columns 'chunk_id' and 'chunk'.

    Raises
    ------
    ValueError
        If the input text is empty.

    Examples
    --------
    >>> chunk_text("This is a test article. It contains multiple sentences.", chunk_size=10, chunk_overlap=2)
         chunk_id           chunk
    0  e4b8f8c9e1f14c3a  This is a
    1  9f1c3e4b8fc9e1f1  a test ar
    2  1f4c3a9f1c3e4b8  test article.
    3  c3e4b8e4b8f8c9e  article. It
    4  b8f8c9e1f4c3a9f  It contains
    5  f8c9e1f4c3a9f1c  contains multiple
    6  e1f4c3a9f1c3e4b  multiple sentences.
    '''
    if not text:
        raise ValueError("Chunk Text Input is empty.")

    log_event(logger, logging.INFO, 'Chunking text', chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def recursive_text_split(text: str) -> List[Tuple[str, str]]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return [(uuid4().hex, chunk) for chunk in text_splitter.split_text(text)]

    text_df = pd.DataFrame({'text': [text]})
    text_df_copy_exploded = apply_and_explode(text_df, 'text', recursive_text_split, ['chunk_id', 'chunk'])

    return text_df_copy_exploded


def get_chunk_claims(
    chunk_df: pd.DataFrame,
    claim_model: str = 'gpt-oss:20b',
    structured_output_model: Optional[str] = None,  # Needed for models that don't support structured output
) -> pd.DataFrame:
    '''
    Extracts claims from text chunks using a language model.

    Parameters
    ----------
    chunk_df : pd.DataFrame
        A DataFrame containing the text chunks to process. Must contain `chunk` and `chunk_id` columns.
    claim_model : str
        The model to use for claim extraction.
    structured_output_model : Optional[str]
        The model to use for structured output (if needed).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the extracted claims from each chunk.
        Each row represents a claim with columns 'chunk_id', 'chunk', 'claim_id', and 'claim'.

    Raises
    ------
    KeyError
        If the required columns are missing in `chunk_df`.
    ValueError
        If `chunk_df` is empty.
        If there is an error parsing the model response.
    PydanticValidationError
        If there is an error validating the model response.

    Examples
    --------
    >>> import pandas as pd
    >>> data = {
    ...     'chunk_id': ['1', '2'],
    ...     'chunk': [
    ...         'This is chunk one with claim one and claim two.',
    ...         'This is chunk two with claim one.'
    ...     ]}
    >>> chunk_df = pd.DataFrame(data)
    >>> claims_df = get_chunk_claims(chunk_df, claim_model='gpt-oss:20b')
    >>> print(claims_df)
        chunk_id           chunk                                    claim_id        claim
    0       1      This is chunk one with claim one and claim two.   abc123       Claim one.
    1       1      This is chunk one with claim one and claim two.   def456       Claim two.
    2       2      This is chunk two with claim one.                 ghi789       Claim one.

    '''
    log_event(
        logger,
        logging.INFO,
        'Extracting claims from chunks',
        claim_model=claim_model,
        structured_output_model=structured_output_model,
        num_chunks=len(chunk_df),
    )
    if 'chunk' not in chunk_df.columns:
        raise KeyError("Column 'chunk' does not exist in DataFrame.")
    if 'chunk_id' not in chunk_df.columns:
        raise KeyError("Column 'chunk_id' does not exist in DataFrame.")
    if chunk_df.empty:
        raise ValueError(
            "Get Chunk Claims Input DataFrame is empty. This may happen if no text was extracted from the source URL."
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
            raise PydanticValidationError("Error parsing model response when extracting claims")

        return [(uuid4().hex, claim) for claim in passage_claims]

    chunk_df_exploded = apply_and_explode(chunk_df, 'chunk', extract_claims_from_chunk, ['claim_id', 'claim'])
    return chunk_df_exploded


def get_claim_sources(claim_df: pd.DataFrame, max_retry: int = 1) -> pd.DataFrame:
    '''
    Get the sources for each claim in the DataFrame.
    Parameters
    ----------
    claim_df : pd.DataFrame
        A DataFrame containing the claims to search for sources. Must contain `claim` and `claim_id` columns.
    max_retry : int
        The maximum number of retries for the full text search in case of warnings or errors in trying to find a source.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the sources for each claim.
        Each row represents a claim with columns:
        'claim_id',
        'claim',
        'source_id',
        'url': The URL of the source,
        'query_with_commands': the GDELT query used with commands,
        'query_without_commands': The  GDELT query used without commands,
        'warning': Any warning messages, typically from the GDELT API when trying to find a source.
            Typically indicates that the query does not comply with GDELT's interface or that no results were found.

    Raises
    ------
    KeyError
        If the required columns are missing in `claim_df`.
    ValueError
        If `claim_df` is empty.

    Examples
    --------
    >>> import pandas as pd
    >>> data = {'claim_id': ['1', '2'], 'claim': ['Claim one', 'Claim two']}
    >>> claim_df = pd.DataFrame(data)
    >>> sources_df = get_claim_sources(claim_df, max_retry=1)
    >>> print(sources_df)
        claim_id    claim       source_id       url                                     query_with_commands  query_without_commands  warning # noqa E501
    0       1   Claim one       abc123         http://example.com/source1   ...                     ...                     ... # noqa E501
    1       2   Claim two       def456         http://example.com/source2   ...                     ...                     ... # noqa E501

    '''

    def _perform_full_text_search(claim: str, max_retry: int) -> List[Tuple[str, str]]:
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
        max_retry=max_retry,
    )
    return claim_df_exploded


def create_edge_list(chunk_claims_df: pd.DataFrame, claim_source_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Create an edge list DataFrame mapping the relationships between original content sources and new claim sources.

    Parameters
    ----------
    chunk_claims_df : pd.DataFrame
        A DataFrame containing chunk claims with columns `source_id`, `claim_id`, and `claim`.
         Source_id is the source from which the claim was originally extracted.
    claim_source_df : pd.DataFrame
        A DataFrame containing claim sources with columns `claim_id`,`source_id` and `url`.
            Source_id is the new source where the original claim was also found.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the edge list with columns:
        'source_id_source': The source ID of the original content source.
        'source_id_target': The source ID of the new source where this claim was also found.
        'claim_id': The unique identifier for the claim.
        'claim': The text of the claim.
        'url': The URL of the new claim source.
        'claim_info': A tuple containing (claim_id, claim) for easy reference.

    Examples
    --------
    >>> import pandas as pd
    >>> chunk_claims_data = {'chunk_id': ['1', '2'], 'chunk': ['This is chunk one.', 'This is chunk two.'], 'claim_id': ['c1', 'c2'], 'claim': ['Claim one', 'Claim two'], 'source_id': [1, 1]} # noqa E501
    >>> claim_source_data = {'claim_id': ['c1', 'c2'], 'claim': ['Claim one', 'Claim two'], 'source_id': [2, 3], 'url': ['http://example.com/source1', 'http://example.com/source2']} # noqa E501
    >>> chunk_claims_df = pd.DataFrame(chunk_claims_data)
    >>> claim_source_df = pd.DataFrame(claim_source_data)
    >>> edge_list_df = create_edge_list(chunk_claims_df, claim_source_df)
    >>> print(edge_list_df)
                source_id_source  source_id_target            claim_id      claim                          url                         claim_info # noqa E501
    0                 1                 2                       c1          Claim one           http://example.com/source1             (c1, Claim one) # noqa E501
    1                 1                 3                       c2          Claim two           http://example.com/source2             (c2, Claim two) # noqa E501

    '''
    joint_df = pd.merge(chunk_claims_df, claim_source_df, on='claim_id', how='inner', suffixes=('_source', '_target'))
    joint_df = joint_df[['source_id_source', 'source_id_target', 'claim_id', 'claim', 'url']]
    joint_df['claim_info'] = list(zip(joint_df['claim_id'], joint_df['claim']))
    return joint_df


def _fetch_source_html(url: str) -> str:
    try:
        log_event(logger, logging.INFO, 'Fetching source HTML %s', url=url)
        response = requests.get(url)
        log_event(logger, logging.INFO, 'Fetched source HTML', status_code=response.status_code, url=url)
        response.raise_for_status()
    except requests.HTTPError as e:
        log_event(
            logger,
            logging.ERROR,
            'HTTP error occurred while fetching source HTML',
            status_code=e.response.status_code,
            error=str(e),
            url=url,
        )
        return ''
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


def create_source_claim_graph(
    series: pd.Series,
    claim_model: str = 'gpt-oss:20b',
    structured_output_model: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> tuple:
    log_event(logger, logging.INFO, 'Processing single row', source_id=series['source_id'], url=series['url'])
    text = get_text_from_source(series['url'])
    if not text:
        log_event(
            logger,
            logging.WARNING,
            'No text extracted from source URL, skipping',
            source_id=series['source_id'],
            url=series['url'],
        )
        return pd.DataFrame(), pd.DataFrame()
    text_chunks_df = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks_df['source_id'] = series['source_id']
    text_chunks_df['url'] = series['url']
    chunk_claims_df = get_chunk_claims(text_chunks_df, claim_model=claim_model, structured_output_model=structured_output_model)
    claim_source_df = get_claim_sources(chunk_claims_df[['claim_id', 'claim']])
    edge_list = create_edge_list(chunk_claims_df['source_id', 'claim_id'], claim_source_df)

    return edge_list, claim_source_df


def loop(
    content_source_df: pd.DataFrame,
    claim_model: str = 'gpt-oss:20b',
    structured_output_model: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
):
    # Don't loop on rows with no URL to avoid unnecessary LLM requests and runtime
    content_source_df = content_source_df.dropna(subset=['url'])
    results = content_source_df.apply(
        lambda row: create_source_claim_graph(
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
    for i in range(hops):
        claim_source_df, edge_list = loop(
            content_source_df=content_source_df,
            claim_model='mistral:7b',
        )
        claim_source_df.to_csv(f'claim_sources_hop_{i}.csv', index=False)
        content_source_df = claim_source_df[['source_id', 'url']]
        edge_lists.append(edge_list)
        log_event(logger, logging.INFO, 'Completed hop', new_edges=len(edge_list))

    return pd.concat(edge_lists, ignore_index=True)


if __name__ == '__main__':
    articles_pd = pd.DataFrame(
        {
            'source_id': [1],
            'url': ['https://www.middleeasteye.net/news/poll-finds-60-percent-americans-oppose-military-aid-israel'],
        }
    )

    edge_list = main(articles_pd, hops=2)
    edge_list.to_csv('edge_list_2_hop.csv', index=False)
