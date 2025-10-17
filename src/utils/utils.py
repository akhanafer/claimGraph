import logging
from pprint import pformat
from typing import Any, Callable, List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def log_event(logger: logging.Logger, level: str, message: str, **kwargs):
    if kwargs:
        message = f"{message} - {pformat(kwargs)}"
    if level == logging.INFO:
        logger.info(message)
    elif level == logging.WARNING:
        logger.warning(message)
    elif level == logging.ERROR:
        logger.error(message)
    elif level == logging.DEBUG:
        logger.debug(message)
    elif level == logging.CRITICAL:
        logger.critical(message)
    else:
        raise ValueError(f"Unknown log level: {level}")


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


def explode(df: pd.DataFrame, column: str, new_columns: List[str]):
    df_exploded = df.copy().explode(column)
    df_exploded = df_exploded.dropna(subset=[column])
    df_exploded[new_columns] = pd.DataFrame(df_exploded[column].tolist(), index=df_exploded.index)
    df_exploded = df_exploded.drop(columns=[column])
    return df_exploded


def fetch_source_html(url: str) -> str:
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


def extract_text_from_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, 'html.parser')
    for script in soup(['script', 'style']):
        script.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split('  '))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text
