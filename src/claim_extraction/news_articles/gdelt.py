# TODO: Handle CSV Column Names
import asyncio
import logging
from typing import List, Tuple
from uuid import uuid4

import pandas as pd
import requests
from openai import AsyncOpenAI

from src.claim_extraction.claim_extractor import ClaimExtractor
from src.gdelt_api_client import full_text_search
from src.prompts import FORMAT_QUERY_RETRY_PROMPT
from src.pydantic_models.gdelt_api_params import (
    FullTextSearchParams,
    FullTextSearchQueryCommands,
)
from src.utils.utils import (
    explode,
    extract_text_from_html,
    fetch_source_html,
    log_event,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class GDELTClaimExtractor(ClaimExtractor):
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        # gdelt_client,
        claim_model: str = 'gpt-oss:20b',
        structured_output_model: str = 'gpt-oss:20b',
        text_to_gdelt_query_model: str = 'mistral:7b',
        text_to_gdelt_query_max_retry: int = 1,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ):
        super().__init__(
            openai_client=openai_client,
            claim_model=claim_model,
            structured_output_model=structured_output_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.text_to_gdelt_query_model = text_to_gdelt_query_model
        self.text_to_gdelt_query_max_retry = text_to_gdelt_query_max_retry

    def get_text_from_url(self, url: str) -> str:
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
        log_event(logger, logging.INFO, 'Getting text from source URL', url=url)
        raw_html = fetch_source_html(url)
        content_text = extract_text_from_html(raw_html)
        return content_text

    async def perform_full_text_search(self, claim: str, max_retry: int, domain_exclude: str) -> List[Tuple[str, str]]:
        retry_count = 0
        retry_prompt = []
        while retry_count < max_retry:
            sources_df = full_text_search(
                url_parameters=FullTextSearchParams(query=claim, maxrecords=1),
                query_commands=FullTextSearchQueryCommands(domain_exclude=domain_exclude),
                retry_prompt=retry_prompt,
                text_to_gdelt_query_model=self.text_to_gdelt_query_model,
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

    async def get_claim_sources(self, claim_df: pd.DataFrame, domain_exclude: str, max_retry: int = 1) -> pd.DataFrame:
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
        tasks = [self.perform_full_text_search(claim, max_retry, domain_exclude) for claim in claim_df['claim']]
        results = await asyncio.gather(*tasks)
        claim_df['claim_sources'] = results
        claim_df_exploded = explode(
            claim_df, 'claim_sources', ['source_id', 'url', 'query_with_commands', 'query_without_commands', 'warning']
        )
        claim_df_exploded['domain'] = claim_df_exploded['url'].apply(
            lambda x: requests.utils.urlparse(x).netloc if pd.notna(x) else None
        )

        # Drop rows with missing URLs as they are not useful.
        # Since a claim will always have at least one source, it's safe to drop rows with missing URLs
        claim_df_exploded = claim_df_exploded.dropna(subset=['url'])
        return claim_df_exploded

    async def create_source_claim_graph(self, series: pd.Series) -> tuple:
        log_event(logger, logging.INFO, 'Processing single row', source_id=series['source_id'], url=series['url'])
        text = self.get_text_from_url(series['url'])
        if not text:
            log_event(
                logger,
                logging.WARNING,
                'No text extracted from source URL, skipping',
                source_id=series['source_id'],
                url=series['url'],
            )
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        text_chunks_df = self.chunk_text(text, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        text_chunks_df['source_id'] = series['source_id']
        text_chunks_df['url'] = series['url']
        domain = requests.utils.urlparse(series['url']).netloc
        text_chunks_df['domain'] = domain
        chunk_claims_df = await self.get_chunk_claims(text_chunks_df)
        claim_source_df = await self.get_claim_sources(
            chunk_claims_df[['claim_id', 'claim']], domain_exclude=domain, max_retry=self.text_to_gdelt_query_max_retry
        )
        source_to_source_edge_list, source_to_claim_edge_list = self.create_edge_lists(chunk_claims_df, claim_source_df)

        return source_to_source_edge_list, source_to_claim_edge_list, claim_source_df
