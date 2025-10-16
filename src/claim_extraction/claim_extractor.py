# TODO: Handle CSV Column Names
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from uuid import uuid4

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from src.exceptions import PydanticValidationError
from src.prompts import (
    CLAIM_EXTRACTION_PROMPT,
    CLAIM_EXTRACTION_SYSTEM_PROMPT,
    STRUCTURED_OUTPUT_PROMPT,
)
from src.utils.utils import apply_and_explode, log_event

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class TextPassage(BaseModel):
    subject: str
    claims: List[str]


class ClaimExtractor(ABC):
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        claim_model: str = 'gpt-oss:20b',
        structured_output_model: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ):
        self.openai_client = openai_client
        self.claim_model = claim_model
        self.structured_output_model = structured_output_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> pd.DataFrame:
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

    async def extract_claims_from_chunk(self, chunk: str) -> List[Tuple[str, str]]:
        response = await self.openai_client.chat.completions.parse(
            model=self.claim_model,
            messages=[
                {'role': 'system', 'content': CLAIM_EXTRACTION_SYSTEM_PROMPT},
                {'role': 'system', 'content': 'provide your answer in a list like so "1. subject: <subject>, claim: <claim>"'},
                {'role': 'user', 'content': CLAIM_EXTRACTION_PROMPT.format(passage=chunk)},
            ],
            response_format=TextPassage if not self.structured_output_model else None,
        )

        if self.structured_output_model:
            response = await self.openai_client.chat.completions.parse(
                model=self.structured_output_model,
                messages=[
                    {
                        'role': 'system',
                        'content': STRUCTURED_OUTPUT_PROMPT,
                    },
                    {'role': 'user', 'content': response.choices[0].message.content},
                ],
                response_format=TextPassage,
            )

        try:
            passage_claims = TextPassage.model_validate_json(response.choices[0].message.parsed.model_dump_json()).claims
        except ValidationError as e:
            log_event(
                logger, logging.ERROR, 'Error parsing model response', error=str(e), response=response.choices[0].message.content
            )
            raise PydanticValidationError("Error parsing model response when extracting claims")

        return [(uuid4().hex, claim) for claim in passage_claims]

    async def get_chunk_claims(
        self,
        chunk_df: pd.DataFrame,
    ) -> pd.DataFrame:
        '''Async version of get_chunk_claims.
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
            claim_model=self.claim_model,
            structured_output_model=self.structured_output_model,
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

        tasks = [self.extract_claims_from_chunk(chunk) for chunk in chunk_df['chunk']]
        results = await asyncio.gather(*tasks)

        # TODO: see if you extract to a function
        chunk_df['claims'] = results
        chunk_df_exploded = chunk_df.explode('claims').rename(columns={'claims': 'claim_info'})
        chunk_df_exploded = chunk_df_exploded.dropna(subset=['claim_info'])
        chunk_df_exploded[['claim_id', 'claim']] = pd.DataFrame(
            chunk_df_exploded['claim_info'].tolist(), index=chunk_df_exploded.index
        )
        chunk_df_exploded = chunk_df_exploded.drop(columns=['claim_info'])
        #
        # Drop rows with missing or empty claim_id and claim. This happens if no claims were found in a chunk.
        chunk_df_exploded = chunk_df_exploded.dropna(subset=['claim_id', 'claim'])
        chunk_df_exploded = chunk_df_exploded[chunk_df_exploded['claim_id'] != '']
        chunk_df_exploded = chunk_df_exploded[chunk_df_exploded['claim'] != '']
        return chunk_df_exploded

    def create_edge_lists(
        self, chunk_claims_df: pd.DataFrame, claim_source_df: pd.DataFrame, source_to_source=True, source_to_claim=True
    ) -> pd.DataFrame:
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
        source_to_source_edge_list = pd.merge(
            chunk_claims_df, claim_source_df, on='claim_id', how='inner', suffixes=('_source', '_target')
        )

        source_to_claim_edge_list = pd.concat(
            [
                chunk_claims_df[['source_id', 'url', 'domain', 'claim_id', 'claim']],
                claim_source_df[['source_id', 'url', 'domain', 'claim_id', 'claim']],
            ]
        ).reset_index(drop=True)
        source_to_claim_edge_list['relation'] = 'makes'
        return source_to_source_edge_list, source_to_claim_edge_list

    @abstractmethod
    async def create_source_claim_graph(self):
        pass
