from unittest.mock import patch

import pandas as pd
import pytest

from src.claim_extraction import (
    apply_and_explode,
    chunk_text,
    create_edge_list,
    get_chunk_claims,
    get_claim_sources,
    get_text_from_source,
)
from src.exceptions import PydanticValidationError


@patch('src.claim_extraction._fetch_source_html')
@patch('src.claim_extraction._extract_text_from_html')
def test_get_text_from_source(mock_extract_text, mock_fetch_html, mock_content_source_df):
    mock_fetch_html.return_value = '<p>Example HTML content</p>'
    mock_extract_text.return_value = 'Example text content'
    result_df = get_text_from_source(mock_content_source_df)
    assert 'content_text' in result_df.columns
    assert result_df['content_text'][0] == 'Example text content'


@pytest.mark.parametrize(
    'column, func_return, new_columns, expected_df, expected_error, expected_error_message',
    [
        pytest.param(
            'content_text',
            [(12, 'this is a test article'), (21, 'it contains multiple sentences'), (15, 'here is another sentence')],
            ['chunk_id', 'chunk'],
            pd.DataFrame(
                {
                    'source_id': [1, 1, 1],
                    'content_text': ['This is a test article. It contains multiple sentences. Here is another sentence.'] * 3,
                    'chunk_id': [12, 21, 15],
                    'chunk': ['this is a test article', 'it contains multiple sentences', 'here is another sentence'],
                }
            ),
            None,
            None,
            id='Basic case with multiple tuples',
        ),
        pytest.param(
            'content_text',
            [],
            ['chunk_id', 'chunk'],
            pd.DataFrame(
                {
                    'source_id': [1],
                    'content_text': ['This is a test article. It contains multiple sentences. Here is another sentence.'],
                    'chunk_id': [None],
                    'chunk': [None],
                }
            ),
            None,
            None,
            id='Empty list doesn\'t explode',
        ),
        pytest.param(
            'content_text',
            [(12, 'this is a test article'), (21, 'it contains multiple sentences'), (15, 'here is another sentence')],
            ['chunk'],
            pd.DataFrame(
                {
                    'source_id': [1],
                    'content_text': ['This is a test article. It contains multiple sentences. Here is another sentence.'],
                    'chunk_id': [None],
                    'chunk': [None],
                }
            ),
            ValueError,
            "Function return length does not match new_columns length for column 'content_text'.",
            id='Invalid new_columns length',
        ),
        pytest.param(
            'content_text',
            [],
            ['chunk_id', 'content_text'],
            pd.DataFrame(
                {
                    'source_id': [1],
                    'content_text': ['This is a test article. It contains multiple sentences. Here is another sentence.'],
                    'chunk_id': [None],
                    'chunk': [None],
                }
            ),
            ValueError,
            'One or more new_columns already exist in DataFrame.',
            id='New columns already exist',
        ),
        pytest.param(
            'texting',
            None,
            ['chunk_id', 'chunk'],
            None,
            KeyError,
            "Column 'texting' does not exist in DataFrame.",
            id='Column does not exist',
        ),
    ],
)
def test_apply_and_explode(column, func_return, new_columns, expected_df, expected_error, expected_error_message, mock_text_df):
    def example_func(_):
        return func_return

    if expected_error:
        with pytest.raises(expected_error, match=expected_error_message):
            apply_and_explode(mock_text_df, column, example_func, new_columns)
    else:
        result_df = apply_and_explode(mock_text_df, column, example_func, new_columns)
        pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df.reset_index(drop=True))


@pytest.mark.parametrize(
    'input_text, split_return, expected_df, expected_error, expected_error_message',
    [
        pytest.param(
            'This is a test article. It contains multiple sentences. Here is another sentence.',
            [(12, 'this is a test article'), (21, 'it contains multiple sentences'), (15, 'here is another sentence')],
            pd.DataFrame(
                {
                    'text': ['This is a test article. It contains multiple sentences. Here is another sentence.'] * 3,
                    'chunk_id': [12, 21, 15],
                    'chunk': ['this is a test article', 'it contains multiple sentences', 'here is another sentence'],
                }
            ),
            None,
            None,
            id='Basic case with multiple chunks',
        ),
        pytest.param(
            '',
            [],
            pd.DataFrame({'text': [''], 'chunk_id': [None], 'chunk': [None]}),
            ValueError,
            'Chunk Text Input is empty.',
            id='Empty text results in empty chunks',
        ),
    ],
)
@patch('src.claim_extraction.uuid4')
@patch('src.claim_extraction.RecursiveCharacterTextSplitter')
def test_chunk_text(mock_splitter, mock_uuid, input_text, split_return, expected_df, expected_error, expected_error_message):
    mock_splitter.return_value.split_text.return_value = [chunk for _, chunk in split_return]
    mock_uuid.side_effect = [type('UUID', (), {'hex': uuid})() for uuid, _ in split_return]
    if expected_error:
        with pytest.raises(expected_error, match=expected_error_message):
            chunk_text(input_text, chunk_size=10, chunk_overlap=0)
    else:
        result_df = chunk_text(input_text, chunk_size=10, chunk_overlap=0)
        print(result_df)
        pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df.reset_index(drop=True))


@pytest.mark.parametrize(
    "input_df, llm_response, claim_ids, expected_df, expected_error, expected_error_message",
    [
        pytest.param(
            pd.DataFrame({'source_id': [1], 'chunk_id': ['chunk1'], 'chunk': ['Test chunk with two claims.']}),
            '{"subject": "test", "claims": ["claim1", "claim2"]}',
            ['id_claim1', 'id_claim2'],
            pd.DataFrame(
                {
                    'source_id': [1, 1],
                    'chunk_id': ['chunk1', 'chunk1'],
                    'chunk': ['Test chunk with two claims.', 'Test chunk with two claims.'],
                    'claim_id': ['id_claim1', 'id_claim2'],
                    'claim': ['claim1', 'claim2'],
                }
            ),
            None,
            None,
            id="Basic case with multiple claims",
        ),
        pytest.param(
            pd.DataFrame({'source_id': [1], 'chunk_id': ['chunk1'], 'chunk': ['Test chunk with no claims.']}),
            '{"subject": "test", "claims": []}',
            [],
            pd.DataFrame(
                {
                    'source_id': [1],
                    'chunk_id': ['chunk1'],
                    'chunk': ['Test chunk with no claims.'],
                    'claim_id': [None],
                    'claim': [None],
                }
            ),
            None,
            None,
            id="Case with no claims returned",
        ),
        pytest.param(
            pd.DataFrame({'source_id': [1], 'chunk_id': ['chunk1'], 'bad_chunk': ['Test chunk with no claims.']}),
            '{"subject": "test", "claims": []}',
            [],
            pd.DataFrame(
                {
                    'source_id': [1],
                    'chunk_id': ['chunk1'],
                    'chunk': ['Test chunk with no claims.'],
                    'claim_id': [None],
                    'claim': [None],
                }
            ),
            KeyError,
            "Column 'chunk' does not exist in DataFrame.",
            id="Missing 'chunk' column in input DataFrame",
        ),
        pytest.param(
            pd.DataFrame({'source_id': [], 'chunk_id': [], 'chunk': []}),
            '',
            [],
            pd.DataFrame(),
            ValueError,
            "Get Chunk Claims Input DataFrame is empty.",
            id="Empty input DataFrame",
        ),
        pytest.param(
            pd.DataFrame({'source_id': [1], 'chunk_id': ['chunk1'], 'chunk': ['Test chunk with bad response.']}),
            'this is not valid json',
            [],
            None,
            PydanticValidationError,
            "Error parsing model response when extracting claims",
            id="LLM returns invalid JSON",
        ),
    ],
)
@patch('src.claim_extraction.uuid4')
@patch('src.claim_extraction.chat')
def test_get_chunk_claims(
    mock_chat, mock_uuid, input_df, llm_response, claim_ids, expected_df, expected_error, expected_error_message
):
    mock_chat.return_value.message.content = llm_response
    mock_uuid.side_effect = [type('UUID', (), {'hex': claim_id})() for claim_id in claim_ids]

    if expected_error:
        with pytest.raises(expected_error, match=expected_error_message):
            get_chunk_claims(input_df)
    else:
        result_df = get_chunk_claims(input_df)
        pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df.reset_index(drop=True))


@pytest.mark.parametrize(
    "input_df, search_return, source_ids, expected_df, expected_error, expected_error_message",
    [
        pytest.param(
            pd.DataFrame({'claim_id': ['claim1'], 'claim': ['Test claim.']}),
            pd.DataFrame(
                {
                    'warning': [''],
                    'query_with_commands': ['query'],
                    'query_without_commands': ['query'],
                    'url': ['http://example.com'],
                }
            ),
            ['source1'],
            pd.DataFrame(
                {
                    'claim_id': ['claim1'],
                    'claim': ['Test claim.'],
                    'source_id': ['source1'],
                    'url': ['http://example.com'],
                    'query_with_commands': ['query'],
                    'query_without_commands': ['query'],
                    'warning': [''],
                }
            ),
            None,
            None,
            id="Basic case with one claim and one source",
        ),
        pytest.param(
            pd.DataFrame({'claim_id': ['claim1'], 'claim': ['Test claim with no results.']}),
            pd.DataFrame(
                {'warning': ['No_Results'], 'query_with_commands': ['query'], 'query_without_commands': ['query'], 'url': [None]}
            ),
            ['source1'],
            pd.DataFrame(
                {
                    'claim_id': ['claim1'],
                    'claim': ['Test claim with no results.'],
                    'source_id': ['source1'],
                    'url': [None],
                    'query_with_commands': ['query'],
                    'query_without_commands': ['query'],
                    'warning': ['No_Results'],
                }
            ),
            None,
            None,
            id="Case where GDELT returns no results",
        ),
        pytest.param(
            pd.DataFrame({'claim_id': ['claim1'], 'claim': ['Test claim with no results.']}),
            pd.DataFrame(
                {
                    'warning': ['GDELT_API_Error'],
                    'query_with_commands': ['query'],
                    'query_without_commands': ['query'],
                    'url': [None],
                }
            ),
            ['source1'],
            pd.DataFrame(
                {
                    'claim_id': ['claim1'],
                    'claim': ['Test claim with no results.'],
                    'source_id': ['source1'],
                    'url': [None],
                    'query_with_commands': ['query'],
                    'query_without_commands': ['query'],
                    'warning': ['GDELT_API_Error'],
                }
            ),
            None,
            None,
            id="Case where GDELT returns no results",
        ),
    ],
)
@patch('src.claim_extraction.uuid4')
@patch('src.claim_extraction.full_text_search')
def test_get_claim_sources(
    mock_full_text_search, mock_uuid, input_df, search_return, source_ids, expected_df, expected_error, expected_error_message
):
    if expected_error:
        with pytest.raises(expected_error, match=expected_error_message):
            get_claim_sources(input_df)
    else:
        mock_full_text_search.return_value = search_return
        mock_uuid.side_effect = [type('UUID', (), {'hex': source_id})() for source_id in source_ids]
        result_df = get_claim_sources(input_df)
        pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df.reset_index(drop=True))


@pytest.mark.parametrize(
    "chunk_claims_data, claim_source_data, expected_df",
    [
        pytest.param(
            {'source_id': [1], 'claim_id': ['claim123'], 'claim': ['Claim Text']},
            {'claim_id': ['claim123'], 'source_id': [2], 'url': ['http://example.com']},
            pd.DataFrame(
                {
                    'source_id_source': [1],
                    'source_id_target': [2],
                    'claim_id': ['claim123'],
                    'claim': ['Claim Text'],
                    'url': ['http://example.com'],
                    'claim_info': [('claim123', 'Claim Text')],
                }
            ),
            id="Basic case with one match",
        ),
        pytest.param(
            {'source_id': [1], 'claim_id': ['claim123'], 'claim': ['Claim Text']},
            {'claim_id': ['claim456'], 'source_id': [2], 'url': ['http://example.com']},
            pd.DataFrame(
                {'source_id_source': [], 'source_id_target': [], 'claim_id': [], 'claim': [], 'url': [], 'claim_info': []}
            ),
            id="No matching claim_id results in empty DataFrame",
        ),
        pytest.param(
            {'source_id': [], 'claim_id': [], 'claim': []},
            {'claim_id': [], 'source_id': [], 'url': []},
            pd.DataFrame(
                {'source_id_source': [], 'source_id_target': [], 'claim_id': [], 'claim': [], 'url': [], 'claim_info': []}
            ),
            id="Empty input dataframes result in empty DataFrame",
        ),
    ],
)
def test_create_edge_list(chunk_claims_data, claim_source_data, expected_df):
    chunk_claims_df = pd.DataFrame(chunk_claims_data)
    claim_source_df = pd.DataFrame(claim_source_data)
    result_df = create_edge_list(chunk_claims_df, claim_source_df)
    # The merge operation can change column order, so we sort them to ensure the assertion is stable.
    result_df = result_df.reindex(sorted(result_df.columns), axis=1)
    expected_df = expected_df.reindex(sorted(expected_df.columns), axis=1)
    pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=False)
