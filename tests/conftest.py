import pandas as pd
import pytest


@pytest.fixture
def mock_content_source_df():
    return pd.DataFrame(
        {'source_id': [1], 'timestamp': ['20250824151500'], 'language': ['en'], 'url': ['http://example.com/news']}
    )


@pytest.fixture
def mock_text_df():
    return pd.DataFrame(
        {
            'source_id': [1],
            'content_text': ['This is a test article. It contains multiple sentences. Here is another sentence.'],
        }
    )
