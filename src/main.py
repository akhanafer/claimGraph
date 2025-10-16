import asyncio
import logging
from typing import Optional

import pandas as pd
from openai import AsyncOpenAI

from src.claim_extraction.claim_extractor import ClaimExtractor
from src.claim_extraction.news_articles.gdelt import GDELTClaimExtractor
from src.utils.utils import log_event

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


async def loop(content_source_df: pd.DataFrame, claim_extractor: ClaimExtractor):
    # Don't loop on rows with no URL to avoid unnecessary LLM requests and runtime
    content_source_df = content_source_df.dropna(subset=['url'])
    tasks = [claim_extractor.create_source_claim_graph(source) for _, source in content_source_df.iterrows()]
    results = await asyncio.gather(*tasks)
    source_to_source_edge_list = [result[0] for result in results]
    source_to_claim_edge_list = [result[1] for result in results]
    claim_source_dfs = [result[2] for result in results]

    return (
        pd.concat(source_to_source_edge_list, ignore_index=True),
        pd.concat(source_to_claim_edge_list, ignore_index=True),
        pd.concat(claim_source_dfs, ignore_index=True),
    )


async def main(content_source_df: pd.DataFrame, hops: int = 2, prompt: Optional[str] = None) -> pd.DataFrame:
    ollama_openai_client = AsyncOpenAI(base_url='http://localhost:11434/v1', api_key='dummy')

    gdelt_claim_extractor = GDELTClaimExtractor(
        openai_client=ollama_openai_client,
        claim_model='mistral:7b',
        structured_output_model=None,
        text_to_gdelt_query_model='mistral:7b',
        text_to_gdelt_query_max_retry=1,
        chunk_size=1000,
        chunk_overlap=100,
    )
    source_to_source_edge_lists = []
    source_to_claim_edge_lists = []
    for i in range(hops):
        log_event(logger, logging.INFO, 'Starting hop', hop=i + 1, total_hops=hops)
        source_to_source_edge_list, source_to_claim_edge_list, claim_source_df = await loop(
            content_source_df=content_source_df,
            claim_extractor=gdelt_claim_extractor,
        )
        content_source_df = claim_source_df[['source_id', 'url']]
        source_to_source_edge_lists.append(source_to_source_edge_list)
        source_to_claim_edge_lists.append(source_to_claim_edge_list)

    source_to_source_edge_list = pd.concat(source_to_source_edge_lists, ignore_index=True)
    source_to_claim_edge_list = pd.concat(source_to_claim_edge_lists, ignore_index=True)

    source_to_source_edge_list.to_csv(f'storage/sts_edge_list_{prompt.strip().replace(" ", "_")}_{hops}_hop.csv', index=False)
    source_to_claim_edge_list.to_csv(f'storage/stc_edge_list_{prompt.strip().replace(" ", "_")}_{hops}_hop.csv', index=False)
    return source_to_source_edge_list, source_to_claim_edge_list


if __name__ == '__main__':
    articles_pd = pd.DataFrame(
        {
            'source_id': [1],
            'url': ['https://www.middleeasteye.net/news/poll-finds-60-percent-americans-oppose-military-aid-israel'],
        }
    )

    hops = 1
    source_to_source_edge_list, source_to_claim_edge_list = asyncio.run(
        main(articles_pd, hops=hops, prompt="American Opinion on Israel Military Aid")
    )
