import asyncio
import logging
from typing import Optional

import pandas as pd

from src.claim_extraction.claim_extractor import ClaimExtractor
from src.utils.utils import log_event

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


async def loop(content_source_df: pd.DataFrame, claim_extractor: ClaimExtractor, prompt: str) -> tuple:
    # Don't loop on rows with no URL to avoid unnecessary LLM requests and runtime
    content_source_df = content_source_df.dropna(subset=['url'])
    tasks = [claim_extractor.create_source_claim_graph(source, prompt=prompt) for _, source in content_source_df.iterrows()]
    results = await asyncio.gather(*tasks)
    source_to_source_edge_list = [result[0] for result in results]
    source_to_claim_edge_list = [result[1] for result in results]
    claim_source_dfs = [result[2] for result in results]

    return (
        pd.concat(source_to_source_edge_list, ignore_index=True),
        pd.concat(source_to_claim_edge_list, ignore_index=True),
        pd.concat(claim_source_dfs, ignore_index=True),
    )


async def main(
    claim_extractor: ClaimExtractor,
    content_source_df: pd.DataFrame,
    prompt: str,
    hops: int = 2,
    write_file_path: Optional[str] = None,
) -> pd.DataFrame:
    source_to_source_edge_lists = []
    source_to_claim_edge_lists = []
    for i in range(hops):
        log_event(logger, logging.INFO, 'Starting hop', hop=i + 1, total_hops=hops)
        source_to_source_edge_list, source_to_claim_edge_list, claim_source_df = await loop(
            content_source_df=content_source_df,
            claim_extractor=claim_extractor,
            prompt=prompt,
        )

        content_source_df = claim_source_df[['source_id', 'url', 'tone']]
        source_to_source_edge_lists.append(source_to_source_edge_list)
        source_to_claim_edge_lists.append(source_to_claim_edge_list)

    source_to_source_edge_list = pd.concat(source_to_source_edge_lists, ignore_index=True)
    source_to_claim_edge_list = pd.concat(source_to_claim_edge_lists, ignore_index=True)

    if write_file_path:
        source_to_source_edge_list.to_csv(f'{write_file_path}/sts_edge_list.csv', index=False)
        source_to_claim_edge_list.to_csv(f'{write_file_path}/stc_edge_list.csv', index=False)
    return source_to_source_edge_list, source_to_claim_edge_list
