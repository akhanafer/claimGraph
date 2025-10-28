import asyncio
from typing import Optional

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity


class ClaimEmbeddingClient:
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        embedding_model: str = 'nomic-embed-text',
    ):
        self.openai_client = openai_client
        self.embedding_model = embedding_model

    async def embed_claims(self, claims_df: pd.DataFrame, write_file_path: Optional[str] = None) -> pd.DataFrame:
        embeddings = await asyncio.gather(
            *[self.openai_client.embeddings.create(input=claim, model=self.embedding_model) for claim in claims_df['claim']]
        )
        claims_df['claim_embedding'] = [embedding.data[0].embedding for embedding in embeddings]
        claims_df.to_pickle(f'{write_file_path}/claim_embeddings.pkl')
        return claims_df

    async def embed_user_query(self, query: str):
        query_embedding = await self.openai_client.embeddings.create(input=query, model=self.embedding_model)
        return query_embedding.data[0].embedding

    async def get_similar_claims(self, query: str, claims_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
        query_embedding = np.array(await self.embed_user_query(query)).reshape(1, -1)
        claim_embeddings = np.array(claims_df['claim_embedding'].values.tolist())

        similarities = cosine_similarity(query_embedding, claim_embeddings).flatten()
        claims_df['similarity'] = similarities
        top_claims = claims_df.nlargest(top_k, 'similarity')
        return top_claims
