import numpy as np
import ollama
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def embed_claims(claims_df: pd.DataFrame, embedding_model: str = 'nomic-embed-text'):
    embeddings = claims_df['claim'].apply(lambda claim: ollama.embeddings(prompt=claim, model=embedding_model).embedding)
    claims_df['claim_embedding'] = embeddings
    claims_df['claim_embedding'] = claims_df['claim_embedding']
    return claims_df


def embed_user_query(query: str, embedding_model: str = 'nomic-embed-text'):
    query_embedding = ollama.embeddings(prompt=query, model=embedding_model).embedding
    return query_embedding


def get_similar_claims(query: str, claims_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    query_embedding = np.array(embed_user_query(query)).reshape(1, -1)
    claim_embeddings = np.array(claims_df['claim_embedding'].values.tolist())

    similarities = cosine_similarity(query_embedding, claim_embeddings).flatten()
    claims_df['similarity'] = similarities
    top_claims = claims_df.nlargest(top_k, 'similarity')
    return top_claims
