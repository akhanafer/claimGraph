from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ollama import chat
from pydantic import BaseModel

from src.prompts import CLAIM_EXTRACTION_PROMPT, CLAIM_EXTRACTION_SYSTEM_PROMPT


class TextPassage(BaseModel):
    subject: str
    claims: List[str]


def get_article_text(article_links_pd: pd.DataFrame, keep_html_col: bool = False) -> pd.DataFrame:
    article_links_pd['raw_html'] = article_links_pd['url'].apply(_fetch_article_html)
    article_links_pd['article_text'] = article_links_pd['raw_html'].apply(_extract_article_text)
    if not keep_html_col:
        article_links_pd = article_links_pd.drop(columns=['raw_html'])
    return article_links_pd


def chunk_article(article_text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(article_text)


def extract_claims_from_chunk(chunk: str, model: str = 'gpt-oss:20b') -> TextPassage:
    response = chat(
        messages=[
            {'role': 'system', 'content': CLAIM_EXTRACTION_SYSTEM_PROMPT},
            {'role': 'user', 'content': CLAIM_EXTRACTION_PROMPT.format(passage=chunk)},
        ],
        model=model,
        format=TextPassage.model_json_schema(),
    )

    passage_claims = TextPassage.model_validate_json(response.message.content)
    return passage_claims


def _fetch_article_html(url: str) -> str:
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    return ''


def _extract_article_text(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, 'html.parser')
    for script in soup(['script', 'style']):
        script.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split('  '))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

def main(
       articles_pd: pd.DataFrame,
       model: str,
       chunk_size: int = 1000,
       chunk_overlap: int = 100
):
    articles_pd = get_article_text(articles_pd)
    articles_pd['chunk'] = articles_pd['article_text'].apply(
        lambda x: chunk_article(x, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    )
    articles_pd = articles_pd.explode('chunk').reset_index(drop=True)
    articles_pd['claims'] = articles_pd['chunk'].apply(extract_claims_from_chunk, model=model)
    return articles_pd

if __name__ == '__main__':
    articles_pd = pd.DataFrame({
        'timestamp': ['20250824151500'],
        'language': ['en'],
        'url': [
            'https://tribune.com.pk/story/2562808/famine-in-gaza'
        ],
    })

    articles_pd = main(articles_pd, model='mistral:7b')
    articles_pd.to_csv('article_claims.csv')