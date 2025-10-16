import logging

import pandas as pd
from ollama import chat

from src.utils.utils import log_event

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

SYSTEM_PROMPT = '''
You are an agent responsible for providing detailed information about a research topic,
claim, or question based on a user input. Your task is to generate a comprehensive report
summary that includes relevant claims, sources, and a synthesized overview of the topic.

You will be given a comma separated list of claims and their associated sources and relevancy score.
Each claim may have multiple sources. Below is an example of the input format:

url,claim,similarity
https://www.middleeasteye.net/news/poll-finds-60-percent-americans-oppose-military-aid-israel,Asked where their sympathies lie, 37 percent of US voters said the Palestinians.,0.9657629800072585 # noqa E501
https://www.middleeasteye.net/news/poll-finds-60-percent-americans-oppose-military-aid-israel,While 36 percent said the Israelis, and 27 percent said they had no opinion.,0.8231273459489017 # noqa E501
https://www.middleeasteye.net/news/poll-finds-60-percent-americans-oppose-military-aid-israel,Pollsters at Quinnipiac University found that 60 percent of Americans are opposed to the US sending arms to Israel.,0.7809814805716445 # noqa E501
https://www.middleeasteye.net/news/poll-finds-60-percent-americans-oppose-military-aid-israel,Thirty-five percent of voters do not think Israel is committing genocide in the Gaza Strip.,0.7780150788116804 # noqa E501
https://www.middleeasteye.net/news/poll-finds-60-percent-americans-oppose-military-aid-israel,Fifty percent of voters think Israel is committing genocide in the Gaza Strip.,0.7629918000807852 # noqa E501
https://www.middleeasteye.net/news/poll-finds-60-percent-americans-oppose-military-aid-israel,Only 32 percent of Americans support additional aid from the US to Israel.,0.7553649634581567 # noqa E501

Sources should be cited and listed in a "Sources" section at the end of the report.
The Sources section should include the URLs of the sources used in the report in markdown format.
'''

PROMPT = '''
Generate a report summary based on the following claims and sources:

Claims and Sources:
{claims_data}

User Prompt:
{user_prompt}

'''


def generate_report_summary(user_prompt: str, claims_data: pd.DataFrame, model: str = 'gpt-oss:20b') -> str:
    user_prompt = PROMPT.format(claims_data=claims_data.to_csv(index=False), user_prompt=user_prompt)
    log_event(logger, logging.INFO, 'generate_report_summary', system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt, model=model)

    response = chat(
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt},
        ],
        model=model,
    )

    return response.message.content
