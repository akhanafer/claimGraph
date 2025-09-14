from typing import Literal, Optional

from pydantic import BaseModel, Field


class FullTextSearchQueryCommands(BaseModel):
    domain: Optional[str] = Field(
        default=None,
        description='This allows you to restrict your search to a particular news outlet.'
        'Specifying an outlet like “cnn.com” will return only matching coverage from CNN',
    )
    domain_exclude: Optional[str] = Field(
        default=None,
        description='This allows you to exclude a particular news outlet from your search.'
        'Specifying an outlet like “cnn.com” will return matching coverage from all outlets'
        'except CNN',
    )
    sourcelang: Literal['english', 'french'] = Field(default='english')
    theme: Optional[str] = Field(
        default=None,
        description='Searches for any of the GDELT Global Knowledge Graph (GKG) Themes. GKG'
        'Themes offer a more powerful way of searching for complex topics, since they can include'
        'hundreds or even thousands of different phrases or names under a single heading.',
    )


class FullTextSearchParams(BaseModel):
    query: Optional[str] = Field(description='The actual query string to be searched on')

    mode: Literal['artlist'] = Field(
        default='artlist',
        description='Specifies the specific output you would like from the API, ranging from timeline'
        ' to word clouds to article lists',
    )

    format: Literal['HTML', 'CSV', 'JSON', 'JSONFeed'] = Field(
        default='CSV',
        description='This controls what file format the results are displayed in. Not all formats' 'are available for all modes.',
    )

    timespan: Optional[str] = Field(
        default=None,
        description=' By default the DOC API searches the last 3 months of coverage monitored by GDELT.'
        'You can narrow this range by using this option to specify the number of months, weeks, days,'
        'hours or minutes (minimum of 15 minutes)',
        examples=['4min', '1d', '1w', '4m'],
    )

    maxrecords: int = Field(
        default=10,
        description='Configures how many records should be returned. For article list outputs, '
        'this controls how many articles are shown in the results',
    )

    sort: Optional[Literal['datedesc', 'dateasc', 'hybridrel']] = Field(
        default='hybridrel', description='How to sort the results. Defaults to relevance'
    )
