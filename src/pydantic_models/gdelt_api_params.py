from typing import Literal, Optional

from pydantic import BaseModel, Field


class FullTextSearchQueryCommands(BaseModel):
    domain: Optional[str] = Field(
        default=None,
        description='This allows you to restrict your search to a particular news outlet.'
        'Specifying an outlet like “cnn.com” will return only matching coverage from CNN',
    )

    tonemorethan: Optional[int] = Field(
        default=None,
        description='This returns only coverage with a sentiment/tone score greater than'
        '(happier than) the given score. This uses the base GDELT Tone score, which offers a'
        'general purpose tonal indicator that ranges from -100 (extremely sad) to 100 (extremely happy).'
        'In practice, most tone scores range between -10 and 10, with numbers closer to 0 being more neutral.'
        'You may find that you must adjust this value for different searches to get the best results.',
    )

    tonelessthan: Optional[int] = Field(default=None, description='Same as tonemorethan except filters for sadness')

    lastminutes: int = Field(
        default=4320,
        description='By default the API searches the last 24 hours of monitored coverage.'
        'You can use this option to restrict the search to the last X minutes',
    )

    sourcelang: Literal['eng', 'fra'] = Field(default='eng')
    sortby: Literal['rel', 'date'] = Field(default='date')


class FullTextSearchParams(BaseModel):
    query: str = Field(description='The actual query string to be searched on')

    output: Literal['urllist'] = Field(default='urllist', description='The type of output that should be produced')

    maxrows: int = Field(
        default=10,
        description='Configures how many records should be returned. For article list outputs, '
        'this controls how many articles are shown in the results',
    )

    dropdup: bool = Field(
        default=True,
        description='Performs very basic deduplication in an attempt to filter out duplicate articles,'
        'such as wire stories that were republished by large numbers of news outlets. Normally disabled'
        'for maximum search speed, but can be enabled if your search frequently generates large numbers of'
        'duplicative results',
    )
