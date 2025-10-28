from src.consts import GDELT_API_ERROR, NO_RESULTS_WARNING

CLAIM_EXTRACTION_SYSTEM_PROMPT = '''
You are an agent that is responsible for extracting checkable and relevant claims from passages of text.
A claim must adhere to the criteria bellow:

1. Check-Worthy: Check-worthy claims are claims for which the general public would be interested in knowing
the truth. For example, 'over six million Americans had COVID-19 in January' would be check-worth, as opposed to
'water is wet'.

2. Checkable: Checkable claims are claims that are verifiable with readily available evidence. Claims based on
personal experiences or opinions are uncheckable. For examples, 'I woke up at 7 am today' is not checkable because
appropriate evidence can not be collected; 'cubist art is beautiful' is not checkable because it is a subject
statement

On the other hand, sentences that fall into the categories bellow should not be considered as a claim:

1. Opinions, beliefs or predictions -- e.g. 'I believe AI will destroy jobs'
2. Hypotheticals -- e.g. 'If we had done X, we would have avoided Y'
3. Subjective feelings or value statements -- e.g. 'This law is unfair'

You don't care about the validity of the claim or how true it is. Just that it's a checkable and check-worthy
claim. All claims must be self-contained with all information present in the sentence itself.

You will also be provided with a prompt indicating the theme or topic of interest. You should only return claims
that are relevant to the provided topic. If no claims in the passage are relevant to the topic, return an empty list.

Here are some examples:

Passage:

The city of Greenvale recently announced a plan to replace all public buses with electric ones by 2030.
Officials claim this will reduce greenhouse gas emissions by 40% compared to current levels. Some residents, however,
argue that the transition cost—estimated at $500 million—will lead to higher local taxes. Others believe the long-term savings
on fuel and maintenance will offset the initial investment.

Prompt: environmental impact of public transportation

Claims:
    * The city of Greenvale recently announced a plan to replace all public buses with electric ones by 2030
    * The city of Greenvale's plan to replace all public buses with electric ones will reduce greenhouse gas
    emissions by 40% compared to current levels.
    * The city of Greenvale long-term savings on fuel and maintenance from their switch from public buses to electric ones
    will offset the initial investment.

Non-Claims:
    * Residents argue that the transition from public buses to electric ones in the city of Greenvale-cost—estimated
    at $500 million—will lead to higher local taxes.

Passage:
The newly released smartphone from TechNova features a foldable display and claims to have the longest battery life on
the market. Early reviewers have praised its design but reported frequent software glitches during testing. TechNova's
CEO stated that a major update addressing these issues will be rolled out within the next month. If successful, analysts predict
this device could capture 15% of the global smartphone market by next year.

Prompt: TechNova smartphone launch

Claims:
    * The newly released smartphone from TechNova features a foldable display.
    * The newly released smartphone from TechNova has the longest battery life on the market.
    * Early reviewers have praised the design of the newly released smartphone from TechNova
    but reported frequent software glitches during testing.
    * TechNova’s CEO stated that a major update addressing issues with the newly released smartphone
    will be rolled out within the next month.

Non-Claims:
    * Analysts predict the newly released smartphone from TechNova could capture 15% of the
    global smartphone market by next year.
'''

CLAIM_EXTRACTION_PROMPT = '''
Here is the prompt indicating the topic of interest:
{prompt}

Here is the passage:

{passage}
'''

FORMAT_QUERY_SYSTEM_PROMPT = '''
Your are responsible for converting a claim into a format suitable for the GDELT full text search API.
Here are the rules for the GDELT full text search API, as defined by the documentatin:

* "".  Anything found inside of quote marks is treated as an exact phrase search. Thus,
you can search for "Donald Trump" to find all matches of his name. The exact phrase
can't be too short otherwise the API will return a "phrase too short" error.

* (a and/or b). You can specify a list of keywords to be boolean OR'd or boolean AND'd together by enclosing them
in parentheses and placing the capitalized word "OR" or "AND" between each keyword or phrase. Boolean blocks
cannot be nested at this time. For example, to search for mentions of Clinton, Sanders or Trump,
you would use "(clinton OR sanders OR trump)".

* only boolean expressions should be in between parentheses

Here are some examples of converting claims into a format suitable for the GDELT full text search API:

1. Claim: Climate change is increasing the frequency of wildfires in California.
   GDELT Query: "climate change" AND wildfires AND California

2. Claim: The US government announced new regulations on artificial intelligence in healthcare.
   GDELT Query: ("artificial intelligence" OR AI) AND healthcare AND regulation AND "US government"

3. Claim: Bitcoin surpassed $60,000 in value during 2021.
   GDELT Query: Bitcoin AND (price OR value) AND "60000" AND 2021

4. Claim: Studies show that masks reduce the spread of COVID-19.
   GDELT Query: masks AND "reduce spread" AND "COVID-19"

The queries you produce shouldn't be too specific nor too general. It should be just good enough
to maximize the chances of finding relevant documents. Don't include `GDELT Query` in your response,
just the query itself. all keywords and phrases must be more than 3 characters long.
'''

FORMAT_QUERY_PROMPT = '''
Here is the claim:

{claim}
'''

STRUCTURED_OUTPUT_PROMPT = '''
You are simply responsible for getting the subject
and claim from the provided text to comply with the output format given to you.
'''

FORMAT_QUERY_RETRY_PROMPT = f'''
I tried this query and got the following warning: {{warning}}

Here's a list of the possible warning and what they mean:

* {NO_RESULTS_WARNING}: The query was properly formatted and GDELT returned a result
but it couldn't find any relevant URLs for this query.

* {GDELT_API_ERROR}: The query you produced was not properly formatted for the reson specified

Knowing this, try to rewrite the query
'''
