from typing import Annotated
from bs4 import BeautifulSoup, Tag
from markdownify import markdownify

import requests

from answerbot.reactor import (
        LLMReactor, 
        OpenDocument, 
        ReactorResponse, 
        llm_function, 
        DEFAULT_SYSTEM_PROMPT, 
        opens_document, 
        cost
        )
from llm_easy_tools.tool_box import llm_function
import json


def format_table_of_contents() -> str:
    with open('data/aia_dump.json', 'r', encoding='utf-8') as f:
        obj = json.load(f)

    result = ''

    toc = obj['table_of_contents']

    for item in toc:
        if item[2]:
            result += f'{item[1]}\n'
            for si in item[2]:
                if si[2]:
                    result += f'  {si[1]}\n'
                    for si_ in si[2]:
                        result += f'    [{si_[1]}]({si_[0]})\n'
                else:
                    result += f'  [{si[1]}]({si[0]})\n'
        else:
            result += f'[{item[1]}]({item[0]})\n'
    return result


TOC_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT + '''

You can use links from this table of contents of the EU Artificial Intelligence Act:

Table of Contents:
''' + format_table_of_contents()


class AiActReactor(LLMReactor):
    SYSTEM_PROMPT = TOC_SYSTEM_PROMPT
    DOMAIN = 'EU Artificial Intelligence Act'

    @llm_function()
    @opens_document
    @cost(20)
    def goto_url_read_paragraphs(
            self,
            url: Annotated[str, 'The url to go to']
            ) -> OpenDocument:
        '''
        Read a page located at EU Artificial Intelligence Act explorer websiste.
        Only urls on https://artificialintelligenceact.eu/ are accepted.
        '''
        response = requests.get(url)
        response.raise_for_status()
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')

        post_content = soup.find('div', class_='et_pb_post_content')
        if not isinstance(post_content, Tag):
            raise RuntimeError('No information was found in this article')

        paragraphs = []

        for p in post_content.find_all('p'):
            if not isinstance(p, Tag):
                continue
            paragraphs.append(markdownify(str(p)))

        
        recitals_grid = soup.find('div', class_='recitals-grid')
        if not recitals_grid or not isinstance(recitals_grid, Tag):
            return OpenDocument(url, paragraphs)

        recitals = []

        for a in recitals_grid.find_all('a'):
            if not isinstance(recitals_grid, Tag):
                continue
            try:
                recitals.append((a['href'], a.div.p.text))
            except (KeyError, AttributeError):
                continue

        if recitals:
            paragraphs.append(
                    '# Recitals Relevant to this document:\n' + 
                    '\n'.join(f'[{text}]({url})' for url, text in recitals)
                    )

        return OpenDocument(url, paragraphs)


def format_results(response: ReactorResponse) -> str:
    relevant_paragraphs = '\n\n'.join(f'({p.url})\n{p.paragraph}' for p in response.relevant_paragraphs)
    followup_assumptions = '\n'.join(response.followup_assumptions)
    result = (
            '#Question\n'
            f'{response.question}\n\n'
            '#Assumptions\n'
            f'{response.assumptions}\n\n'
            '#Answer\n'
            f'{response.answer.answer}\n\n'
            '#Reasoning\n'
            f'{response.answer.reasoning}\n\n'
            '#Relevant paragraphs\n'
            f'{relevant_paragraphs}'
            '#Followup assumptions\n'
            f'{followup_assumptions}'
            )
    return result


