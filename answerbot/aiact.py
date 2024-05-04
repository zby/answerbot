from typing import Annotated, Any
from dataclasses import dataclass
from bs4 import BeautifulSoup, Tag
from markdownify import markdownify

import requests

from answerbot.reactor import (
        LLMReactor, 
        OpenDocument, 
        ReactorResponse, 
        llm_function, 
        DEFAULT_SYSTEM_PROMPT, 
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

@dataclass(frozen=True)
class RelevantParagraph:
    url: str
    paragraph: str

@dataclass(frozen=True)
class OpenDocument:
    '''
    url: url of the document
    paragraphs: list of paragraphs in the document
    '''
    url: str
    paragraphs:list[str]



class AiActTool:
    READ_PARAGRAPHS_COST = 10

    def __init__(self, 
                 paragraph_size: int=55):
        self.relevant_paragraphs: list[RelevantParagraph] = []
        self._document: OpenDocument|None = None
        self._paragraph_size=paragraph_size
        self.relevant_paragraphs: list[RelevantParagraph] = []


    @llm_function()
    @cost(20)
    def goto_url_read_paragraphs(
            self,
            url: Annotated[str, 'The url to go to']
            ) -> str:
        '''
        Read a page located at EU Artificial Intelligence Act explorer websiste.
        Only urls on https://artificialintelligenceact.eu/ are accepted.
        '''
        result = self.get_document(url)
        self._document = result
        return 'Paragraphs in the current document:\n' + '\n'.join(
                f'{i}: {s[:self._paragraph_size]}' for i, s in enumerate(result.paragraphs)
                )

    def get_document(
            self,
            url: Annotated[str, 'The url to go to']
            ) -> OpenDocument:
 
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

    @llm_function()
    @cost(10)
    def read_paragraphs(
            self,
            paragraphs: Annotated[list[int], 'The indices of paragraphs to read']
            ):
        ''' Return the full text of paragraphs at specified indices '''
        if not self._document:
            raise RuntimeError('No document is currently open')
        result = ''
        for index in paragraphs:
            try:
                p = self._document.paragraphs[index]
            except IndexError:
                continue
            result += f'# Paragraph {index}:\n'
            result += p + '\n'
        return result

        
    @llm_function()
    def reflect(
            self,
            paragraphs: Annotated[list[int], 'The indices of paragraphs you found to be relevant AND did contain the required information']
            ):
        assert self._document is not None
        for index in paragraphs:
            try:
                self.relevant_paragraphs.append(
                        RelevantParagraph(
                            url=self._document.url,
                            paragraph=self._document.paragraphs[index]
                            )
                        )
            except IndexError:
                continue
        return str(paragraphs)


def format_results(reactor: LLMReactor) -> str:
    ai_act_tool_object = reactor._toolbox._tool_sets['AiActTool']
    rp = ai_act_tool_object.relevant_paragraphs
    relevant_paragraphs = '\n\n'.join(f'({p.url})\n{p.paragraph}' for p in rp)
    followup_assumptions = '\n'.join(reactor.followup_assumptions)
    result = (
            '#Question\n'
            f'{reactor.question}\n\n'
            '#Assumptions\n'
            f'{ai_act_tool_object.assumptions}\n\n'
            '#Answer\n'
            f'{ai_act_tool_object.answer}\n\n'
            '#Reasoning\n'
            f'{ai_act_tool_object.reasoning}\n\n'
            '#Relevant paragraphs\n'
            f'{relevant_paragraphs}'
            '#Followup assumptions\n'
            f'{followup_assumptions}'
            )
    return result


def run_reactor(question: str, client: Any, model: str, energy: int) -> ReactorResponse:
    ai_act_tool = AiActTool()
    reactor = LLMReactor(system_prompt=TOC_SYSTEM_PROMPT, model=model, client=client, question=question, toolbox=ai_act_tool, energy=energy)
    return reactor()

