from functools import cached_property
from typing import Annotated, Callable, Iterator
import requests
from bs4 import BeautifulSoup, PageElement, Tag
from answerbot.tools.markdown_document import MarkdownDocument
from dataclasses import dataclass, field
from markdownify import markdownify

from tenacity import retry, stop_after_attempt

from answerbot.tools.observation import Observation

import logging


MAX_RETRIES = 3
BASE_URL = 'https://artificialintelligenceact.eu/'
CHUNK_SIZE = 1024
MIN_CHUNK_SIZE = 100


@dataclass(frozen=True)
class DocumentSection:
    title: str
    children: list['DocumentSection|DocumentArticle']

    def find_article(self, id: str) -> 'DocumentArticle|None':
        for child in self.children:

            if isinstance(child, DocumentSection) and (found:=child.find_article(id)):
                return found
            if isinstance(child, DocumentArticle) and child.id == id:

                return child


    def to_string(self, depth: int=0) -> str:
        result = ('\t'*depth) + self.title
        if self.children:
            result += '\n' + '\n'.join(child.to_string(depth+1) for child in self.children)
        return result


@dataclass(frozen=True)
class DocumentArticle:
    title: str
    id: str
    retrieve: Callable[[], str]
    url: str


    def to_string(self, depth: int=0) -> str:
        return ('\t'*depth) + f'[{self.id}] {self.title}'    


class EUAIAct:
    def __init__(self, table_of_contents: DocumentSection, **_) -> None:
        self.table_of_contents = table_of_contents
        self._articles_shown: set[str] = set()

    def get_llm_tools(self) -> list[Callable]:
        return [self.show_table_of_contents, self.read_eu_ai_act_article_by_article_id]

    def show_table_of_contents(self):
        ''' Show EU artificial intelligence act's table of contents
        

        The table of contents is structured as follows:


        Document Title
          Chapter Title
            [article id] Article Title
            [article id] Article Title
        '''
        logging.getLogger(__name__).info('Opening table of contents')
        result = self.table_of_contents.to_string()
        return Observation(content='Opening table of contents\n\n' + result, operation='show_table_of_contents')

    def read_eu_ai_act_article_by_article_id(
            self, 
            article_id: Annotated[str, 'Article id from the table of contents']
            ):
        ''' Show an article from the EU Artificial intelligence act '''
        if article_id in self._articles_shown:
            return Observation(content='You have already seen this article', operation='read_eu_ai_act_article_by_article_id')
        logging.getLogger(__name__).info(f'Showing article with id: {article_id}')
        article = self.table_of_contents.find_article(article_id)
        if article is None:
            logging.getLogger(__name__).info(f'No article found with id: {article_id}')
            return Observation(content='No Article found with this id', operation='read_eu_ai_act_article_by_article_id')

        logging.getLogger(__name__).info(f'Found article with id: {article_id} ({article.url})')
        try:
            result = article.retrieve()
        except Exception as e:
            logging.getLogger(__name__).exception(e)
            return Observation(
                    content=f"Couldn't open the article due to exception: {e}",
                    operation='read_eu_ai_act_article_by_article_id'
                    )
        self._articles_shown.add(article_id)
        return Observation(
                content=f'Opening article: {article.title}\n\n{result}',
                source=article.url,
                operation='read_eu_ai_act_article_by_article_id',
                quotable=True
                )


def get_eu_act_toc(raw: list[dict], retrieve: Callable[[str, str, str], str]) -> DocumentSection:
    id_generator = map(str, range(1000000))
    result = convert_to_document_section(raw, id_generator, retrieve)
    return DocumentSection(
            title='EU Artificial Intelligence Act',
            children=result,
            )


def get_eu_act_toc_raw():

    r = requests.get('https://artificialintelligenceact.eu/ai-act-explorer/')
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, features='html.parser')
    div = soup.find('div', class_='aia-table-of-contents')
    parsed_data = []
    assert isinstance(div, Tag)

    accordion_headers = div.find_all('div', class_='accordion-header')

    for header in accordion_headers:
        parent_title = header.find('p', class_='parent-title')
        parent_data = parse_element(parent_title)

        accordion_content = header.find_next_sibling('div', class_='accordion-content')

        if accordion_content:
            parent_data['children'] = parse_accordion_content(accordion_content)
        parsed_data.append(parent_data)
    return parsed_data


def parse_element(element):
    title = element.a.text if element.a else element.text
    url = element.a['href'] if element.a else None
    return {'title': title, 'url': url, 'children': []}


def parse_accordion_content(content):
    children = []
    for p in content.find_all('p', class_='child-article'):
        children.append(parse_element(p))
    for p in content.find_all('p', class_='child-chapter'):
        chapter = parse_element(p)
        sub_content = p.find_next_sibling('div', class_='accordion-content')
        if sub_content:
            chapter['children'] = parse_accordion_content(sub_content)
        children.append(chapter)
    return children


def convert_to_document_section(parsed_data, id_generator, retrieve: Callable[[str, str, str], str]):

    children = []
    for item in parsed_data:
        title = item['title']
        url = item['url']
        child_elements = item.get('children', [])

        
        if child_elements:
            section = create_document_section(title, convert_to_document_section(child_elements, id_generator, retrieve))
            children.append(section)

        else:
            if url:
                article = create_document_article(title, url, id_generator, retrieve)
                children.append(article)

            else:
                section = create_document_section(title, [])

                children.append(section)
    return children


def create_document_article(title: str, url: str, id_generator: Iterator[str], retrieve: Callable[[str, str, str], str]) -> DocumentArticle:
    article_id = next(id_generator)
    def _retrieve():
        return retrieve(title, article_id, url)
    return DocumentArticle(title=title, id=article_id, retrieve=_retrieve, url=url)


def create_document_section(title: str, children: list[DocumentSection|DocumentArticle]) -> DocumentSection:
    return DocumentSection(title=title, children=children)


def retrieve(url: str) -> str:
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


    return '\n\n'.join(paragraphs)


def sanitize_string(src: str) -> str:
    return src\
        .replace('\n', '_')\
        .replace(' ', '-')\
        .replace(':','-')\
        .replace('/', '-')\
        .replace('\\', '-')