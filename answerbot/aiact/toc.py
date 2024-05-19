from typing import Iterator
from bs4 import BeautifulSoup, Tag
import requests
from answerbot.subtasks.table_of_contents import DocumentArticle, DocumentSection
from markdownify import markdownify


def get_eu_act_toc() -> DocumentSection:
    raw = get_eu_act_toc_raw()
    id_generator = map(str, range(1000000))
    result = convert_to_document_section(raw, id_generator)
    return DocumentSection(
            title='EU Artificial Intelligence Act',
            children=result,
            )


def get_eu_act_toc_raw():
    r = requests.get('https://artificialintelligenceact.eu/ai-act-explorer/')
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html)
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


def convert_to_document_section(parsed_data, id_generator):
    children = []
    for item in parsed_data:
        title = item['title']
        url = item['url']
        child_elements = item.get('children', [])
        
        if child_elements:
            section = create_document_section(title, convert_to_document_section(child_elements, id_generator))
            children.append(section)
        else:
            if url:
                article = create_document_article(title, url, id_generator)
                children.append(article)
            else:
                section = create_document_section(title, [])
                children.append(section)
    return children

def create_document_article(title: str, url: str, id_generator: Iterator[str]) -> DocumentArticle:
    article_id = next(id_generator)
    def _retrieve():
        return retrieve(url)
    return DocumentArticle(title=title, id=article_id, retrieve=_retrieve)


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
