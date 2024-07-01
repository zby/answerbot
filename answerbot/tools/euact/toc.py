from typing import TypedDict
import requests
from bs4 import BeautifulSoup, Tag


class TocEntry(TypedDict):
    title: str
    url: str|None
    children: list['TocEntry']
    hidden: bool


def iter_toc(toc: TocEntry):
    yield toc
    for child in toc['children']:
        yield from iter_toc(child)


def fetch_toc() -> TocEntry:
    r = requests.get('https://artificialintelligenceact.eu/ai-act-explorer/')
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, features='html.parser')

    entries = []

    for header in soup.select('div.aia-table-of-contents > div.accordion-header'):
        parent_data_html = header.find('p', class_='parent-title')
        assert isinstance(parent_data_html, Tag)
        parent_data = _fetch_toc_parse_element(parent_data_html)

        accordion_content = header.find_next_sibling('div', class_='accordion-content')

        if isinstance(accordion_content, Tag):
            parent_data['children'] = _fetch_toc_parse_accordion_content(accordion_content)

        entries.append(parent_data)

    annexes = []

    for annex_link in soup.select('div.annexes-list > a'):
        if not isinstance(url := annex_link['href'], str):
            continue
        if not isinstance(p:=annex_link.select_one('div.annex-wrapper > p.annex'), Tag):
            continue
        annexes.append({
            'title': p.text.strip().replace('\n', ' '),
            'url': url,
            'children': [],
            'hidden': False,
            })

    entries.append({
        'title': 'Annexes',
        'url': None,
        'children': annexes,
        'hidden': False,
        })

    recitals = []

    for recital_link in soup.select('div.recitals-grid > a'):
        if not isinstance(url:=recital_link['href'], str):
            continue
        if not isinstance(p:=recital_link.select_one('div.recital-wrapper > p.recital'), Tag):
            continue
        recitals.append({
            'title': p.text.strip().replace('\n', ' '),
            'url': url,
            'children': [],
            'hidden': True,
            })

    entries.append({
        'title': 'Recitals',
        'url': None,
        'children': recitals,
        'hidden': True,
        })
        

    return {
            'title': 'EU Artificial Intelligence Act',
            'url': None,
            'children': entries, 
            'hidden': False,
            }


def format_toc_entry(toc: TocEntry, depth: int=0, show_hidden: bool=False) -> str:
    """ Format TocEntry as markdown """
    if toc['children']:
        d = '#' * (depth+1)
        return f'{d} {_format_toc_entry_str(toc)}\n\n{_format_toc_entry_children(toc, depth, show_hidden)}'
    return _format_toc_entry_str(toc)
        

def _fetch_toc_parse_element(element: Tag) -> TocEntry:
    title = element.a.text if element.a else element.text
    url = element.a['href'] if element.a else None
    assert not isinstance(url, list)
    return {'title': title.replace('\n', ' '), 'url': url, 'children': [], 'hidden': False}


def _fetch_toc_parse_accordion_content(content: Tag) -> list[TocEntry]:
    result = []
    for p in content.find_all('p', class_='child-article'):
        result.append(_fetch_toc_parse_element(p))
    for p in content.find_all('p', class_='child-chapter'):
        chapter = _fetch_toc_parse_element(p)
        sub_content = p.find_next_sibling('div', class_='accordion-content')
        if sub_content:
            chapter['children'] = _fetch_toc_parse_accordion_content(sub_content)
        result.append(chapter)
    return result 


def _format_toc_entry_str(toc: TocEntry) -> str:
    if toc['url'] and not toc['children']:
        return f'[{toc["title"]}]({toc["url"]})'
    return toc["title"]


def _format_toc_entry_children(toc: TocEntry, depth: int, show_hidden: bool) -> str:
    if all(not child['children'] for child in toc['children']):
        lst = [
                f'- {_format_toc_entry_str(child)}'
                for child in toc['children']
                if show_hidden or not child["hidden"]
                ]
        return '\n'.join(lst)
    
    lst = [
            format_toc_entry(child, depth+1, show_hidden)
            for child in toc['children']
            if show_hidden or not child["hidden"]
            ]
    return '\n\n'.join(lst)

