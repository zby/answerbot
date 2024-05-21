import requests
import html2text
import time
import traceback

from typing import Annotated, Optional
from dataclasses import dataclass, field
from answerbot.markdown_document import MarkdownDocument 
from bs4 import BeautifulSoup, NavigableString
from urllib.parse import urlparse, urljoin
from pprint import pprint

from llm_easy_tools import llm_function

MAX_RETRIES = 3
# BASE_URL = 'https://pl.wikipedia.org/wiki/'
# API_URL = 'https://pl.wikipedia.org/w/api.php'
BASE_URL = 'https://en.wikipedia.org/wiki/'
API_URL = 'https://en.wikipedia.org/w/api.php'
CHUNK_SIZE = 1024

@dataclass
class InfoPiece:
    text: str
    source: Optional[str] = None
    quotable: bool = False
    def to_markdown(self) -> str:
        if self.source:
            return f"{self.text}\n— *from {self.source}*"
        else:
            return self.text

@dataclass
class Observation:
    info_pieces: list[InfoPiece]
    is_refined: bool = False
    interesting_links: list[str] = field(default_factory=list)
    comment: Optional[str] = None

    def clear_info_pieces(self):
        self.info_pieces = []

    def add_info_piece(self, info_piece: InfoPiece):
        self.info_pieces.append(info_piece)


    def __str__(self) -> str:
        result = ''
        if self.is_refined:
            result += 'Relevant quotes:\n'
        for info in self.info_pieces:
            result += f"\n\n{info.to_markdown()}"
        if self.interesting_links:
            result += '\n\nNew sources:\n'
            for link in self.interesting_links:
                result += f"\n\n- {link}"
        if self.comment:
            result += f"\n\nComment: {self.comment}"
        return result

class WikipediaTool:
    def __init__(self, 
                document=None,
                current_url=None,
                max_retries=MAX_RETRIES,
                min_chunk_size=100,
                chunk_size=CHUNK_SIZE,
                base_url=BASE_URL,
                api_url=API_URL,
                ):
        self.document = document
        self.current_url = current_url
        self.min_chunk_size = min_chunk_size
        self.max_retries = max_retries
        self.chunk_size = chunk_size
        self.base_url = base_url
        self.api_url = api_url

    @classmethod
    def clean_html_and_textify(self, html):

        # remove table of content
        soup = BeautifulSoup(html, 'html.parser')
        content = soup.find('div', id='bodyContent')

        for tag in content.find_all():
            if tag.name == 'a' and 'action=edit' in tag.get('href', ''):
                tag.decompose()
            if tag.name == 'div' and tag.get('id') == 'toc':
                tag.decompose()
            if tag.name == 'div' and 'vector-body-before-content' in tag.get('class', []):
                tag.decompose()
            if tag.name == 'div' and 'infobox-caption' in tag.get('class', []):
                tag.decompose()
            if tag.name == 'img':
                tag.decompose()
            if tag.name == 'span' and 'hide-when-compact' in tag.get('class', []):
                tag.decompose()
            if tag.name == 'span' and 'mw-editsection' in tag.get('class', []):
                tag.decompose()
            if tag.name == 'div' and tag.get('id') == 'mw-fr-revisiondetails-wrapper':
                tag.decompose()
            if tag.name == 'figure':
                tag.decompose()

        # Remove some metadata - we need compact information
        search_text = "This article relies excessively on"
        text_before_anchor = "relies excessively on"
        required_href = "/wiki/Wikipedia:Verifiability"

        # Find all <div> elements with class 'mbox-text-span'
        for div in content.find_all('div', class_='mbox-text-span'):
            gtex = div.get_text().strip()
            if search_text in div.get_text():
                for a_tag in div.find_all('a', href=required_href):
                    preceding_text = ''.join([str(sibling) for sibling in a_tag.previous_siblings if isinstance(sibling, NavigableString)])
                    if text_before_anchor in preceding_text:
                        div.decompose()
                        break  # Stop checking this div, as we found a match
        for div in content.find_all('div', class_='mbox-text-span'):
            print(div)
        modified_html = str(content)

        converter = html2text.HTML2Text()
        # Avoid breaking links into newlines
        converter.body_width = 0
        # converter.protect_links = True # this does not seem to work
        markdown = converter.handle(modified_html)
        cleaned_content = markdown.strip()
        return cleaned_content

    @llm_function()
    def get_url(self, url: Annotated[str, "The URL to get"]):
        """
        Retrieves a page from Wikipedia by its URL.
        """
        return self._get_url(url)

    def _get_url(self, url: str, title=None, limit_sections = None):
        retries = 0
        print(f"Getting {url}")
        info_pieces = []
        document = None
        while retries < self.max_retries:
            response = requests.get(url)
            if response.status_code == 404:
                info_pieces.append(InfoPiece(text="Page does not exist.", source=url))
                break
            elif response.status_code == 200:
                html = response.text
                cleaned_content = self.clean_html_and_textify(html)

                document = MarkdownDocument(cleaned_content, min_size=self.min_chunk_size, max_size=self.chunk_size)
                if title is not None:
                    info_pieces.append(InfoPiece(text=f"Successfully retrieved page {title} from wikipedia"))
                else:
                    info_pieces.append(InfoPiece(text=f"Successfully retrieved document from url: '{url}'"))
                break
            else:
                info_pieces.append(InfoPiece(text=f"HTTP error occurred: {response.status_code}", source=url))
            retries += 1

        if retries == self.max_retries:
            info_pieces.append(InfoPiece(text=f"Retries exhausted. No options available.", source=url))

        if document is not None:
            self.document = document
            self.current_url = url
            sections = document.section_titles()
            if limit_sections is not None:
                sections = sections[:limit_sections]
            sections_list_md = "\n- ".join(sections)
            if len(sections) > 0:
                info_pieces.append(InfoPiece(text=f'The retrieved page contains the following sections:\n{sections_list_md}', source=url, quotable=True))
            chunk = document.read_chunk()
            quoted_text = "\n".join([f"> {line}" for line in chunk.split('\n')])
            info_pieces.append(InfoPiece(text=f"The retrieved page starts with:\n{quoted_text}", source=url, quotable=True))
        result = Observation(info_pieces)
        pprint(result)
        return result

    #@llm_function()
    def get_page(self, title: Annotated[str, "The title of the page to get"]):
        url = self.base_url + title
        return self._get_url(url, title)


    @llm_function()
    def search(self, query: Annotated[str, "The query to search for on Wikipedia"]):
        """
        Searches Wikipedia using the provided search query. Reports the search results and the content of the first page.
        """
        info_pieces = []
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'srlimit': 10,  # Limit the number of results
        }

        try:
            response = requests.get(self.api_url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'error' in data and data['error']['code'] == 'cirrussearch-too-busy-error':
                time.sleep(10)
                response = requests.get(self.api_url, params=params)
                response.raise_for_status()
                data = response.json()

            if 'error' in data:
                raise RuntimeError(data['error']['info'])

            search_results = data['query']['search']
            if search_results:
                search_results_text = self.search_result_to_text(query, search_results)
                info_pieces.append(InfoPiece(text=search_results_text, source=self.api_url))
                first_title = search_results[0]['title']
                info_pieces.extend(self.get_page(first_title).info_pieces)
            else:
                info_pieces.append(InfoPiece(text="No results found", source=self.api_url))

        except requests.exceptions.HTTPError as e:
            info_pieces.append(InfoPiece(text=f"HTTP error occurred during search: {e}", source=self.api_url))
        except Exception as e:
            stack_trace = traceback.format_exc()
            info_pieces.append(InfoPiece(text=f"Error during search: {stack_trace}", source=self.api_url))

        return Observation(info_pieces)

    def search_result_to_text(self, query, items):
        text = f"Wikipedia search results for query: '{query}' are:\n"
        results = []
        for item in items:
            results.append(f"[{item['title']}]({urljoin(self.base_url, item['title'])})")

        search_results = "\n- ".join(results)
        return text + search_results

    @llm_function()
    def lookup(self, keyword: Annotated[str, "The keyword to search"] ):
        """
        Looks up a word on the current page.
        """
        if self.document is None:
            info_text="No document defined, cannot lookup, you need to use search first to retrieve a document"
        else:
            text = self.document.lookup(keyword)
            quoted_text = "\n".join([f"> {line}" for line in text.split('\n')])
            if text:
                num_of_results = len(self.document.lookup_results)
                info_text = f'Keyword "{keyword}" found on current page in {num_of_results} places. The first occurrence:\n{quoted_text}'
            else:
                info_text = f'Keyword "{keyword}" not found in current page'
        return Observation([InfoPiece(text=info_text, source=self.current_url, quotable=True)])

    @llm_function('next')
    def next_lookup(self):
        """
        Jumps to the next occurrence of the word searched previously.
        """
        if self.document is None:
            info_text = "No document defined, cannot lookup"
        elif not self.document.lookup_results:
            info_text = "No lookup results found"
        else:
            text = self.document.next_lookup()
            num_of_results = len(self.document.lookup_results)
            quoted_text = "\n".join([f"> {line}" for line in text.split('\n')])
            info_text = f'Keyword "{self.document.lookup_word}" found in: \n{quoted_text}\n- *{self.document.lookup_position} of {num_of_results} places*'
        return Observation([InfoPiece(text=info_text, source=self.current_url, quotable=True)])


    @llm_function()
    def read_chunk(self):
        """
        Reads the next chunk of text from the current location in the current document.
        """
        if self.document is None:
            info_text = "No document defined, cannot read"
        else:
            info_text = self.document.read_chunk()
        return Observation([InfoPiece(text=info_text, source=self.current_url, quotable=True)])

if __name__ == "__main__":
    tool = WikipediaTool()
    #pprint(tool.get_page("Wiaaa"))
    #pprint(tool.search("Oxygen"))
    pprint(tool.get_url("https://en.wikipedia.org/wiki/Ann_B._Davis"))

