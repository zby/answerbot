import requests
import html2text
import time
import traceback

from typing import Annotated
from dataclasses import dataclass
from answerbot.document import MarkdownDocument 
from bs4 import BeautifulSoup, NavigableString
from pprint import pprint


MAX_RETRIES = 3
# BASE_URL = 'https://pl.wikipedia.org/wiki/'
# API_URL = 'https://pl.wikipedia.org/w/api.php'
BASE_URL = 'https://en.wikipedia.org/wiki/'
API_URL = 'https://en.wikipedia.org/w/api.php'
CHUNK_SIZE = 1024

@dataclass
class InfoPiece:
    text: str
    source: str

@dataclass
class Observation:
    info_pieces: list[InfoPiece]

    def to_content(self) -> str:
        return "\n\n".join([info.text for info in self.info_pieces])

class WikipediaTool:
    def __init__(self, 
                document=None,
                max_retries=MAX_RETRIES,
                chunk_size=CHUNK_SIZE,
                base_url=BASE_URL,
                api_url=API_URL,
                ):
        self.document = document

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

 
    def get_url(self, url, title=None, limit_sections=None):
        retries = 0
        print(f"Getting {url}")
        info_pieces = []
        document = None
        while retries < self.max_retries:
            response = requests.get(url)
            if response.status_code == 404:
                info_pieces.append(InfoPiece(text="Page does not exist", source=url))
                break
            elif response.status_code == 200:
                html = response.text
                cleaned_content = self.clean_html_and_textify(html)

                document = MarkdownDocument(cleaned_content, chunk_size=self.chunk_size)
                if title is not None:
                    info_pieces.append(InfoPiece(text=f"Successfully retrieved page {title} from wikipedia", source=url))
                else:
                    info_pieces.append(InfoPiece(text="Successfully retrieved document from url", source=url))
                break
            else:
                info_pieces.append(InfoPiece(text=f"HTTP error occurred: {response.status_code}", source=url))
            retries += 1
            info_pieces.append(InfoPiece(text=f"Retries exhausted. No options available.", source=url))
        if document is not None:
            self.document = document
            sections = document.section_titles()
            if limit_sections is not None:
                sections = sections[:limit_sections]
            sections_list_md = "\n".join(sections)
            if len(sections) > 0:
                info_pieces.append(InfoPiece(text=f'The retrieved page contains the following sections:\n{sections_list_md}\n\n', source=url))
            info_pieces.append(InfoPiece(text=f"The retrieved page starts with:\n{document.read_chunk()}\n", source=url))
        return Observation(info_pieces)

    def get_page(self, title):
        url = self.base_url + title
        return self.get_url(url, title)
    

        """
        Searches Wikipedia, saves the first result page, and informs about the content of that page using InformationPieces.
        """
        info_pieces = []
        search_record = self.wiki_api_search(query)
        self.document = search_record.document
        if search_record.document is None:
            info_pieces.append(InfoPiece(text="No document found", source=self.api_url))
        else:
            info_pieces.append(InfoPiece(text=f"Document retrieved successfully: {query}", source=self.api_url))

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
        text = f"Wikipedia search results for query: '{query}' are: "
        results = []
        for item in items:
            results.append(f"[[{item['title']}]]")
            #self.extra_links.append(item['title'])

        search_results = ", ".join(results)
        return text + search_results




if __name__ == "__main__":
    tool = WikipediaTool()
    #pprint(tool.get_page("Wiaaa"))
    pprint(tool.search("Oxygen"))

