import requests
import html2text
import time
import traceback

from typing import Annotated
from dataclasses import dataclass
from answerbot.document import MarkdownDocument 
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
    source: str

@dataclass
class Observation:
    info_pieces: list[InfoPiece]

    def to_content(self) -> str:
        return "\n\n".join([info.text for info in self.info_pieces])

class WikipediaTool:
    def __init__(self, 
                document=None,
                current_url=None,
                max_retries=MAX_RETRIES,
                chunk_size=CHUNK_SIZE,
                base_url=BASE_URL,
                api_url=API_URL,
                extra_links=None
                ):
        self.document = document
        self.current_url = current_url

        self.max_retries = max_retries
        self.chunk_size = chunk_size
        self.base_url = base_url
        self.api_url = api_url
        self.extra_links = extra_links if extra_links is not None else []

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
                info_pieces.append(InfoPiece(text="Page does not exist.", source=url))
                break
            elif response.status_code == 200:
                html = response.text
                cleaned_content = self.clean_html_and_textify(html)

                document = MarkdownDocument(cleaned_content, chunk_size=self.chunk_size)
                if title is not None:
                    info_pieces.append(InfoPiece(text=f"Successfully retrieved page {title} from wikipedia", source=url))
                else:
                    info_pieces.append(InfoPiece(text=f"Successfully retrieved document from url: '{url}'", source=url))
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
            sections_list_md = "\n".join(sections)
            if len(sections) > 0:
                info_pieces.append(InfoPiece(text=f'The retrieved page contains the following sections:\n{sections_list_md}', source=url))
            info_pieces.append(InfoPiece(text=f"The retrieved page starts with:\n{document.read_chunk()}", source=url))
        return Observation(info_pieces)

    #@llm_function()
    def get_page(self, title: Annotated[str, "The title of the page to get"]):
        url = self.base_url + title
        return self.get_url(url, title)
    

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
        text = f"Wikipedia search results for query: '{query}' are: "
        results = []
        for item in items:
            results.append(f"[{item['title']}]({item['title']})")
            self.extra_links.append(item['title'])

        search_results = ", ".join(results)
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
            if text:
                num_of_results = len(self.document.lookup_results)
                info_text = f'Keyword "{keyword}" found on current page in {num_of_results} places. The first occurrence:\n{text}'
            else:
                info_text = f'Keyword "{keyword}" not found in current page'
        return Observation([InfoPiece(text=info_text, source=self.current_url)])

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
            info_text = f'Keyword "{self.document.lookup_word}" found in: \n{text}\n{self.document.lookup_position} of {num_of_results} places'
        return Observation([InfoPiece(text=info_text, source=self.current_url)])


    @llm_function()
    def read_chunk(self):
        """
        Reads the next chunk of text from the current location in the current document.
        """
        if self.document is None:
            info_text = "No document defined, cannot read"
        else:
            info_text = self.document.read_chunk()
        return Observation([InfoPiece(text=info_text, source=self.current_url)])

    def make_absolute_url(self, link_address):
        # Check if the link address is already an absolute URL
        parsed_link = urlparse(link_address)
        if parsed_link.scheme:
            return link_address

        # Check if the link address starts with '/'
        if link_address.startswith('/'):
            # Extract the host part of the base URL
            parsed_base = urlparse(self.base_url)
            base_host = parsed_base.scheme + '://' + parsed_base.netloc
            return urljoin(base_host, link_address)

        # Concatenate the base URL with the link address
        return urljoin(self.base_url, link_address)


    @llm_function()
    def follow_link(self, link: Annotated[str, "The link to follow"]):
        """
        Follows a link from the current page and saves the retrieved page as the next current page
        """
        if self.document is None:
            return Observation([InfoPiece(text="No current page, cannot follow", source=self.current_url)])
        else:
            link_with_spaces_restituted = link.replace('_', ' ') # sometimes the LLM tries to replace spaces in links
            if link in self.extra_links:
                url = link
            if link_with_spaces_restituted in self.extra_links:
                url = link
            else:
                url = self.document.resolve_link(link)
            if url is None:
                return Observation([InfoPiece(text=f"There is no '{link}' link on current page", source=self.current_url)])
            else:
                url = self.make_absolute_url(url)
                return self.get_url(url)




if __name__ == "__main__":
    tool = WikipediaTool()
    #pprint(tool.get_page("Wiaaa"))
    pprint(tool.search("Oxygen"))

