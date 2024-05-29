import requests
import html2text
import time
import traceback

from typing import Annotated, Optional
from answerbot.markdown_document import MarkdownDocument 
from bs4 import BeautifulSoup, NavigableString
from urllib.parse import urlparse, urljoin
from pprint import pprint

from llm_easy_tools import llm_function, ToolResult

from answerbot.observation import Observation, InfoPiece
from answerbot.clean_reflection import ReflectionResult

MAX_RETRIES = 3
# BASE_URL = 'https://pl.wikipedia.org/wiki/'
# API_URL = 'https://pl.wikipedia.org/w/api.php'
BASE_URL = 'https://en.wikipedia.org/'
API_URL = 'https://en.wikipedia.org/w/api.php'
CHUNK_SIZE = 1024

class WikipediaTool:
    def __init__(self, 
                document=None,
                url_shortener=None,
                current_url=None,
                max_retries=MAX_RETRIES,
                min_chunk_size=100,
                chunk_size=CHUNK_SIZE,
                absolute_base_url=BASE_URL,
                api_url=API_URL,
                ):
        self.document = document
        self.url_shortener = url_shortener
        self.current_url = current_url
        self.min_chunk_size = min_chunk_size
        self.max_retries = max_retries
        self.chunk_size = chunk_size
        self.absolute_base_url = absolute_base_url
        self.base_url = self.absolute_base_url + 'wiki/'
        self.api_url = api_url

        self.checked_urls = []

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
            
        for a_tag in content.find_all('a', href=True):
            relative_href = a_tag['href']
            if 'title' in a_tag.attrs:
                del a_tag.attrs['title']  # Remove the title attribute if it exists
            if not relative_href.startswith(('http://', 'https://')):
                absolute_href = urljoin(self.absolute_base_url, relative_href)
                a_tag['href'] = absolute_href
            if self.url_shortener:
                a_tag['href'] = self.url_shortener.shorten(a_tag['href'])

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
        if self.url_shortener:
            url = self.url_shortener.retrieve(url)
        print(f"Getting {url}")
        info_pieces = []
        document = None
        self.checked_urls.append(url)
        info_pieces.append(InfoPiece(text=f"Trying to retrieve url: `{url}`."))
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
                    info_pieces.append(InfoPiece(text=f'Successfully retrieved page "{title}" from wikipedia'))
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
            sections_list_md = "- " + "\n- ".join(sections) if sections else ""
            if len(sections) > 0:
                text = f'The retrieved page contains the following sections:\n{sections_list_md}'
                info_pieces.append(InfoPiece(text=text, source=url))
            chunk = document.read_chunk()
            quoted_text = self.quote_text(chunk)
            info_pieces.append(InfoPiece("The retrieved page starts with:"))
            info_pieces.append(InfoPiece(quoted_text, source=url, quotable=True))
            info = "If you want to continue reading the page, you can call `read_more`. "
            info += "If you want to jump to a specific keyword on this page (for example a section of the article) `lookup`."
            info_pieces.append(InfoPiece(info))
        result = Observation(info_pieces)
        #pprint(result)
        return result

    #@llm_function()
    def get_page(self, title: Annotated[str, "The title of the page to get"]):
        url_title = title.replace(' ', '_')
        url = self.base_url + url_title
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
        info_pieces.append(InfoPiece(text=f"The following wikipedia search was used: `{query}`."))

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
                get_page_observation = self.get_page(first_title)
                info_pieces.extend(get_page_observation.info_pieces)
                info_pieces.append(InfoPiece(text="If you want to get a different page you can call `get_url`."))
            else:
                info_pieces.append(InfoPiece(text="No results found", source=self.api_url))

        except requests.exceptions.HTTPError as e:
            info_pieces.append(InfoPiece(text=f"HTTP error occurred during search: {e}", source=self.api_url))
            return Observation(info_pieces)
        except Exception as e:
            stack_trace = traceback.format_exc()
            info_pieces.append(InfoPiece(text=f"Error during search: {stack_trace}", source=self.api_url))
            return Observation(info_pieces)

        return Observation(info_pieces)

    def search_result_to_text(self, query, items):
        text = f"Wikipedia search results for query: '{query}' are:"
        results = []
        for item in items:
            results.append(f"[{item['title']}]({urljoin(self.base_url, item['title'])})")

        search_results = "\n- ".join(results)
        return text + "\n- " + search_results

    def quote_text(self, text):
        quoted_text = "\n".join([f"> {line}" for line in text.split('\n')])
        return quoted_text

    @llm_function()
    def lookup(self, keyword: Annotated[str, "The keyword to search"] ):
        """
        Looks up a word on the current page.
        """
        if self.document is None:
            return Observation([InfoPiece(text="No document defined, cannot lookup, you need to use search first to retrieve a document", source=self.current_url)])
        else:
            text = self.document.lookup(keyword)
            current_url = self.current_url
            if text:
                quoted_text = self.quote_text(text)
                num_of_results = len(self.document.lookup_results)
                info = f'If you want to continue reading from this point, you can call `read_more`.'
                if num_of_results > 1:
                    info += f' If you want to jump to the next occurence of the keyword you can call `next`.'
                return Observation([
                    InfoPiece(f'Keyword "{self.document.lookup_word}" found at "{current_url}" in {num_of_results} places. The first occurrence:'),
                    InfoPiece(quoted_text, source=self.current_url, quotable=True),
                    InfoPiece(f"- *{self.document.lookup_position + 1} of {num_of_results} places*"),
                    InfoPiece(info),
                ])

                return Observation([InfoPiece(text=info, source=self.current_url, quotable=True)])
            else:
                info = f'Keyword "{keyword}" not found at "{current_url}". You might try using `lookup` with a modified keyword - for example use synonyms.\n'
                if " " in keyword:
                    info += "\nNote: Your keyword contains spaces. Consider using a single word for more effective lookups, multiple words can be separated by additional text or whitespace, or they might occur in different order."
                return Observation([InfoPiece(info)])

    @llm_function('next')
    def next_lookup(self):
        """
        Jumps to the next occurrence of the word searched previously.
        """
        if self.document is None:
            info = "No document defined, cannot lookup, you need to use search or get_url first to retrieve a document"
            return Observation([InfoPiece(text=info, source=self.current_url)])
        elif not self.document.lookup_results:
            info = "No lookup results found, you need to use lookup first to find the places with the word you are looking for"
            return Observation([InfoPiece(text=info, source=self.current_url)])
        else:
            text = self.document.next_lookup()
            num_of_results = len(self.document.lookup_results)
            quoted_text = self.quote_text(text)
            info = f'If you want to continue reading from this point, you can call `read_more`.'
            if num_of_results > self.document.lookup_position + 1:
                info += f' If you want to jump to the next occurence of the keyword you can call `next`.'
            return Observation([
                InfoPiece(f'Keyword "{self.document.lookup_word}" found in:'),
                InfoPiece(quoted_text, source=self.current_url, quotable=True),
                InfoPiece(f"*{self.document.lookup_position} of {num_of_results} places*"),
                InfoPiece(info),
            ])


    @llm_function('read_more')
    def read_chunk(self):
        """
        Reads the next chunk of text from the current location in the current document.
        """
        if self.document is None:
            return Observation([InfoPiece(text="No document defined, cannot read", source=self.current_url)])
        else:
            text = self.document.read_chunk()
            quoted_text = self.quote_text(text)
            return Observation([
                InfoPiece("A new fragment from the current page was read:"),
                InfoPiece(text=quoted_text, source=self.current_url, quotable=True),
                InfoPiece("If you want to continue reading the page, you can call `read_more`."),
            ])


    def remove_checked_urls(self, reflection: ReflectionResult):
        for url in self.checked_urls:
            reflection.remove_source(url)


if __name__ == "__main__":
    from answerbot.url_shortener import UrlShortener
    url_shortener = UrlShortener()
    tool = WikipediaTool(url_shortener=url_shortener)
    #pprint(tool.get_page("Wiaaa"))
    #pprint(tool.search("Oxygen"))
    #pprint(tool.get_url("https://en.wikipedia.org/wiki/Ann_B._Davis"))
    #tool.get_url("https://en.wikipedia.org/wiki/Kiss_and_Tell_(1945_film)")
    #print(str(tool.lookup("Cast")))

    tool.get_url("https://en.wikipedia.org/wiki/Lewiston_Maineiacs")
    print(str(tool.lookup("arena")))

