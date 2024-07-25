import requests
import html2text
import time
import traceback

from typing import Annotated, Optional, Callable, Union
from answerbot.tools.markdown_document import MarkdownDocument 
from bs4 import BeautifulSoup, NavigableString
from urllib.parse import urlparse, urljoin
from pprint import pprint

from llm_easy_tools import LLMFunction, ToolResult, get_tool_defs

from answerbot.tools.observation import Observation

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
                limit_sections=None,
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
        self.limit_sections = limit_sections

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
            #print(div)
            pass
            
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

    def mk_observation(self, content, operation, quotable=False, goal=None):
        return Observation(content, source=self.current_url, operation=operation, quotable=quotable, goal=goal)

    def make_hint(self, text):
        return f"\n\n**Hint:** {text}\n"

    def get_sections_infopiece(self) -> Union[str, None]:
        if self.document is None:
            return None

        sections = self.document.section_titles()
        if not sections:
            return None
        if self.limit_sections is not None:
            sections = sections[:self.limit_sections]
        quoted_sections = [f'"{section}"' for section in sections]
        sections_str = ", ".join(quoted_sections)
        text = f'The retrieved page contains the following sections:\n{sections_str}\n\n'
        text += self.make_hint("You can easily jump to any section by using `lookup` function with the section name.")
        return text

    def get_url(self, url: Annotated[str, "The URL to get"], goal: Annotated[str, "What information do you expect to find on this page and why did you choose this url?"]):
        """
        Retrieves a page from Wikipedia by its URL, saves it as the current page and shows the top of that page. You need to use `lookup` or `read_more` to read the rest of the page.
        """
        return self._get_url(url, goal)

    def _get_url(self, url: str, goal=None, title=None, limit_sections = None):
        self.current_url = None
        document = None
        quotable = False
        retries = 0
        if self.url_shortener:
            url = self.url_shortener.retrieve(url)
        print(f"\nGetting {url}\n")
        if not url.startswith(self.absolute_base_url):
            return self.mk_observation(f"This tool can only work on pages from {self.absolute_base_url}.", f"get_url('{url}')")
        self.checked_urls.append(url)
        content = ""
        while retries < self.max_retries:
            response = requests.get(url, allow_redirects=True)
            if response.status_code == 404:
                content += "Page does not exist."
                break
            elif response.status_code == 200:
                self.current_url = response.url  # this takes into account redirections
                html = response.text
                cleaned_content = self.clean_html_and_textify(html)

                document = MarkdownDocument(cleaned_content, min_size=self.min_chunk_size, max_size=self.chunk_size)
                if title is not None:
                    content += f'Successfully retrieved page "{title}" from wikipedia\n'
                else:
                    content += f"Successfully retrieved document from url: '{url}'\n"
                break
            else:
                content += f"HTTP error occurred: {response.status_code} when trying to retrieve url: '{url}'\n"
            retries += 1

        if retries == self.max_retries:
            content += f"Retries exhausted. No options available."

        if document is not None:
            quotable = True
            self.document = document
            sections_info = self.get_sections_infopiece()
            if sections_info:
                content += sections_info + "\n"
            chunk = document.read_chunk()
            quoted_text = self.quote_text(chunk)
            content += "The retrieved page starts with:\n\n"
            content += quoted_text + "\n\n"
            if True:  # TODO: check if this is the full content of the page
                content += self.make_hint("This was not the full content of the page. If you want to continue reading the page, you can call `read_more`."
                "If you want to jump to a specific keyword on this page (for example a section of the article) `lookup('keyword')`.")
        return self.mk_observation(content, f"get_url('{url}')", quotable, goal)

    def get_page(self, title: Annotated[str, "The title of the page to get"]):
        url_title = title.replace(' ', '_')
        url = self.base_url + url_title
        return self._get_url(url, title)

    def search(self, query: Annotated[str, "The query to search for on Wikipedia"]):
        """
Searches Wikipedia using the provided search query. It is a keyword search and works best with simple queries.
Think about what pages exist on wikipedia and search accordingly. Never put two proper nouns into the same query,
because there are no wikipedia pages about two topics. When trying to learn about a property of an object or a person,
first search for that object then use other tools to locate the needed information.
        """
        self.current_url = self.api_url
        print(f"\nSearching for '{query}'\n")
        content = f"The following wikipedia search was used: `{query}`.\n"
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
                content += self.search_result_to_text(query, search_results)
            else:
                content += "No results found"

        except requests.exceptions.HTTPError as e:
            content += f"HTTP error occurred during search: {e}"
        except Exception as e:
            stack_trace = traceback.format_exc()
            content += f"Error during search: {stack_trace}"

        return self.mk_observation(content, f"search('{query}')")

    def search_result_to_text(self, query, items):
        text = f"Wikipedia search results for query: '{query}' are:"
        results = []
        for item in items:
            results.append(f"[{item['title']}]({urljoin(self.base_url, item['title'])})")

        search_results = "\n- ".join(results)
        hint = self.make_hint("Have a look at the search results - you can use `get_url` to retrieve the page of a result.")
        return f"{text}\n- {search_results}\n\n{hint}"

    def quote_text(self, text):
        quoted_text = "\n".join([f"> {line}" for line in text.split('\n')])
        return quoted_text

    def lookup(self, keyword: Annotated[str, "The keyword to search"] ):
        """
Looks up a word on the current page. Use it if you think you are on the right page and want to jump to a specific word on it.
Don't use it for searching for new pages - use `search` for that.
Be careful with using multiple words as a keyword - it might be better to choose one of them because it often
happens that the words appear in different order or separated by a word or a formatting in the text."""

        print(f"\nLooking up '{keyword}'\n")
        if self.document is None:
            return self.mk_observation(
                "No document defined, cannot lookup, you need to use search first to retrieve a document",
                f"lookup('{keyword}')")
        else:
            text = self.document.lookup(keyword)
            current_url = self.current_url
            if text:
                quoted_text = self.quote_text(text.strip())
                num_of_results = len(self.document.lookup_results)
                content = f'Keyword "{self.document.lookup_word}" found at "{current_url}" in {num_of_results} places. The first occurrence:\n\n'
                content += quoted_text + f"\n\n- *{self.document.lookup_position + 1} of {num_of_results} places*\n\n"
                content += f'If you want to continue reading from this point, you can call `read_more`.'
                if num_of_results > 1:
                    content += f' If you want to jump to the next occurence of the keyword you can call `next`.'
                return self.mk_observation(content, f"lookup('{keyword}')", True)
            else:
                content = f'Keyword "{keyword}" not found at "{current_url}". You might try using `lookup` with a modified keyword - for example use synonyms.\n'
                if " " in keyword:
                    content += self.make_hint("Your keyword contains spaces. Consider using a single word for more effective lookups, multiple words can be separated by additional text or whitespace, or they might occur in different order.")
                return self.mk_observation(content, f"lookup('{keyword}')")

    def next_lookup(self):
        """
Jumps to the next occurrence of the word searched previously."""

        print(f"\nLooking up next occurrence of '{self.document.lookup_word}'\n")
        if self.document is None:
            return self.mk_observation("No document defined, cannot lookup, you need to use search or get_url first to retrieve a document", "next")
        elif not self.document.lookup_results:
            return self.mk_observation("No lookup results found, you need to use lookup first to find the places with the word you are looking for", "next")
        else:
            text = self.document.next_lookup()
            text = text.strip()
            num_of_results = len(self.document.lookup_results)
            quoted_text = self.quote_text(text)
            content = f'Keyword "{self.document.lookup_word}" found in:\n\n'
            content += quoted_text + f"\n\n*{self.document.lookup_position} of {num_of_results} places*\n\n"
            content += f'If you want to continue reading from this point, you can call `read_more`.'
            if num_of_results > self.document.lookup_position + 1:
                content += f' If you want to jump to the next occurence of the keyword you can call `next`.'
            return self.mk_observation(content, 'next()', True)

    def read_chunk(self):
        """
Reads the next chunk of text from the current location in the current page.
Use it if the information just received was interesting but seems to be cut short and you want to continue reading.
"""
        print(f"\nReading more from current position\n")
        if self.document is None:
            return self.mk_observation("No document defined, cannot read", 'read_more()')
        else:
            text = self.document.read_chunk()
            text = text.strip()
            quoted_text = self.quote_text(text)
            content = "A new fragment from the current page was read:\n\n"
            content += quoted_text + "\n\nIf you want to continue reading the page, you can call `read_more`."
            return self.mk_observation(content, 'read_more()', True)

    def get_llm_tools(self) -> list[Callable]:
        result = [self.search, self.get_url]

        if self.document:
            result.append(LLMFunction(self.read_chunk, name="read_more"))   # changing name to read_more
            result.append(self.lookup)
            if self.document.lookup_results:
                result.append(self.next_lookup)

        return result

if __name__ == "__main__":
    tool = WikipediaTool(chunk_size=400)
    observation = tool.get_url("https://en.wikipedia.org/wiki/Androscoggin_Bank_Colisée", 'aa')
    pprint(observation)

    exit()
    from answerbot.tools.url_shortener import UrlShortener
    url_shortener = UrlShortener()
    observation = tool.get_url("https://en.wikipedia.org/wiki/Kiss_and_Tell_(1945_film)", '')
    print(str(observation))
    print()
    print('-'*100)
    observation = tool.lookup("cast")
    print("\nObservation:\n\n")
    print(str(observation))
    pprint(observation)
    print(str(tool.lookup("Corliss Archer")))

    pprint(tool.get_page("Wiaaa"))
    pprint(tool.search("Oxygen"))
    #pprint(tool.get_url("https://en.wikipedia.org/wiki/Ann_B._Davis"))
    #tool.get_url("https://en.wikipedia.org/wiki/Kiss_and_Tell_(1945_film)")
    #print(str(tool.lookup("Cast")))

    #tool.get_url("https://en.wikipedia.org/wiki/Lewiston_Maineiacs")
    #print(str(tool.lookup("arena")))
    #print(observation.available_tools)
