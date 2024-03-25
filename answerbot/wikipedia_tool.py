import requests
import html2text
import traceback
import os
import time

from pprint import pprint

from pydantic import BaseModel, Field
from typing import Optional
from bs4 import BeautifulSoup, NavigableString

from answerbot.document import MarkdownDocument
from llm_easy_tools import external_function, extraction_model
from urllib.parse import urlparse, urljoin

MAX_RETRIES = 3
# BASE_URL = 'https://pl.wikipedia.org/wiki/'
# API_URL = 'https://pl.wikipedia.org/w/api.php'
BASE_URL = 'https://en.wikipedia.org/wiki/'
API_URL = 'https://en.wikipedia.org/w/api.php'

class ContentRecord:
    def __init__(self, document, retrieval_history):
        self.document = document
        self.retrieval_history = retrieval_history
    @classmethod
    def load_from_disk(self, title, chunk_size):
        """
        Load a ContentRecord from saved wikitext and retrieval history files based on a given title.

        Returns:
        - ContentRecord: A ContentRecord object reconstructed from the saved files.
        """
        directory = "data/wikipedia_pages"
        sanitized_title = title.replace("/", "_").replace("\\", "_")  # To ensure safe filenames
        sanitized_title = sanitized_title.replace(" ", "_")
        wikitext_filename = os.path.join(directory, f"{sanitized_title}.md")
        history_filename = os.path.join(directory, f"{sanitized_title}.retrieval_history")

        # Load wikitext content
        with open(wikitext_filename, "r", encoding="utf-8") as f:
            document_content = f.read()

        # Load retrieval history
        retrieval_history = []
        with open(history_filename, "r", encoding="utf-8") as f:
            for line in f:
                retrieval_history.append(line.strip())

        document = MarkdownDocument(
            document_content, chunk_size=chunk_size)
        return ContentRecord(document, retrieval_history)

class WikipediaSearch:
    def __init__(self, cached=False, document=None,
                 max_retries=MAX_RETRIES, chunk_size=1024, base_url=BASE_URL, api_url=API_URL,
                 extra_links=None
                 ):
        self.cached = cached
        self.document = document

        self.max_retries = max_retries
        self.chunk_size = chunk_size
        self.base_url = base_url
        self.api_url = api_url
        if extra_links is None:
            extra_links = []
        self.extra_links = extra_links

    @extraction_model()
    class Finish(BaseModel):
        """
        Finish the task and return the answer.
        """
        answer: str = Field(description="The answer to the user's question")
        answer_short: str = Field(description="A short version of the answer")
        reasoning: str = Field(description="The reasoning behind the answer. Think step by step. Mention all assumptions you make.")
#        ambiguity: Optional[str] = Field(description="Have you found anything in the retrieved information that makes the question ambiguous? For example a search for some name can show that there are many different entities with the same name.")

        def normalize_answer(self, answer):
            answer = answer.strip(' \n.\'"')
            answer = answer.replace('’', "'")  # Replace all curly apostrophes with straight single quotes
            answer = answer.replace('"', "'")  # Replace all double quotes with straight single quotes
            if answer.lower() == 'yes' or answer.lower() == 'no':
                answer = answer.lower()
            return answer

        def normalized_answer(self):
            return (
                self.normalize_answer(self.answer),
                self.normalize_answer(self.answer_short),
            )


    class Search(BaseModel):
        query: str = Field(description="The query to search for on Wikipedia")

    @external_function()
    def search(self, param: Search):
        """
        Searches Wikipedia, saves the first result page, and informs about the content of that page.
        """
        if not self.cached:
            search_query = param.query
            search_record = self.wiki_api_search(search_query)
        else:
            title = param.query
            chunk_size = self.chunk_size
            search_record = ContentRecord.load_from_disk(title, chunk_size)
        self.document = search_record.document
        return self._retrieval_observations(search_record)

    def search_result_to_text(self, query, items):
        text = f"Wikipedia search results for query: '{query}' are: "
        results = []
        for item in items:
            results.append(f"[[{item['title']}]]")
            self.extra_links.append(item['title'])

        search_results = ", ".join(results)
        return text + search_results

    def wiki_api_search(self, search_query):
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': search_query,
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
            search_history = [self.search_result_to_text(search_query, search_results)]
            if search_results:
                first_title = search_results[0]['title']
                content_record = self.get_page(first_title)
                combined_history = search_history + content_record.retrieval_history
                return ContentRecord(content_record.document, combined_history)
            else:
                return ContentRecord(None, search_history)

        except requests.exceptions.HTTPError as e:
            search_history = [f"HTTP error occurred during search: {e}"]
            return ContentRecord(None, search_history)
        except Exception as e:
            stack_trace = traceback.format_exc()
            return ContentRecord(None, [stack_trace])

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

    def get_url(self, url, title=None):
        retries = 0
        retrieval_history = []
        print(f"Getting {url}")
        while retries < self.max_retries:
            response = requests.get(url)
            if response.status_code == 404:
                retrieval_history.append(f"Page '{title}' does not exist.")
                break
            elif response.status_code == 200:
                response.raise_for_status()
                html = response.text
                cleaned_content = self.clean_html_and_textify(html)

                document = MarkdownDocument(cleaned_content, chunk_size=self.chunk_size)
                if title is not None:
                    retrieval_history.append(f"Successfully retrieved '{title}' from Wikipedia.")
                else:
                    retrieval_history.append(f"Successfully retrieved '{url}'")
                return ContentRecord(document, retrieval_history)
            else:
                retrieval_history.append(f"HTTP error occurred: {response.status_code}")
            retries += 1
        retrieval_history.append(f"Retries exhausted. No options available.")
        return ContentRecord(None, retrieval_history)

    def get_page(self, title):
        url = self.base_url + title
        return self.get_url(url, title)

    class Get(BaseModel):
        title: str = Field(description="The wikipedia page title")
#    @external_function()
    def get(self, param: Get):
        """
        Retrieves a Wikipedia page, saves the result, and informs about the content of that page.
        """

        if self.cached:
            raise Exception("Cached get not implemented")
        search_record = self.get_page(param.title)
        self.document = search_record.document
        return self._retrieval_observations(search_record)

    class Lookup(BaseModel):
        keyword: str = Field(description="The keyword to search")

    @external_function()
    def lookup(self, param: Lookup):
        """
        Looks up a word on the current page.
        """
        if self.document is None:
            observations = "No document defined, cannot lookup"
        else:
            text = self.document.lookup(param.keyword)
            observations = 'Keyword "' + param.keyword + '" '
            if text:
                num_of_results = len(self.document.lookup_results)
                observations = observations + f"found on current page in {num_of_results} places. The first occurence:\n" + text
            else:
                observations = observations + "not found in current page"
        return observations

    class Next_Lookup(BaseModel):
        pass

    @external_function('next')
    def next_lookup(self, param: Next_Lookup):
        """
        Jumps to the next occurrence of the word searched previously.
        """
        if self.document is None:
            observations = "No document defined, cannot lookup"
        elif not self.document.lookup_results:
            observations = "No lookup results found"
        else:
            text = self.document.next_lookup()
            observations = 'Keyword "' + self.document.lookup_word + '" found in: \n' + text
            num_of_results = len(self.document.lookup_results)
            observations = observations + f"\n{self.document.lookup_position} of {num_of_results} places"
        return observations

    class ReadChunk(BaseModel):
        pass

    @external_function()
    def read_chunk(self, param: ReadChunk):
        """
        Reads the next chunk of text from the current location in the current document.
        """
        if self.document is None:
            observations = "No document defined, cannot read"
        else:
            observations = self.document.read_chunk()
        return observations

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

    class FollowLink(BaseModel):
        link: str = Field(description="The link to follow")

    @external_function()
    def follow_link(self, param: FollowLink):
        """
        Follows a link from the current page and saves the retrieved page as the next current page
        """
        if self.document is None:
            observations = "No current page, cannot follow "
        else:
            link_with_spaces_restituted = param.link.replace('_', ' ') # sometimes the LLM tries to replace spaces in links
            if param.link in self.extra_links:
                url = param.link
            if link_with_spaces_restituted in self.extra_links:
                url = param.link
            else:
                url = self.document.resolve_link(param.link)
            if url is None:
                observations = "There is not such link on current page"
            else:
                url = self.make_absolute_url(url)
                search_record = self.get_url(url)
                self.document = search_record.document
                observations = self._retrieval_observations(search_record)
        return observations

    def _retrieval_observations(self, search_record, limit_sections=None):
        observations = ""
        document = search_record.document
        for record in search_record.retrieval_history:
            observations = observations + record + "\n"
        if document is None:
            observations = observations + "No wikipedia page found"
        else:
            sections = document.section_titles()
            if limit_sections is not None:
                sections = sections[:limit_sections]
            sections_list_md = "\n".join(sections)
            if len(sections) > 0:
                observations = observations + f'The retrieved page contains the following sections:\n{sections_list_md}\n\n'
            observations = observations + "The retrieved page summary starts with:\n" + document.read_chunk() + "\n"
        return observations


if __name__ == "__main__":
    scraper = WikipediaSearch(chunk_size=800)

    query='Kansas Jayhawks fight song'
    searchparam = WikipediaSearch.Search(query=query)
    print(scraper.search(searchparam))
    linkparam = WikipediaSearch.FollowLink(link='Rock Chalk,_Jayhawk')
    print(scraper.follow_link(linkparam))
    exit()

#    title = 'Eileen Heckart'
#    getparam = WikipediaSearch.Get(title=title)
#    scraper.get(getparam)
#    if scraper.document:
#        print(f"{title} found\n")
#        pprint(scraper.document.text_to_url)
#
#    exit()
#
#    title = "Lewiston Maineiacs"
#    getparam = WikipediaSearch.Get(title=title)
#    scraper.get(getparam)
#    if scraper.document:
#        print(f"{title} found\n")
#        flink = scraper.FollowLink(link='Androscoggin Bank Colisée')
#        print(scraper.follow_link(flink))
#
#
#
#    exit()
    title = "Shirley Temple"
    title = "Kiss and Tell 1945 film"
    content_record = scraper.wiki_api_search(title)
    if content_record.document:
        print(f"Searching for {title}:\n")
        document = content_record.document
        print(document.read_chunk())
        keyword = 'Shirley Temple'
        print(f'\nLooking up {keyword}:\n')
        print(document.lookup(keyword))
    else:
        print("No document found")

    print('\n')
    print('------------------\n')
    exit()

    title = "Oxygen"
    content_record = scraper.wiki_api_search(title)
    if content_record.document:
        print(f"Searching for {title}:\n")
        document = content_record.document
        print(document.read_chunk())
        print('\nLooking up atomic weight:\n')
        print(document.lookup('atomic weight'))
        print('\nSection titles:\n')
        print(document.section_titles())
        print("\nRetrieval History:")
        for record in content_record.retrieval_history:
            print(record)
    else:
        print("No document found")

    print('\n')
    print('------------------\n')


#    search_query = "Machine learning"
#    search_record = scraper.search(search_query)
#    if search_record.document:
#        print(search_record.document.first_chunk())
#        print("\nRetrieval History:")
#        for record in search_record.retrieval_history:
#            print(record)
#    else:
#        print("No document found")
