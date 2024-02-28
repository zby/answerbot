import requests
import html2text
import traceback
import os

from pydantic import BaseModel, Field
from bs4 import BeautifulSoup, NavigableString

from answerbot.document import MarkdownDocument
from llm_easy_tools import external_function

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
                 max_retries=MAX_RETRIES, chunk_size=1024, base_url=BASE_URL, api_url=API_URL):
        self.cached = cached
        self.document = document

        self.max_retries = max_retries
        self.chunk_size = chunk_size
        self.base_url = base_url
        self.api_url = api_url

    class Finish(BaseModel):
        """
        Finish the task and return the answer.
        """
        reason: str = Field(description="The reasoning behind the answer")
        answer: str = Field(description="The answer to the user's question")

        def normalized_answer(self):
            answer = self.answer
            if answer.lower() == 'yes' or answer.lower() == 'no':
                answer = answer.lower()
            return answer


    class Search(BaseModel):
        reason: str = Field(description="The reason for searching")
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

            search_results = [item['title'] for item in data['query']['search']]
            squared_results = [f"[[{result}]]" for result in search_results]
            search_history = [f"Wikipedia search results for query: '{search_query}' are: " + ", ".join(squared_results)]

            if search_results:
                content_record = self.wiki_api_get_page(search_results[0])
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

    def wiki_api_get_page(self, title):
        url = self.base_url + title
        return self.get_url(url, title)

    class Get(BaseModel):
        reason: str = Field(description="The reason for retrieving the page")
        title: str = Field(description="The wikipedia page title")
    @external_function()
    def get(self, param: Get):
        """
        Retrieves a Wikipedia page, saves the result, and informs about the content of that page.
        """

        if self.cached:
            raise Exception("Cached get not implemented")
        search_record = self.wiki_api_get_page(param.title)
        self.document = search_record.document
        return self._retrieval_observations(search_record)

    class Lookup(BaseModel):
        reason: str = Field(description="The reason for searching")
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
        reason: str = Field(description="The reason for searching")

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
        reason: str = Field(description="The reason for continuing reading in the current place")

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

    class FollowLink(BaseModel):
        reason: str = Field(description="The reason for following a link")
        link: str = Field(description="The link to follow")

    @external_function()
    def follow_link(self, param: FollowLink):
        """
        Follows a link from the current page and saves the retrieved page as the next current page
        """
        if self.document is None:
            observations = "No current page, cannot follow "
        else:
            url = self.document.links[param.link]
            search_record = self.wiki_api_get_page(url)
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
            observations = observations + f'The retrieved page contains the following sections:\n{sections_list_md}\n'
            observations = observations + "The retrieved page summary starts with:\n" + document.read_chunk() + "\n"
        return observations


if __name__ == "__main__":
    scraper = WikipediaSearch(chunk_size=800)
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
