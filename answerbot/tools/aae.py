from answerbot.tools.observation import Observation
from typing import Annotated, Callable
import requests
import html2text
from bs4 import BeautifulSoup, PageElement, Tag
from answerbot.tools.markdown_document import MarkdownDocument

from tenacity import retry, stop_after_attempt


MAX_RETRIES = 3
BASE_URL = 'https://artificialintelligenceact.eu/'
CHUNK_SIZE = 1024
MIN_CHUNK_SIZE = 100



class AAESearch:

    def __init__(
            self, 
            max_retries=MAX_RETRIES,
            base_url=BASE_URL,
            chunk_size=CHUNK_SIZE,
            min_chunk_size=MIN_CHUNK_SIZE,
            ):
        self._max_retries = max_retries
        self._base_url = base_url
        self._document: tuple[MarkdownDocument, str]|None = None
        self._chunk_size = chunk_size
        self._min_chunk_size = min_chunk_size

    def get_llm_tools(self) -> list[Callable]:
        result = [
                self.search_aae,
                self.goto_url,
                ]
        if self._document:
            result.append(self.read_chunk)
            if self._document[0].lookup_results:
                result.append(self.lookup_next)
            else:
                result.append(self.lookup)
        return result


    def search_aae(self, query: Annotated[str, 'The query to search for']):
        """
        Searches Artificial Intelligence Act EU website, and return a list of documents
        matching the query, with short blocks of texts containing the query
        """
        print(f'Searching for "{query}"')
        result = retry(stop=stop_after_attempt(self._max_retries))(_aae_search)(query, self._base_url)
        return Observation(content=result, operation='search_aae')

    def lookup(self, keyword: Annotated[str, 'The keyword to search for']) -> Observation:
        """
        Look up a word on the current page.
        """
        if self._document is None:
            return Observation(content='No document defined, cannot lookup', operation='lookup')

        text = self._document[0].lookup(keyword)
        if text:
            result = (
                    f'Keyword "{keyword}" found in current page in '
                    f'{len(self._document[0].lookup_results)} places. '
                    f'The first occurrence:\n{text}'
                    )
            url = self._document[1]
            return Observation(content=result, source=url, operation='lookup', quotable=True)
        else:
            return Observation(content=f'Keyword "{keyword}" not found on current page', operation='lookup')

    def lookup_next(self) -> Observation:
        """
        Jumps to the next occurrence of the word searched previously.
        """
        if self._document is None:
            return Observation(content='No document defined, cannot lookup', operation='lookup_next')
        if not self._document[0].lookup_results:
            return Observation(content='No lookup results found', operation='lookup_next')
        text = self._document[0].next_lookup()
        result = (
                f'Keyword "{self._document[0].lookup_word}" found in: \n'
                f'{text}\n'
                f'{self._document[0].lookup_position} of {len(self._document[0].lookup_results)}'
                )
        return Observation(content=result, source=self._document[1], operation='lookup_next', quotable=True)

    def read_chunk(self) -> Observation:
        """
        Reads the next chunk of text from the current location in the current document.
        """
        if self._document is None:
            return Observation(content='No document defined, cannot read', operation='read_chunk')
        return Observation(
                content=self._document[0].read_chunk(),
                source=self._document[1],
                operation='read_chunk',
                quotable=True
                )

    def goto_url(self, url: Annotated[str, "The url to go to"] ) -> Observation:
        """
        Retreive a page at url and saves the retrieved page as the next current page
        """
        response = requests.get(url)
        if response.status_code == 404:
            return Observation(content='Page does not Exist', operation='goto_url')
        try:
            response.raise_for_status()
        except:
            return Observation(content='Could not open the page', operation='goto_url')
        html = response.text
        cleaned_content = clean_html_and_textify(html)
        document = MarkdownDocument(cleaned_content, max_size=self._chunk_size, min_size=self._min_chunk_size)
        self._document = (document, url)
        return Observation(content=f'{url} retrieved successfully', operation='goto_url')

def _aae_search(query: str, base_url: str = BASE_URL) -> str:
    params = {
            's': query,
            'et_pb_searchform_submit': 'et_search_proccess',
            'et_pb_include_posts': 'yes',
            'et_pb_include_pages': 'yes',
            }
    response = requests.get(base_url, params=params)
    if response.status_code == 404:
        return 'page not found'
    response.raise_for_status()
    parsed = _aae_search_parse(response.text)
    if len(parsed) == 0:
        # This error message is copied from the page
        # when we make it more generic we'll need to extract the error message from the page dynamically
        return 'No Results Found\n\nThe page you requested could not be found. Try refining your search, or use the navigation above to locate the post.'


    result = '\n\n'.join(f'Title: {title}\nUrl: {url}\n Excerpt: {excerpt}'
                         for url, title, excerpt in parsed)
    return result


def _aae_search_parse(html: str):
    soup = BeautifulSoup(html, features='html.parser')
    post_blocks = soup.find_all('article', class_='et_pb_post')

    result = []

    for block in post_blocks:
        block: PageElement
        title_block = block.find_next('h2', class_ = 'entry-title')
        assert title_block
        anchor = title_block.find_next('a')
        assert isinstance(anchor, Tag)
        url = anchor.get('href')
        title = title_block.text.replace('\n', '')

        post_block = block.find_next('span', class_='excerpt_part')
        assert post_block is not None
        post_text = post_block.text
        
        result.append((url, title, post_text))

    return result
    

def clean_html_and_textify(html):
    # remove table of content
    soup = BeautifulSoup(html, 'html.parser')
    content = soup.find('div', class_='et_pb_post_content')


    converter = html2text.HTML2Text()
    # Avoid breaking links into newlines
    converter.body_width = 0
    # converter.protect_links = True # this does not seem to work
    markdown = converter.handle(content.text)
    cleaned_content = markdown.strip()
    return cleaned_content


if __name__ == "__main__":
    scraper = AAESearch()

    query="training generative AI"
    #query="generative AI training steps"
    #query="generative AI training authorization"
    query="authorization of generative AI training"
    query="generative AI training authorization"

    print(scraper.search_aae(query))